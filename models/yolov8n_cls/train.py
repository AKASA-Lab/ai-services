"""Train and evaluate YOLOv8n-cls on the shared split files (Person 4).

For every (seed, fold) run:

1. write the run's ``train`` / ``validation`` rows as an Ultralytics classification folder
   (``prepare_data.py``; the ``test`` rows are kept out of it);
2. fine-tune the pretrained ``yolov8n-cls.pt`` with the Ultralytics trainer, validating and saving
   a checkpoint after every epoch;
3. pick the epoch with the best validation macro F1 (ties: lower validation loss, then earlier);
4. predict the held-out ``test`` rows once with that checkpoint and append them to a prediction
   CSV in the common format.

Example (all 12 runs)::

    python -m models.yolov8n_cls.train --output-dir outputs/yolov8n_cls
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

from lettuce_benchmark import CLASS_NAMES
from lettuce_benchmark.benchmarking import checkpoint_size_bytes, inference_latency, parameter_count
from lettuce_benchmark.config import load_config
from lettuce_benchmark.evaluation import PREDICTION_COLUMNS
from lettuce_benchmark.reproducibility import experiment_metadata, set_seed
from models.yolov8n_cls.prepare_data import build_split_dataset, cached_path, preprocess_images

MODEL_NAME = "yolov8n_cls"


# --------------------------------------------------------------------------- pure helpers (unit-tested)
def macro_f1(targets: np.ndarray, predictions: np.ndarray, n_classes: int = len(CLASS_NAMES)) -> float:
    return float(f1_score(targets, predictions, labels=list(range(n_classes)), average="macro", zero_division=0))


def select_best_epoch(history: pd.DataFrame) -> pd.Series:
    """Row with the highest ``val_macro_f1``; ties go to lower ``val_loss``, then the earlier epoch."""
    ranked = history.sort_values(["val_macro_f1", "val_loss", "epoch"], ascending=[False, True, True], kind="stable")
    return ranked.iloc[0]


def to_canonical_order(probabilities: np.ndarray, model_names: dict[int, str]) -> np.ndarray:
    """Reorder probability columns from the model's class index order to ``CLASS_NAMES`` order.

    Ultralytics numbers classes by sorted folder name (``-K, -N, -P, FN``), not by the project's
    canonical order, so columns are matched by name rather than by position.
    """
    if sorted(model_names.values()) != sorted(CLASS_NAMES):
        raise ValueError(f"Model classes {sorted(model_names.values())} do not match {list(CLASS_NAMES)}")
    by_name = {name: index for index, name in model_names.items()}
    return probabilities[:, [by_name[name] for name in CLASS_NAMES]]


def read_history(results_csv: Path) -> pd.DataFrame:
    """Per-epoch history from the Ultralytics ``results.csv`` (with our extra macro-F1 column)."""
    frame = pd.read_csv(results_csv)
    frame.columns = [column.strip() for column in frame.columns]
    history = pd.DataFrame({
        "epoch": frame["epoch"].astype(int),
        "train_loss": frame["train/loss"],
        "val_loss": frame["val/loss"],
        "val_macro_f1": frame["metrics/macro_f1"],
        "val_accuracy": frame["metrics/accuracy_top1"],
        "lr": frame["lr/pg0"],
    })
    # Ultralytics appends a row when a run is resumed; keep the last entry per epoch.
    return history.drop_duplicates("epoch", keep="last").reset_index(drop=True)


# --------------------------------------------------------------------------- ultralytics glue
def _ultralytics():
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.models.yolo.classify import ClassificationTrainer
    from ultralytics.utils import events

    events.events.enabled = False  # no usage analytics from benchmark runs
    return ultralytics, YOLO, ClassificationTrainer


def macro_f1_trainer_class():
    """Ultralytics ClassificationTrainer whose validation also reports macro F1 and uses it as fitness."""
    _, _, ClassificationTrainer = _ultralytics()

    class MacroF1ClassificationTrainer(ClassificationTrainer):
        def validate(self):
            metrics = self.validator(self)
            if metrics is None:
                return None, None
            metrics.pop("fitness", None)
            predicted = torch.cat(self.validator.pred)[:, 0].numpy()
            targets = torch.cat(self.validator.targets).numpy()
            fitness = macro_f1(targets, predicted, n_classes=self.data["nc"])
            metrics["metrics/macro_f1"] = round(fitness, 5)
            if self.best_fitness is None or self.best_fitness < fitness:
                self.best_fitness = fitness
            return metrics, fitness

    return MacroF1ClassificationTrainer


def ensure_weights(name: str, weights_dir: Path) -> Path:
    """Download the pretrained Ultralytics weights once into ``weights_dir`` (not the repo root)."""
    from ultralytics.utils.downloads import attempt_download_asset

    weights_dir.mkdir(parents=True, exist_ok=True)
    target = weights_dir / name
    if not target.is_file():
        attempt_download_asset(target)
    if not target.is_file():
        raise FileNotFoundError(f"Could not obtain pretrained weights {name}")
    return target


def predict_paths(model: Any, paths: list[Path], imgsz: int, device: str) -> np.ndarray:
    """Class probabilities (canonical ``CLASS_NAMES`` column order) via the Ultralytics predictor."""
    rows = []
    for result in model.predict(source=[str(p) for p in paths], imgsz=imgsz, device=device, verbose=False,
                                stream=True, batch=16):
        rows.append(result.probs.data.float().cpu().numpy())
    return to_canonical_order(np.stack(rows), model.names)


# --------------------------------------------------------------------------- one run
def run_one(split: pd.DataFrame, seed: int, fold: int, config: dict, cache_root: Path, weights: Path,
            device: str, output_dir: Path) -> tuple[pd.DataFrame, dict]:
    ultralytics, YOLO, _ = _ultralytics()
    set_seed(seed)
    train_rows = split[split["partition"] == "train"]
    val_rows = split[split["partition"] == "validation"]
    test_rows = split[split["partition"] == "test"]
    run_name = f"seed_{seed}_fold_{fold}"
    data_dir = build_split_dataset(split, cache_root, output_dir / "datasets" / run_name)

    train_args = {**config["training"], "data": str(data_dir), "seed": seed, "device": device,
                  "project": str((output_dir / "ultralytics").resolve()), "name": run_name, "exist_ok": True}
    ultra_dir = output_dir / "ultralytics" / run_name
    if ultra_dir.exists():
        shutil.rmtree(ultra_dir)  # never mix epochs from an earlier attempt into model selection

    model = YOLO(str(weights))
    started = time.perf_counter()
    model.train(trainer=macro_f1_trainer_class(), **train_args)
    training_seconds = time.perf_counter() - started

    history = read_history(ultra_dir / "results.csv")
    if len(history) != config["training"]["epochs"]:
        raise RuntimeError(f"Expected {config['training']['epochs']} epochs, found {len(history)} in results.csv")
    best = select_best_epoch(history)
    best_epoch = int(best["epoch"])
    epoch_checkpoint = ultra_dir / "weights" / f"epoch{best_epoch - 1}.pt"  # Ultralytics names them 0-based
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{MODEL_NAME}_seed_{seed}_fold_{fold}.pt"
    from ultralytics.utils.torch_utils import get_cpu_info, strip_optimizer

    shutil.copy2(epoch_checkpoint, checkpoint_path)
    strip_optimizer(checkpoint_path)
    for extra in (ultra_dir / "weights").glob("epoch*.pt"):
        extra.unlink()  # 15 per run; the selected one is kept in checkpoints/

    selected = YOLO(str(checkpoint_path))
    imgsz = config["training"]["imgsz"]

    # Consistency check: the selected checkpoint, run through the predictor, must reproduce the
    # validation macro F1 that the trainer logged for that epoch.
    val_prob = predict_paths(selected, [cached_path(cache_root, p) for p in val_rows["image_path"]], imgsz, device)
    val_true = np.array([CLASS_NAMES.index(label) for label in val_rows["label"]])
    val_f1_recomputed = macro_f1(val_true, val_prob.argmax(1))

    test_prob = predict_paths(selected, [cached_path(cache_root, p) for p in test_rows["image_path"]], imgsz, device)
    test_true = np.array([CLASS_NAMES.index(label) for label in test_rows["label"]])
    predicted = test_prob.argmax(1)
    predictions = pd.DataFrame({
        "image_id": test_rows["image_id"].to_numpy(),
        "seed": seed,
        "fold": fold,
        "true_label": test_rows["label"].to_numpy(),
        "predicted_label": [CLASS_NAMES[i] for i in predicted],
        "confidence": test_prob.max(1).round(6),
    })[PREDICTION_COLUMNS]

    # Efficiency: batch-1 latency of the bare PyTorch network on CPU with a fixed thread count.
    network = selected.model.float().eval()
    effcfg = config["efficiency"]
    threads = torch.get_num_threads()
    torch.set_num_threads(effcfg["latency_threads"])
    try:
        latency = inference_latency(network, torch.zeros(1, 3, imgsz, imgsz), torch.device(effcfg["latency_device"]),
                                    warmup=effcfg["latency_warmup"], iterations=effcfg["latency_iterations"])
    finally:
        torch.set_num_threads(threads)
    fp32_state = checkpoint_dir / f".{run_name}_fp32_state_dict.pt"
    torch.save(network.state_dict(), fp32_state)
    fp32_bytes = checkpoint_size_bytes(fp32_state)
    fp32_state.unlink()

    # Same formula as Ultralytics' cls_pw=1: inverse train-partition frequency, normalised to mean 1.
    inverse = 1.0 / train_rows["label"].value_counts().reindex(list(CLASS_NAMES)).to_numpy(dtype=float)
    class_weights = dict(zip(CLASS_NAMES, (inverse / inverse.mean()).round(4).tolist()))
    summary = {
        "seed": seed, "fold": fold, "best_epoch": best_epoch,
        "best_val_macro_f1": float(best["val_macro_f1"]), "best_val_loss": float(best["val_loss"]),
        "val_macro_f1_recomputed": val_f1_recomputed,
        "test_macro_f1_this_fold": macro_f1(test_true, predicted),
        "test_accuracy_this_fold": float((predicted == test_true).mean()),
        "n_train": len(train_rows), "n_validation": len(val_rows), "n_test": len(test_rows),
        "training_seconds": training_seconds,
        "parameter_count": parameter_count(network),
        "checkpoint_bytes": checkpoint_size_bytes(checkpoint_path),
        "fp32_state_dict_bytes": fp32_bytes,
        "latency_ms_batch1": latency["mean_milliseconds"],
        "latency_threads": effcfg["latency_threads"],
        "training_threads": threads,
    }
    run_dir = output_dir / "runs" / f"seed_{seed}" / f"fold_{fold}"
    run_dir.mkdir(parents=True, exist_ok=True)
    history.to_csv(run_dir / "history.csv", index=False)
    metadata = experiment_metadata(MODEL_NAME, seed, fold, config)
    metadata["ultralytics_version"] = ultralytics.__version__
    metadata["cpu_name"] = get_cpu_info()
    metadata["ultralytics_train_args"] = {k: (str(v) if isinstance(v, Path) else v) for k, v in train_args.items()}
    metadata["class_weights"] = class_weights
    with (run_dir / "run.json").open("w", encoding="utf-8") as stream:
        json.dump({**metadata, "result": summary}, stream, indent=2, default=str)
    return predictions, summary


def _append(frame: pd.DataFrame, path: Path, seed: int, fold: int) -> None:
    """Replace this run's rows in a CSV that accumulates all runs."""
    if path.is_file():
        existing = pd.read_csv(path, dtype={"image_id": str})
        existing = existing[~((existing["seed"] == seed) & (existing["fold"] == fold))]
        frame = pd.concat([existing, frame], ignore_index=True)
    frame.sort_values(["seed", "fold"] + (["image_id"] if "image_id" in frame else [])).to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("models/yolov8n_cls/config.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/yolov8n_cls"))
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--splits-dir", type=Path, default=None)
    parser.add_argument("--weights", type=Path, default=None, help="local yolov8n-cls.pt (optional)")
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--folds", type=int, nargs="*", default=[1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=None, help="override (smoke tests only)")
    parser.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu",
                        help="Ultralytics device string: cpu, 0, mps")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    image_root = args.image_root or Path(config["data"]["image_root"])
    splits_dir = args.splits_dir or Path(config["data"]["splits_dir"])
    seeds = args.seeds or sorted(int(p.name.split("_")[1]) for p in splits_dir.glob("seed_*"))
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    weights = args.weights or ensure_weights(config["model"]["weights"], output_dir / "pretrained")
    config["runtime"] = {"device": args.device, "weights_file": str(weights)}

    first = pd.read_csv(splits_dir / f"seed_{seeds[0]}" / "fold_1.csv")
    cache_root = preprocess_images(sorted(first["image_path"]), image_root, output_dir / "preprocessed",
                                   config["data"]["safe_crop"], config["data"]["cache_size"])
    for seed in seeds:
        for fold in args.folds:
            split = pd.read_csv(splits_dir / f"seed_{seed}" / f"fold_{fold}.csv", dtype={"image_id": str})
            predictions, summary = run_one(split, seed, fold, config, cache_root, weights, args.device, output_dir)
            _append(predictions, output_dir / "predictions.csv", seed, fold)
            _append(pd.DataFrame([summary]), output_dir / "runs.csv", seed, fold)
            print(f"DONE seed={seed} fold={fold} best_epoch={summary['best_epoch']} "
                  f"val_macro_f1={summary['best_val_macro_f1']:.4f} (recomputed {summary['val_macro_f1_recomputed']:.4f}) "
                  f"test_macro_f1={summary['test_macro_f1_this_fold']:.4f} time={summary['training_seconds']:.0f}s",
                  flush=True)


if __name__ == "__main__":
    main()
