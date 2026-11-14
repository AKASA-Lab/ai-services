"""Train and evaluate EfficientNet-B0 on the shared split files (Person 2).

For every (seed, fold) run: train on ``train`` rows, pick the best epoch on
``validation`` rows (macro F1, then loss), predict ``test`` rows once and append
them to a prediction CSV in the common format.

Example (all 12 runs)::

    python -m models.efficientnet_b0.train --output-dir outputs/efficientnet_b0

Optional ``--weights`` points at a local copy of the timm ImageNet weights
for machines without Hugging Face access.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import f1_score
from torch import nn
from torchvision.transforms import InterpolationMode
from torchvision.transforms import v2 as T
from torchvision.transforms.v2 import functional as TF

from lettuce_benchmark import CLASS_NAMES
from lettuce_benchmark.benchmarking import checkpoint_size_bytes, inference_latency, parameter_count
from lettuce_benchmark.config import load_config
from lettuce_benchmark.evaluation import PREDICTION_COLUMNS
from lettuce_benchmark.preprocessing import watermark_safe_crop
from lettuce_benchmark.reproducibility import experiment_metadata, set_seed

LABEL_TO_INDEX = {label: index for index, label in enumerate(CLASS_NAMES)}


# --------------------------------------------------------------------------- data
def load_image_cache(paths: list[str], image_root: Path, crop: int, size: int, cache_file: Path) -> dict[str, torch.Tensor]:
    """Crop + resize each source image once and keep it as a uint8 tensor (3 x size x size)."""
    cache: dict[str, torch.Tensor] = {}
    if cache_file.is_file():
        stored = torch.load(cache_file, weights_only=False)
        if stored.get("crop") == crop and stored.get("size") == size:
            cache = stored["images"]
    missing = [path for path in paths if path not in cache]
    for path in missing:
        with Image.open(image_root / path) as image:
            image = watermark_safe_crop(image.convert("RGB"), crop)
            image = image.resize((size, size), Image.Resampling.BICUBIC)
            cache[path] = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).contiguous()
    if missing:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"crop": crop, "size": size, "images": cache}, cache_file)
    return cache


class BorderFillRotation(nn.Module):
    """Small random rotation; uncovered corners take the image's own mean border colour."""

    def __init__(self, degrees: float) -> None:
        super().__init__()
        self.degrees = degrees

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if self.degrees <= 0:
            return image
        angle = float(torch.empty(1).uniform_(-self.degrees, self.degrees))
        border = torch.cat([image[:, 0, :], image[:, -1, :], image[:, :, 0], image[:, :, -1]], dim=1)
        fill = border.float().mean(dim=1).round().tolist()
        return TF.rotate(image, angle, interpolation=InterpolationMode.BILINEAR, fill=fill)


class SplitDataset(torch.utils.data.Dataset):
    def __init__(self, rows: pd.DataFrame, cache: dict[str, torch.Tensor], transform: nn.Module) -> None:
        self.paths = rows["image_path"].tolist()
        self.targets = [LABEL_TO_INDEX[label] for label in rows["label"]]
        self.cache = cache
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return self.transform(self.cache[self.paths[index]]), self.targets[index]


def build_transforms(config: dict) -> tuple[nn.Module, nn.Module]:
    data, aug = config["data"], config["augmentation"]
    normalize = [T.ToDtype(torch.float32, scale=True), T.Normalize(data["mean"], data["std"])]
    train_steps: list[nn.Module] = [BorderFillRotation(aug["rotation_degrees"])]
    train_steps.append(
        T.RandomResizedCrop(
            data["input_size"],
            scale=tuple(aug["random_resized_crop_scale"]),
            ratio=tuple(aug["random_resized_crop_ratio"]),
            antialias=True,
        )
    )
    if aug["horizontal_flip"]:
        train_steps.append(T.RandomHorizontalFlip())
    train = T.Compose(train_steps + normalize)
    evaluation = T.Compose([T.Resize(data["input_size"], antialias=True)] + normalize)
    return train, evaluation


# --------------------------------------------------------------------------- model
def create_model(config: dict, weights: Path | None) -> nn.Module:
    import timm

    model_config = config["model"]
    kwargs = {
        "pretrained": model_config["pretrained"],
        "num_classes": model_config["num_classes"],
        "drop_rate": model_config.get("drop_rate", 0.2),
    }
    name = model_config["timm_name"]
    if weights is not None:
        kwargs["pretrained_cfg_overlay"] = {"file": str(weights)}
    else:
        name = f"{name}.{model_config['pretrained_tag']}"
    model = timm.create_model(name, **kwargs)
    nn.init.normal_(model.classifier.weight, 0.0, 0.01)
    nn.init.zeros_(model.classifier.bias)
    return model


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    for name, parameter in model.named_parameters():
        if not name.startswith("classifier."):
            parameter.requires_grad = trainable


def freeze_backbone_batchnorm(model: nn.Module) -> None:
    """While the backbone is frozen, keep its BatchNorm statistics fixed too."""
    for name, module in model.named_modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm) and not name.startswith("classifier"):
            module.eval()


@torch.inference_mode()
def predict(model: nn.Module, loader: torch.utils.data.DataLoader, device: torch.device,
            criterion: nn.Module) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    probabilities, targets, total_loss = [], [], 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        total_loss += criterion(logits, labels).item() * len(labels)
        probabilities.append(F.softmax(logits, dim=1).cpu().numpy())
        targets.append(labels.cpu().numpy())
    return np.concatenate(probabilities), np.concatenate(targets), total_loss / len(loader.dataset)


def macro_f1(targets: np.ndarray, predictions: np.ndarray) -> float:
    return float(f1_score(targets, predictions, labels=list(range(len(CLASS_NAMES))),
                          average="macro", zero_division=0))


# --------------------------------------------------------------------------- one run
def run_one(split: pd.DataFrame, seed: int, fold: int, config: dict, cache: dict[str, torch.Tensor],
            weights: Path | None, device: torch.device, run_dir: Path, checkpoint_dir: Path) -> tuple[pd.DataFrame, dict]:
    set_seed(seed)
    train_cfg = config["training"]
    train_rows = split[split["partition"] == "train"]
    val_rows = split[split["partition"] == "validation"]
    test_rows = split[split["partition"] == "test"]
    train_tf, eval_tf = build_transforms(config)
    generator = torch.Generator().manual_seed(seed)
    loader_kwargs = {"batch_size": train_cfg["batch_size"], "num_workers": 0}
    train_loader = torch.utils.data.DataLoader(SplitDataset(train_rows, cache, train_tf), shuffle=True,
                                               generator=generator, **loader_kwargs)
    val_loader = torch.utils.data.DataLoader(SplitDataset(val_rows, cache, eval_tf), shuffle=False, **loader_kwargs)
    test_loader = torch.utils.data.DataLoader(SplitDataset(test_rows, cache, eval_tf), shuffle=False, **loader_kwargs)

    counts = train_rows["label"].value_counts()
    class_weights = torch.tensor(
        [len(train_rows) / (len(CLASS_NAMES) * counts[label]) for label in CLASS_NAMES], dtype=torch.float32
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    model = create_model(config, weights).to(device)
    classifier_params = [p for n, p in model.named_parameters() if n.startswith("classifier.")]
    backbone_params = [p for n, p in model.named_parameters() if not n.startswith("classifier.")]
    optimizer = torch.optim.AdamW(
        [{"params": backbone_params, "lr": train_cfg["backbone_lr"]},
         {"params": classifier_params, "lr": train_cfg["classifier_lr"]}],
        weight_decay=train_cfg["weight_decay"],
    )

    history, best = [], {"key": (-math.inf, -math.inf)}
    started = time.perf_counter()
    for epoch in range(1, train_cfg["max_epochs"] + 1):
        frozen = epoch <= train_cfg["frozen_backbone_epochs"]
        set_backbone_trainable(model, not frozen)
        model.train()
        if frozen:
            freeze_backbone_batchnorm(model)
        running_loss, seen = 0.0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(labels)
            seen += len(labels)
        val_prob, val_true, val_loss = predict(model, val_loader, device, criterion)
        val_f1 = macro_f1(val_true, val_prob.argmax(1))
        record = {"epoch": epoch, "backbone_frozen": frozen, "train_loss": running_loss / seen,
                  "val_loss": val_loss, "val_macro_f1": val_f1,
                  "val_accuracy": float((val_prob.argmax(1) == val_true).mean())}
        history.append(record)
        key = (val_f1, -val_loss)
        if key > best["key"]:
            best = {"key": key, "epoch": epoch, "state": copy.deepcopy(model.state_dict())}
        print(f"seed={seed} fold={fold} epoch={epoch:02d} frozen={frozen} train_loss={record['train_loss']:.4f} "
              f"val_loss={val_loss:.4f} val_macro_f1={val_f1:.4f}", flush=True)
    training_seconds = time.perf_counter() - started

    model.load_state_dict(best["state"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"efficientnet_b0_seed_{seed}_fold_{fold}.pt"
    torch.save(best["state"], checkpoint_path)

    test_prob, test_true, _ = predict(model, test_loader, device, criterion)
    predicted = test_prob.argmax(1)
    predictions = pd.DataFrame({
        "image_id": test_rows["image_id"].to_numpy(),
        "seed": seed,
        "fold": fold,
        "true_label": [CLASS_NAMES[i] for i in test_true],
        "predicted_label": [CLASS_NAMES[i] for i in predicted],
        "confidence": test_prob.max(1).round(6),
    })[PREDICTION_COLUMNS]

    latency = inference_latency(model, torch.zeros(1, 3, config["data"]["input_size"], config["data"]["input_size"]),
                                device, warmup=config["efficiency"]["latency_warmup"],
                                iterations=config["efficiency"]["latency_iterations"])
    summary = {
        "seed": seed, "fold": fold, "best_epoch": best["epoch"],
        "best_val_macro_f1": best["key"][0], "best_val_loss": -best["key"][1],
        "test_macro_f1_this_fold": macro_f1(test_true, predicted),
        "test_accuracy_this_fold": float((predicted == test_true).mean()),
        "n_train": len(train_rows), "n_validation": len(val_rows), "n_test": len(test_rows),
        "class_weights": dict(zip(CLASS_NAMES, class_weights.tolist())),
        "training_seconds": training_seconds,
        "parameter_count": parameter_count(model),
        "checkpoint_bytes": checkpoint_size_bytes(checkpoint_path),
        "latency_ms_batch1": latency["mean_milliseconds"],
        "torch_threads": torch.get_num_threads(),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    with (run_dir / "run.json").open("w", encoding="utf-8") as stream:
        json.dump({**experiment_metadata("efficientnet_b0", seed, fold, config), "result": summary}, stream, indent=2)
    return predictions, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("models/efficientnet_b0/config.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/efficientnet_b0"))
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--splits-dir", type=Path, default=None)
    parser.add_argument("--weights", type=Path, default=None, help="local timm weight file (optional)")
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--folds", type=int, nargs="*", default=[1, 2, 3, 4])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = load_config(args.config)
    image_root = args.image_root or Path(config["data"]["image_root"])
    splits_dir = args.splits_dir or Path(config["data"]["splits_dir"])
    seeds = args.seeds or sorted(int(p.name.split("_")[1]) for p in splits_dir.glob("seed_*"))
    device = torch.device(args.device)
    config["runtime"] = {"device": str(device), "weights_file": str(args.weights) if args.weights else None}

    first = pd.read_csv(splits_dir / f"seed_{seeds[0]}" / "fold_1.csv")
    cache = load_image_cache(sorted(first["image_path"]), image_root, config["data"]["safe_crop"],
                             config["data"]["cache_size"], args.output_dir / "image_cache.pt")
    predictions_path = args.output_dir / "predictions.csv"
    summaries_path = args.output_dir / "runs.csv"
    for seed in seeds:
        for fold in args.folds:
            split = pd.read_csv(splits_dir / f"seed_{seed}" / f"fold_{fold}.csv", dtype={"image_id": str})
            predictions, summary = run_one(split, seed, fold, config, cache, args.weights, device,
                                           args.output_dir / "runs" / f"seed_{seed}" / f"fold_{fold}",
                                           args.output_dir / "checkpoints")
            for frame, path in ((predictions, predictions_path), (pd.DataFrame([summary]).drop(columns="class_weights"), summaries_path)):
                if path.is_file():
                    existing = pd.read_csv(path, dtype={"image_id": str})
                    existing = existing[~((existing["seed"] == seed) & (existing["fold"] == fold))]
                    frame = pd.concat([existing, frame], ignore_index=True)
                frame.sort_values(["seed", "fold"] + (["image_id"] if "image_id" in frame else [])).to_csv(path, index=False)
            print(f"DONE seed={seed} fold={fold} best_epoch={summary['best_epoch']} "
                  f"test_macro_f1={summary['test_macro_f1_this_fold']:.4f} time={summary['training_seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
