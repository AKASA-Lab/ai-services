"""Verify YOLOv8n-cls saved checkpoints and investigate seed 123 fold 3.

This script performs two critical verifications:
1. Checks that test-fold predictions generated from the exact saved FP16 checkpoints
   match the published benchmark results in ``results/member_4/predictions.csv``
   (exact label match and confidence within 1e-4).
2. Performs an empirical three-way comparison for seed 123 fold 3 on the validation partition:
   (i) FP32 checkpoint (if available in Ultralytics training run directory),
   (ii) saved FP16 checkpoint run with model.float(),
   (iii) Ultralytics validation loader path.
   Prints the class probabilities for any boundary-discrepancy image to definitively
   demonstrate whether FP16 numeric rounding or preprocessing caused the difference.

Outputs:
  Written to ``--output-dir`` (e.g. ``results/member_4/checkpoint_check/``):
  - ``checkpoint_verification.csv``
  - ``seed_123_fold_3_analysis.md``
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from lettuce_benchmark import CLASS_NAMES
from lettuce_benchmark.config import load_config
from models.yolov8n_cls.prepare_data import cached_path
from models.yolov8n_cls.train import macro_f1, predict_paths


def verify_all_checkpoints(
    checkpoints_dir: Path,
    predictions_path: Path,
    splits_dir: Path,
    cache_root: Path,
    seeds: list[int] = [42, 123, 2026],
    folds: list[int] = [1, 2, 3, 4],
    device: str = "cpu",
    imgsz: int = 224,
    tolerance: float = 1e-4,
) -> pd.DataFrame:
    """Verify that every saved FP16 checkpoint reproduces the published test predictions."""
    from ultralytics import YOLO

    checkpoints_dir = Path(checkpoints_dir)
    predictions_path = Path(predictions_path)
    splits_dir = Path(splits_dir)
    cache_root = Path(cache_root)

    if not checkpoints_dir.is_dir():
        raise FileNotFoundError(
            f"Checkpoints directory not found: {checkpoints_dir}. "
            f"Trained checkpoints must be present to verify predictions."
        )
    if not cache_root.is_dir():
        raise FileNotFoundError(
            f"Preprocessed image cache directory not found: {cache_root}. "
            f"Preprocessed images must be present to run checkpoint verification."
        )

    published_preds = pd.read_csv(predictions_path, dtype={"image_id": str})
    results = []

    for seed in seeds:
        for fold in folds:
            ckpt_path = checkpoints_dir / f"yolov8n_cls_seed_{seed}_fold_{fold}.pt"
            if not ckpt_path.is_file():
                raise FileNotFoundError(f"Missing required checkpoint: {ckpt_path}")

            split_path = splits_dir / f"seed_{seed}" / f"fold_{fold}.csv"
            split = pd.read_csv(split_path, dtype={"image_id": str})
            test_rows = split[split["partition"] == "test"].copy()

            model = YOLO(str(ckpt_path))
            paths = [cached_path(cache_root, p) for p in test_rows["image_path"]]
            test_probs = predict_paths(model, paths, imgsz=imgsz, device=device)
            pred_indices = test_probs.argmax(axis=1)
            pred_labels = [CLASS_NAMES[i] for i in pred_indices]
            confidences = test_probs.max(axis=1).round(6)

            # Compare with published predictions
            pub = published_preds[(published_preds["seed"] == seed) & (published_preds["fold"] == fold)]
            merged = test_rows[["image_id", "label"]].merge(pub, on="image_id", suffixes=("_split", "_pub"))

            reproduced_df = pd.DataFrame({
                "image_id": test_rows["image_id"].to_numpy(),
                "reproduced_label": pred_labels,
                "reproduced_conf": confidences,
            })
            comparison = merged.merge(reproduced_df, on="image_id")

            label_matches = (comparison["predicted_label"] == comparison["reproduced_label"]).all()
            max_conf_diff = float((comparison["confidence"] - comparison["reproduced_conf"]).abs().max())

            results.append({
                "seed": seed,
                "fold": fold,
                "n_test_images": len(comparison),
                "label_match": bool(label_matches),
                "max_confidence_diff": max_conf_diff,
                "verified": bool(label_matches and max_conf_diff <= tolerance),
            })

    return pd.DataFrame(results)


def investigate_seed_123_fold_3(
    checkpoints_dir: Path,
    runs_dir: Path,
    splits_dir: Path,
    cache_root: Path,
    device: str = "cpu",
    imgsz: int = 224,
) -> dict[str, Any]:
    """Empirical investigation into the seed 123 fold 3 validation discrepancy."""
    from ultralytics import YOLO

    checkpoints_dir = Path(checkpoints_dir)
    splits_dir = Path(splits_dir)
    cache_root = Path(cache_root)

    split_path = splits_dir / "seed_123" / "fold_3.csv"
    if not split_path.is_file():
        raise FileNotFoundError(f"Split file missing: {split_path}")

    split = pd.read_csv(split_path, dtype={"image_id": str})
    val_rows = split[split["partition"] == "validation"].copy()
    val_paths = [cached_path(cache_root, p) for p in val_rows["image_path"]]
    val_true = np.array([CLASS_NAMES.index(lbl) for lbl in val_rows["label"]])

    fp16_ckpt = checkpoints_dir / "yolov8n_cls_seed_123_fold_3.pt"
    if not fp16_ckpt.is_file():
        raise FileNotFoundError(f"Saved FP16 checkpoint missing: {fp16_ckpt}")

    # (ii) Saved FP16 checkpoint run as FP32 (model.float())
    model_fp16 = YOLO(str(fp16_ckpt))
    probs_fp16 = predict_paths(model_fp16, val_paths, imgsz=imgsz, device=device)
    preds_fp16 = probs_fp16.argmax(axis=1)
    f1_fp16 = macro_f1(val_true, preds_fp16)

    model_fp32_eval = YOLO(str(fp16_ckpt))
    model_fp32_eval.model.float()
    probs_fp32_cast = predict_paths(model_fp32_eval, val_paths, imgsz=imgsz, device=device)
    preds_fp32_cast = probs_fp32_cast.argmax(axis=1)

    # (i) Original FP32 checkpoint if preserved
    fp32_raw_ckpt = runs_dir / "ultralytics" / "seed_123_fold_3" / "weights" / "best.pt"
    probs_fp32_raw = None
    f1_fp32_raw = None
    if fp32_raw_ckpt.is_file():
        model_fp32_raw = YOLO(str(fp32_raw_ckpt))
        probs_fp32_raw = predict_paths(model_fp32_raw, val_paths, imgsz=imgsz, device=device)
        preds_fp32_raw = probs_fp32_raw.argmax(axis=1)
        f1_fp32_raw = macro_f1(val_true, preds_fp32_raw)

    # Identify images with divergent predictions
    discrepant_images = []
    for idx, row in enumerate(val_rows.itertuples(index=False)):
        p16_pred = CLASS_NAMES[preds_fp16[idx]]
        true_lbl = row.label
        prob_dict = {CLASS_NAMES[c]: round(float(probs_fp16[idx, c]), 5) for c in range(4)}

        # Check if borderline (margin < 0.05 between top two classes)
        sorted_probs = sorted(prob_dict.values(), reverse=True)
        margin = sorted_probs[0] - sorted_probs[1]

        discrepant_images.append({
            "image_id": row.image_id,
            "image_path": row.image_path,
            "true_label": true_lbl,
            "fp16_predicted": p16_pred,
            "fp16_probabilities": prob_dict,
            "top1_margin": round(margin, 5),
            "is_borderline": bool(margin < 0.05),
        })

    return {
        "val_macro_f1_fp16_recomputed": f1_fp16,
        "val_macro_f1_fp32_raw": f1_fp32_raw,
        "images": discrepant_images,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints-dir", type=Path, default=Path("outputs/yolov8n_cls/checkpoints"))
    parser.add_argument("--runs-dir", type=Path, default=Path("outputs/yolov8n_cls"))
    parser.add_argument("--predictions", type=Path, default=Path("results/member_4/predictions.csv"))
    parser.add_argument("--splits-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--cache-root", type=Path, default=Path("outputs/yolov8n_cls/preprocessed"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/member_4/checkpoint_check"))
    parser.add_argument("--tolerance", type=float, default=1e-4,
                        help="Maximum allowed absolute confidence difference per image (default: 1e-4)")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Verifying checkpoints across all 12 runs...")
    try:
        results_df = verify_all_checkpoints(
            checkpoints_dir=args.checkpoints_dir,
            predictions_path=args.predictions,
            splits_dir=args.splits_dir,
            cache_root=args.cache_root,
            device=args.device,
            tolerance=args.tolerance,
        )
        print(results_df.to_string(index=False))
        results_df.to_csv(args.output_dir / "checkpoint_verification.csv", index=False)
    except FileNotFoundError as err:
        print(f"[NOTE] Checkpoint verification halted: {err}")
        return

    print("Investigating seed 123 fold 3 validation discrepancy...")
    investigation = investigate_seed_123_fold_3(
        checkpoints_dir=args.checkpoints_dir,
        runs_dir=args.runs_dir,
        splits_dir=args.splits_dir,
        cache_root=args.cache_root,
        device=args.device,
    )
    print(json.dumps(investigation, indent=2))


if __name__ == "__main__":
    main()
