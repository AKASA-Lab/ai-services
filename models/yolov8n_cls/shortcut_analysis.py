"""Quantitative shortcut-learning and background-leakage analysis for YOLOv8n-cls (Person 4).

Investigates whether the model attends to plant leaf tissue or exploits background, border,
or dataset-specific artifacts (particularly for healthy FN images, all of which belong
to the 1024x1024 acquisition group).

Metrics computed per held-out test image:
1. Border-band CAM mass: fraction of total CAM activation residing in the outer 15% frame band.
2. Leaf-mask CAM mass: fraction of CAM activation residing inside the HSV-segmented leaf tissue.
3. Background ablation control: non-leaf pixels are replaced with neutral gray (RGB 128, 128, 128),
   and the held-out fold model re-predicts the image. We evaluate the resulting change in class
   recall and macro-F1, specifically observing whether FN recall degrades when background cues
   are silenced.

Inference only — no retraining is performed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch

from lettuce_benchmark import CLASS_NAMES
from models.yolov8n_cls.gradcam import (
    YOLOv8GradCAM,
    load_holdout_model_for_image,
    resolve_checkpoint_path,
)
from models.yolov8n_cls.prepare_data import cached_path
from models.yolov8n_cls.train import macro_f1, predict_paths, to_canonical_order


def get_border_band_mask(size: int = 224, fraction: float = 0.15) -> np.ndarray:
    """Create a boolean mask for the outer border band of the image frame."""
    mask = np.ones((size, size), dtype=bool)
    b = int(fraction * size)
    mask[b:-b, b:-b] = False
    return mask


def get_leaf_mask_hsv(rgb_uint8: np.ndarray) -> np.ndarray:
    """Segment leaf lamina based on green/yellow/plant HSV thresholds."""
    hsv = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    # Green and yellowish-green foliage: Hue ~18 to 95, Saturation >= 25, Value >= 30
    leaf = (h >= 18) & (h <= 95) & (s >= 25) & (v >= 30)
    return leaf


def analyze_shortcuts(
    predictions_path: Path,
    audit_path: Path,
    checkpoints_dir: Path,
    cache_root: Path,
    output_dir: Path,
    seed: int = 42,
    imgsz: int = 224,
    device: str = "cpu",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Execute shortcut-learning analysis and neutral-background ablation control."""
    from ultralytics import YOLO

    predictions_path = Path(predictions_path)
    audit_path = Path(audit_path)
    checkpoints_dir = Path(checkpoints_dir)
    cache_root = Path(cache_root)
    output_dir = Path(output_dir)

    if not checkpoints_dir.is_dir():
        raise FileNotFoundError(
            f"Checkpoints directory not found: {checkpoints_dir}. "
            f"Trained checkpoints must be present to run shortcut analysis."
        )
    if not cache_root.is_dir():
        raise FileNotFoundError(
            f"Preprocessed image cache directory not found: {cache_root}. "
            f"Preprocessed images must be present to run shortcut analysis."
        )

    preds = pd.read_csv(predictions_path, dtype={"image_id": str})
    audit = pd.read_csv(audit_path, dtype={"image_id": str})
    preds_seed = preds[preds["seed"] == int(seed)].copy()

    audit_cols = [c for c in ["image_id", "image_path", "width", "height", "dalle_colour_bar_detected"] if c in audit.columns]
    merged = preds_seed.merge(audit[audit_cols], on="image_id", how="left")
    merged["size_group"] = np.where(
        (merged["width"] == 1024) & (merged["height"] == 1024),
        "1024x1024",
        "other_sizes",
    )

    border_mask = get_border_band_mask(size=imgsz, fraction=0.15)
    border_area_frac = float(border_mask.mean())

    records = []
    models_cache: dict[int, Any] = {}

    # Load holdout fold checkpoints
    for fold in range(1, 5):
        ckpt_path = resolve_checkpoint_path(seed, fold, checkpoints_dir)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Missing checkpoint for fold {fold}: {ckpt_path}")
        models_cache[fold] = YOLO(str(ckpt_path))

    control_images_ablation = []
    control_rows_original = []

    print(f"Analyzing {len(merged)} test images for seed {seed}...")
    for row in merged.itertuples(index=False):
        rel_path = getattr(row, "image_path", None)
        if not rel_path or pd.isna(rel_path):
            rel_path = f"{row.true_label}/{row.image_id}.png"
        img_path = cached_path(cache_root, str(rel_path))
        if not img_path.is_file():
            raise FileNotFoundError(f"Cached image missing: {img_path}")

        # Load RGB image at 224x224
        with Image.open(img_path) as pil_img:
            resized = pil_img.resize((imgsz, imgsz), Image.Resampling.BICUBIC).convert("RGB")
            rgb_arr = np.array(resized)

        leaf_mask = get_leaf_mask_hsv(rgb_arr)
        leaf_area_frac = float(leaf_mask.mean())

        # Obtain fold model and Grad-CAM engine
        fold = int(row.fold)
        model = models_cache[fold]
        cam_engine = YOLOv8GradCAM(model)

        tensor = torch.from_numpy(rgb_arr).permute(2, 0, 1).float() / 255.0
        cam, probs, pred_idx = cam_engine(tensor, target_class=row.predicted_label)
        cam_engine.remove_hooks()

        cam_sum = float(cam.sum())
        if cam_sum > 0:
            cam_mass = cam / cam_sum
        else:
            cam_mass = np.full_like(cam, 1.0 / cam.size)

        cam_border_mass = float(cam_mass[border_mask].sum())
        cam_leaf_mass = float(cam_mass[leaf_mask].sum())

        records.append({
            "image_id": row.image_id,
            "seed": seed,
            "fold": fold,
            "true_label": row.true_label,
            "predicted_label": row.predicted_label,
            "confidence": row.confidence,
            "size_group": row.size_group,
            "width": row.width,
            "height": row.height,
            "dalle_colour_bar": getattr(row, "dalle_colour_bar_detected", False),
            "leaf_area_fraction": round(leaf_area_frac, 4),
            "border_area_fraction": round(border_area_frac, 4),
            "cam_mass_on_leaf": round(cam_leaf_mass, 4),
            "cam_mass_on_border_band": round(cam_border_mass, 4),
            "leaf_attention_density": round(cam_leaf_mass / max(leaf_area_frac, 1e-4), 3),
            "border_attention_density": round(cam_border_mass / border_area_frac, 3),
        })

        # Background ablation control: replace non-leaf pixels with neutral gray (128, 128, 128)
        gray_ablated = rgb_arr.copy()
        gray_ablated[~leaf_mask] = 128
        ablated_tensor = torch.from_numpy(gray_ablated).permute(2, 0, 1).float() / 255.0

        # Predict with same fold model
        with torch.no_grad():
            out = model.model(ablated_tensor.unsqueeze(0).to(next(model.model.parameters()).device))
            if isinstance(out, (tuple, list)):
                logits = out[1]
            else:
                logits = out
            probs_ablated = logits.softmax(-1).detach().cpu().numpy()
            probs_canonical = to_canonical_order(probs_ablated, model.names)[0]
            pred_ablated = CLASS_NAMES[int(probs_canonical.argmax())]
            conf_ablated = float(probs_canonical.max())

        control_rows_original.append({
            "image_id": row.image_id,
            "true_label": row.true_label,
            "pred_original": row.predicted_label,
            "pred_ablated": pred_ablated,
            "conf_ablated": round(conf_ablated, 4),
            "size_group": row.size_group,
        })

    metrics_df = pd.DataFrame(records)
    control_df = pd.DataFrame(control_rows_original)

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(output_dir / "shortcut_metrics.csv", index=False)
    control_df.to_csv(output_dir / "neutral_background_control.csv", index=False)

    return metrics_df, control_df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=Path("results/member_4/predictions.csv"))
    parser.add_argument("--audit", type=Path, default=Path("data/processed/audit.csv"))
    parser.add_argument("--checkpoints-dir", type=Path, default=Path("outputs/yolov8n_cls/checkpoints"))
    parser.add_argument("--cache-root", type=Path, default=Path("outputs/yolov8n_cls/preprocessed"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/member_4/gradcam"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    print(f"Running shortcut learning analysis for seed {args.seed}...")
    try:
        metrics_df, control_df = analyze_shortcuts(
            predictions_path=args.predictions,
            audit_path=args.audit,
            checkpoints_dir=args.checkpoints_dir,
            cache_root=args.cache_root,
            output_dir=args.output_dir,
            seed=args.seed,
            device=args.device,
        )
        print(f"Shortcut analysis complete: {len(metrics_df)} images evaluated.")
    except FileNotFoundError as err:
        print(f"[NOTE] Shortcut analysis halted: {err}")


if __name__ == "__main__":
    main()
