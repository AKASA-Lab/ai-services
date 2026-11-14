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


def resolve_image_path(
    image_id: str,
    audit_or_metadata_df: pd.DataFrame | None = None,
    metadata_path: Path | str = Path("data/processed/metadata.csv"),
) -> str:
    """Resolve the real relative image path (e.g. 'FN/fn1.png') for an image_id.

    In this benchmark, image_id is a 16-character hex hash (e.g. 'a914de3754e50ed7'),
    while physical files in data/raw/ and the preprocessed cache use the original
    relative path (e.g. 'FN/fn1.png'). This helper looks up the true path from audit
    or metadata tables.
    """
    if audit_or_metadata_df is not None and "image_id" in audit_or_metadata_df.columns and "image_path" in audit_or_metadata_df.columns:
        match = audit_or_metadata_df.loc[audit_or_metadata_df["image_id"] == str(image_id), "image_path"]
        if not match.empty and pd.notna(match.iloc[0]):
            return str(match.iloc[0])

    meta_p = Path(metadata_path)
    if meta_p.is_file():
        meta_df = pd.read_csv(meta_p, dtype={"image_id": str})
        match = meta_df.loc[meta_df["image_id"] == str(image_id), "image_path"]
        if not match.empty and pd.notna(match.iloc[0]):
            return str(match.iloc[0])

    raise KeyError(
        f"Cannot resolve image_path for image_id '{image_id}'. "
        f"image_id is a hex hash, not a relative path; ensure metadata.csv or audit.csv is provided."
    )



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
            rel_path = resolve_image_path(str(row.image_id), audit_or_metadata_df=audit)
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


def analyze_within_1024_group(
    predictions_path: Path | str = Path("results/member_4/predictions.csv"),
    audit_path: Path | str = Path("data/processed/audit.csv"),
    output_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Analyze model performance restricted to the 1024x1024 image group.

    Computes confusion matrix, per-class recall, and the rate at which
    non-FN images from the 1024x1024 group are predicted as FN.
    This analysis requires only predictions.csv and audit.csv (no checkpoints).

    Args:
        predictions_path: Path to held-out predictions CSV.
        audit_path: Path to audit CSV with width/height columns.
        output_dir: Optional directory to save results CSV.

    Returns:
        dict with keys: 'confusion_matrix', 'per_class_recall',
        'non_fn_predicted_fn_count', 'non_fn_predicted_fn_rate',
        'non_fn_predicted_fn_details', 'total_1024_predictions',
        'results_df'.
    """
    predictions_path = Path(predictions_path)
    audit_path = Path(audit_path)

    preds = pd.read_csv(predictions_path, dtype={"image_id": str})
    audit = pd.read_csv(audit_path, dtype={"image_id": str})

    # Merge width/height from audit
    audit_cols = [c for c in ["image_id", "width", "height"] if c in audit.columns]
    merged = preds.merge(audit[audit_cols], on="image_id", how="left")

    # Filter to 1024x1024 images only
    mask_1024 = (merged["width"] == 1024) & (merged["height"] == 1024)
    df_1024 = merged[mask_1024].copy()

    if df_1024.empty:
        print("[WARNING] No 1024x1024 images found in the dataset.")
        return {"confusion_matrix": None, "per_class_recall": {}, "total_1024_predictions": 0}

    # Confusion matrix (rows = true, cols = predicted)
    labels = sorted(df_1024["true_label"].unique(), key=lambda x: CLASS_NAMES.index(x) if x in CLASS_NAMES else 99)
    cm = pd.crosstab(
        df_1024["true_label"],
        df_1024["predicted_label"],
        rownames=["True"],
        colnames=["Pred"],
    )
    # Reindex to canonical order for both axes
    canonical_in_data = [c for c in CLASS_NAMES if c in set(df_1024["true_label"]) | set(df_1024["predicted_label"])]
    cm = cm.reindex(index=canonical_in_data, columns=canonical_in_data, fill_value=0)

    # Per-class recall
    recall = {}
    for label in canonical_in_data:
        true_count = int((df_1024["true_label"] == label).sum())
        correct = int(((df_1024["true_label"] == label) & (df_1024["predicted_label"] == label)).sum())
        recall[label] = {"correct": correct, "total": true_count, "recall": correct / max(true_count, 1)}

    # Non-FN images predicted as FN
    non_fn_1024 = df_1024[df_1024["true_label"] != "FN"]
    predicted_fn = non_fn_1024[non_fn_1024["predicted_label"] == "FN"]
    non_fn_fn_rate = len(predicted_fn) / max(len(non_fn_1024), 1)

    # Details of non-FN predicted as FN
    fn_fp_details = predicted_fn[["image_id", "seed", "fold", "true_label", "predicted_label", "confidence"]].copy()

    results = {
        "confusion_matrix": cm,
        "per_class_recall": recall,
        "non_fn_predicted_fn_count": len(predicted_fn),
        "non_fn_predicted_fn_total": len(non_fn_1024),
        "non_fn_predicted_fn_rate": non_fn_fn_rate,
        "non_fn_predicted_fn_details": fn_fp_details,
        "total_1024_predictions": len(df_1024),
        "results_df": df_1024,
    }

    # Print results
    print("\n" + "=" * 70)
    print("WITHIN-1024x1024 GROUP ANALYSIS")
    print("=" * 70)
    print(f"\nTotal 1024x1024 predictions (across all seeds): {len(df_1024)}")
    print(f"\nConfusion Matrix (rows=True, cols=Pred):")
    print(cm.to_string())
    print(f"\nPer-class recall within 1024x1024:")
    for label, stats in recall.items():
        print(f"  {label:4s}: {stats['recall']:.4f} ({stats['correct']}/{stats['total']})")
    print(f"\nNon-FN predicted as FN: {len(predicted_fn)} / {len(non_fn_1024)} ({non_fn_fn_rate:.4%})")
    if not fn_fp_details.empty:
        print(f"  Details:")
        for _, row in fn_fp_details.iterrows():
            print(f"    {row['image_id']} (true: {row['true_label']}, seed: {row['seed']}, fold: {row['fold']}, conf: {row['confidence']:.4f})")
    print("=" * 70 + "\n")

    # Save if output_dir provided
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        cm.to_csv(out / "within_1024_confusion_matrix.csv")
        pd.DataFrame([
            {"class": k, **v} for k, v in recall.items()
        ]).to_csv(out / "within_1024_recall.csv", index=False)
        fn_fp_details.to_csv(out / "within_1024_fn_false_positives.csv", index=False)

    return results


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

    # Within-1024 analysis runs unconditionally (no checkpoints needed)
    print("Running within-1024 group analysis...")
    analyze_within_1024_group(
        predictions_path=args.predictions,
        audit_path=args.audit,
        output_dir=args.output_dir,
    )

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
