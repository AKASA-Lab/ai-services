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
from scipy.stats import fisher_exact
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
    """Analyze model performance restricted to the 1024x1024 image group and compare with non-1024.

    Computes:
    1. Per-class recall and confusion matrices for 1024x1024, non-1024, and overall groups.
    2. False-FN predictions and rates across 1024 vs non-1024 groups.
    3. Within-class comparison for -N: false-FN predictions and rates (predictions & unique images),
       plus Fisher's exact test p-values.

    This analysis requires only predictions.csv and audit.csv (no checkpoints).
    """
    predictions_path = Path(predictions_path)
    audit_path = Path(audit_path)

    preds = pd.read_csv(predictions_path, dtype={"image_id": str})
    audit = pd.read_csv(audit_path, dtype={"image_id": str})

    # Merge width/height from audit
    audit_cols = [c for c in ["image_id", "image_path", "width", "height"] if c in audit.columns]
    merged = preds.merge(audit[audit_cols], on="image_id", how="left")

    merged["is_1024"] = (merged["width"] == 1024) & (merged["height"] == 1024)
    merged["original_size"] = merged["width"].astype(str) + "x" + merged["height"].astype(str)

    df_1024 = merged[merged["is_1024"]].copy()
    df_non1024 = merged[~merged["is_1024"]].copy()

    # Confusion matrix for 1024x1024 (rows = true, cols = predicted)
    canonical_in_data = [c for c in CLASS_NAMES if c in set(df_1024["true_label"]) | set(df_1024["predicted_label"])]
    cm_1024 = pd.crosstab(
        df_1024["true_label"],
        df_1024["predicted_label"],
        rownames=["True"],
        colnames=["Pred"],
    ).reindex(index=canonical_in_data, columns=canonical_in_data, fill_value=0)

    # Per-class recall within 1024x1024
    recall_1024 = {}
    for label in canonical_in_data:
        true_count = int((df_1024["true_label"] == label).sum())
        correct = int(((df_1024["true_label"] == label) & (df_1024["predicted_label"] == label)).sum())
        recall_1024[label] = {"correct": correct, "total": true_count, "recall": correct / max(true_count, 1)}

    # All false FN predictions (True label != FN, Predicted == FN)
    false_fn = merged[(merged["true_label"] != "FN") & (merged["predicted_label"] == "FN")].copy()
    false_fn_out = false_fn[[
        "image_id", "image_path", "true_label", "predicted_label",
        "seed", "fold", "confidence", "width", "height", "original_size", "is_1024"
    ]].rename(columns={"is_1024": "is_1024_group"}).reset_index(drop=True)

    # Within-class comparison for -N only
    df_n = merged[merged["true_label"] == "-N"].copy()
    n_1024_preds = df_n[df_n["is_1024"]]
    tot_n_1024_preds = len(n_1024_preds)
    false_fn_n_1024_preds = int((n_1024_preds["predicted_label"] == "FN").sum())
    rate_n_1024_preds = false_fn_n_1024_preds / max(tot_n_1024_preds, 1)

    n_non1024_preds = df_n[~df_n["is_1024"]]
    tot_n_non1024_preds = len(n_non1024_preds)
    false_fn_n_non1024_preds = int((n_non1024_preds["predicted_label"] == "FN").sum())
    rate_n_non1024_preds = false_fn_n_non1024_preds / max(tot_n_non1024_preds, 1)

    # Fisher exact test (predictions level)
    table_n_preds = [
        [false_fn_n_1024_preds, tot_n_1024_preds - false_fn_n_1024_preds],
        [false_fn_n_non1024_preds, tot_n_non1024_preds - false_fn_n_non1024_preds],
    ]
    odds_ratio_preds, p_val_preds = fisher_exact(table_n_preds)

    # Unique images level for -N
    n_1024_imgs = n_1024_preds.groupby("image_id")["predicted_label"].apply(lambda s: (s == "FN").any())
    tot_n_1024_imgs = len(n_1024_imgs)
    false_fn_n_1024_imgs = int(n_1024_imgs.sum())
    rate_n_1024_imgs = false_fn_n_1024_imgs / max(tot_n_1024_imgs, 1)

    n_non1024_imgs = n_non1024_preds.groupby("image_id")["predicted_label"].apply(lambda s: (s == "FN").any())
    tot_n_non1024_imgs = len(n_non1024_imgs)
    false_fn_n_non1024_imgs = int(n_non1024_imgs.sum())
    rate_n_non1024_imgs = false_fn_n_non1024_imgs / max(tot_n_non1024_imgs, 1)

    # Fisher exact test (unique images level)
    table_n_imgs = [
        [false_fn_n_1024_imgs, tot_n_1024_imgs - false_fn_n_1024_imgs],
        [false_fn_n_non1024_imgs, tot_n_non1024_imgs - false_fn_n_non1024_imgs],
    ]
    odds_ratio_imgs, p_val_imgs = fisher_exact(table_n_imgs)

    # Analysis dataframe across groups
    analysis_records = []
    for grp_name, grp in [("1024x1024", df_1024), ("non_1024", df_non1024), ("overall", merged)]:
        for cls in CLASS_NAMES:
            cls_df = grp[grp["true_label"] == cls]
            uniq_imgs = int(cls_df["image_id"].nunique())
            tot_p = len(cls_df)
            corr_p = int((cls_df["predicted_label"] == cls).sum())
            rec = round(corr_p / max(tot_p, 1), 4) if tot_p > 0 else np.nan
            p_counts = cls_df["predicted_label"].value_counts()
            false_fn_c = int(p_counts.get("FN", 0)) if cls != "FN" else 0
            false_fn_r = round(false_fn_c / max(tot_p, 1), 4) if tot_p > 0 and cls != "FN" else np.nan

            analysis_records.append({
                "size_group": grp_name,
                "true_class": cls,
                "unique_images": uniq_imgs,
                "total_predictions": tot_p,
                "correct_predictions": corr_p,
                "recall": rec,
                "pred_FN": int(p_counts.get("FN", 0)),
                "pred_minus_N": int(p_counts.get("-N", 0)),
                "pred_minus_P": int(p_counts.get("-P", 0)),
                "pred_minus_K": int(p_counts.get("-K", 0)),
                "false_fn_rate_within_class": false_fn_r,
            })
    analysis_df = pd.DataFrame(analysis_records)

    # Print results
    print("\n" + "=" * 70)
    print("WITHIN-1024x1024 GROUP ANALYSIS & SHORTCUT INVESTIGATION")
    print("=" * 70)
    print(f"\nTotal 1024x1024 predictions (across all seeds): {len(df_1024)}")
    print(f"\nConfusion Matrix within 1024x1024 (rows=True, cols=Pred):")
    print(cm_1024.to_string())
    print(f"\nPer-class recall within 1024x1024:")
    for label, stats in recall_1024.items():
        print(f"  {label:4s}: {stats['recall']:.4f} ({stats['correct']}/{stats['total']})")

    print(f"\nTotal False FN Predictions (non-FN predicted as FN): {len(false_fn_out)}")
    print(false_fn_out[["image_id", "image_path", "true_label", "seed", "fold", "confidence", "original_size", "is_1024_group"]].to_string())

    print("\n" + "-" * 70)
    print("WITHIN-CLASS (-N ONLY) FALSE-FN COMPARISON")
    print("-" * 70)
    print(f"Predictions level:")
    print(f"  1024x1024 -N : {false_fn_n_1024_preds:2d} / {tot_n_1024_preds:3d} ({rate_n_1024_preds:.4%})")
    print(f"  Non-1024  -N : {false_fn_n_non1024_preds:2d} / {tot_n_non1024_preds:3d} ({rate_n_non1024_preds:.4%})")
    print(f"  Fisher exact test p-value: {p_val_preds:.6f} (odds ratio: {odds_ratio_preds:.4f})")
    print(f"Unique images level:")
    print(f"  1024x1024 -N : {false_fn_n_1024_imgs:2d} / {tot_n_1024_imgs:3d} unique images ({rate_n_1024_imgs:.4%})")
    print(f"  Non-1024  -N : {false_fn_n_non1024_imgs:2d} / {tot_n_non1024_imgs:3d} unique images ({rate_n_non1024_imgs:.4%})")
    print(f"  Fisher exact test p-value: {p_val_imgs:.6f} (odds ratio: {odds_ratio_imgs:.4f})")
    print("=" * 70 + "\n")

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        analysis_df.to_csv(out / "within_1024_analysis.csv", index=False)
        false_fn_out.to_csv(out / "false_fn_predictions.csv", index=False)

    return {
        "confusion_matrix_1024": cm_1024,
        "recall_1024": recall_1024,
        "false_fn_predictions": false_fn_out,
        "analysis_df": analysis_df,
        "within_n_predictions": {
            "false_fn_1024": false_fn_n_1024_preds,
            "total_1024": tot_n_1024_preds,
            "rate_1024": rate_n_1024_preds,
            "false_fn_non1024": false_fn_n_non1024_preds,
            "total_non1024": tot_n_non1024_preds,
            "rate_non1024": rate_n_non1024_preds,
            "p_value": p_val_preds,
            "odds_ratio": odds_ratio_preds,
        },
        "within_n_unique_images": {
            "false_fn_1024": false_fn_n_1024_imgs,
            "total_1024": tot_n_1024_imgs,
            "rate_1024": rate_n_1024_imgs,
            "false_fn_non1024": false_fn_n_non1024_imgs,
            "total_non1024": tot_n_non1024_imgs,
            "rate_non1024": rate_n_non1024_imgs,
            "p_value": p_val_imgs,
            "odds_ratio": odds_ratio_imgs,
        },
    }


def resampling_control(
    predictions_path: Path | str = Path("results/member_4/predictions.csv"),
    audit_path: Path | str = Path("data/processed/audit.csv"),
    checkpoints_dir: Path | str = Path("outputs/yolov8n_cls/checkpoints"),
    cache_root: Path | str = Path("outputs/yolov8n_cls/preprocessed"),
    output_dir: Path | str = Path("results/member_4/gradcam"),
    seed: int = 42,
    cache_size: int = 256,
    imgsz: int = 224,
    device: str = "cpu",
) -> pd.DataFrame:
    """Resampling control: simulate non-1024 acquisition for 1024x1024 images.

    For each 1024x1024 test image:
      1. Load the 256x256 cached image (which was centre-cropped from 1024x1024).
      2. Downscale to a random size drawn from the non-1024 dimension range
         (width 1015-1024, height 999-1008), then scale back up to cache_size.
         This mimics the resampling pipeline experienced by non-1024 images.
      3. Resize to imgsz (224x224) and re-predict with the held-out fold model.
      4. Compare original vs resampled predictions, focusing on FN recall change.

    Requires checkpoints and cached images. Inference only, no retraining.

    Args:
        predictions_path: Predictions CSV path.
        audit_path: Audit CSV with width/height.
        checkpoints_dir: Directory with fold checkpoints.
        cache_root: Preprocessed image cache directory.
        output_dir: Output directory for results.
        seed: Random seed partition.
        cache_size: Cache resolution (default 256).
        imgsz: Model input size (default 224).
        device: torch device string.

    Returns:
        DataFrame with columns: image_id, true_label, pred_original, pred_resampled,
        conf_original, conf_resampled, simulated_width, simulated_height.
    """
    from ultralytics import YOLO

    predictions_path = Path(predictions_path)
    audit_path = Path(audit_path)
    checkpoints_dir = Path(checkpoints_dir)
    cache_root = Path(cache_root)
    output_dir = Path(output_dir)

    if not checkpoints_dir.is_dir():
        raise FileNotFoundError(
            f"Checkpoints directory not found: {checkpoints_dir}. "
            f"Trained checkpoints must be present to run resampling control."
        )
    if not cache_root.is_dir():
        raise FileNotFoundError(
            f"Preprocessed image cache directory not found: {cache_root}. "
            f"Preprocessed images must be present to run resampling control."
        )

    preds = pd.read_csv(predictions_path, dtype={"image_id": str})
    audit = pd.read_csv(audit_path, dtype={"image_id": str})

    preds_seed = preds[preds["seed"] == int(seed)].copy()
    audit_cols = [c for c in ["image_id", "image_path", "width", "height"] if c in audit.columns]
    merged = preds_seed.merge(audit[audit_cols], on="image_id", how="left")

    # Filter to 1024x1024 only
    mask_1024 = (merged["width"] == 1024) & (merged["height"] == 1024)
    df_1024 = merged[mask_1024].copy()

    if df_1024.empty:
        print("[WARNING] No 1024x1024 images found.")
        return pd.DataFrame()

    # Non-1024 dimension ranges (from dataset audit)
    # Width: 1015-1024, Height: 999-1008
    NON_1024_W_MIN, NON_1024_W_MAX = 1015, 1024
    NON_1024_H_MIN, NON_1024_H_MAX = 999, 1008

    rng = np.random.default_rng(seed)

    # Load fold models
    models_cache: dict[int, Any] = {}
    for fold in range(1, 5):
        ckpt_path = resolve_checkpoint_path(seed, fold, checkpoints_dir)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Missing checkpoint for fold {fold}: {ckpt_path}")
        models_cache[fold] = YOLO(str(ckpt_path))

    records = []
    for row in df_1024.itertuples(index=False):
        rel_path = getattr(row, "image_path", None)
        if not rel_path or pd.isna(rel_path):
            rel_path = resolve_image_path(str(row.image_id), audit_or_metadata_df=audit)
        img_path = cached_path(cache_root, str(rel_path))
        if not img_path.is_file():
            raise FileNotFoundError(f"Cached image missing: {img_path}")

        with Image.open(img_path) as pil_img:
            img_256 = pil_img.convert("RGB")

        # Simulate non-1024 resampling: down to random non-1024 size, then back to cache_size
        sim_w = int(rng.integers(NON_1024_W_MIN, NON_1024_W_MAX + 1))
        sim_h = int(rng.integers(NON_1024_H_MIN, NON_1024_H_MAX + 1))

        # Downscale to simulated acquisition size, then back to cache_size
        resampled = img_256.resize((sim_w, sim_h), Image.Resampling.BICUBIC)
        resampled = resampled.resize((cache_size, cache_size), Image.Resampling.BICUBIC)

        # Resize to model input
        resampled_224 = resampled.resize((imgsz, imgsz), Image.Resampling.BICUBIC)
        rgb_arr = np.array(resampled_224)

        tensor = torch.from_numpy(rgb_arr).permute(2, 0, 1).float() / 255.0

        fold = int(row.fold)
        model = models_cache[fold]

        with torch.no_grad():
            out = model.model(tensor.unsqueeze(0).to(next(model.model.parameters()).device))
            if isinstance(out, (tuple, list)):
                logits = out[1]
            else:
                logits = out
            probs_resampled = logits.softmax(-1).detach().cpu().numpy()
            probs_canonical = to_canonical_order(probs_resampled, model.names)[0]
            pred_resampled = CLASS_NAMES[int(probs_canonical.argmax())]
            conf_resampled = float(probs_canonical.max())

        records.append({
            "image_id": row.image_id,
            "true_label": row.true_label,
            "pred_original": row.predicted_label,
            "pred_resampled": pred_resampled,
            "conf_original": row.confidence,
            "conf_resampled": round(conf_resampled, 4),
            "simulated_width": sim_w,
            "simulated_height": sim_h,
        })

    results_df = pd.DataFrame(records)

    # Print summary
    print("\n" + "=" * 70)
    print("RESAMPLING CONTROL (1024x1024 images simulated as non-1024)")
    print("=" * 70)
    print(f"Images tested: {len(results_df)}")
    flipped = results_df[results_df["pred_original"] != results_df["pred_resampled"]]
    print(f"Predictions changed: {len(flipped)} / {len(results_df)} ({len(flipped)/max(len(results_df),1):.2%})")

    # FN recall: original vs resampled
    fn_rows = results_df[results_df["true_label"] == "FN"]
    fn_recall_orig = (fn_rows["pred_original"] == "FN").mean() if len(fn_rows) > 0 else float("nan")
    fn_recall_resamp = (fn_rows["pred_resampled"] == "FN").mean() if len(fn_rows) > 0 else float("nan")
    print(f"\nFN recall (original):   {fn_recall_orig:.4f}")
    print(f"FN recall (resampled):  {fn_recall_resamp:.4f}")
    print(f"FN recall change:       {fn_recall_resamp - fn_recall_orig:+.4f}")

    if not flipped.empty:
        print(f"\nFlipped predictions:")
        for _, r in flipped.iterrows():
            print(f"  {r['image_id']} true={r['true_label']} "
                  f"orig={r['pred_original']}({r['conf_original']:.3f}) "
                  f"-> resamp={r['pred_resampled']}({r['conf_resampled']:.3f}) "
                  f"sim_size={r['simulated_width']}x{r['simulated_height']}")
    print("=" * 70 + "\n")

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_dir / "resampling_control.csv", index=False)

    return results_df


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

    print(f"Running resampling control for seed {args.seed}...")
    try:
        resampling_control(
            predictions_path=args.predictions,
            audit_path=args.audit,
            checkpoints_dir=args.checkpoints_dir,
            cache_root=args.cache_root,
            output_dir=args.output_dir,
            seed=args.seed,
            device=args.device,
        )
    except FileNotFoundError as err:
        print(f"[NOTE] Resampling control halted: {err}")


if __name__ == "__main__":
    main()
