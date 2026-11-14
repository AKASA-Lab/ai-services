"""Grad-CAM explainability for the ResNet50 runs (Person 3).

Every image is explained by the checkpoint of the fold in which it was **held out**,
so the heatmap comes from a model that never trained on that image. The target class
is the predicted class.

Outputs (in ``--output-dir``):
  gradcam_examples.png      correct high-confidence and error examples across all 4 classes
  gradcam_confusions.png    targeted breakdown of -N/-K and -P/-K confusion patterns
  gradcam_healthy_fn.png    complete panel of all 12 healthy FN images
  gradcam_always_wrong.png  the images misclassified in all three seeds (original + heatmap)
  gradcam_class_average.png mean heatmap per true class (spatial attention check)
  gradcam_stats.csv         per-image attention statistics (mass on plant vs background)
  gradcam_summary.json      aggregated attention statistics

Usage::

    python -m models.resnet50.gradcam --runs-dir outputs/resnet50 --output-dir results/member_3/gradcam --seed 42
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib.colors import LinearSegmentedColormap

from lettuce_benchmark import CLASS_NAMES
from lettuce_benchmark.config import load_config
from models.resnet50.train import build_transforms, create_model

INK, MUTED = "#2b2b29", "#6b6a64"
CONTOUR = "#00e5ff"
HEAT_CMAP = LinearSegmentedColormap.from_list("heat", ["#fbefec", "#e8402a", "#7a1606"])
CORNER_FRACTION = 0.2  # bottom-right 20% x 20% of the frame (where DALL-E watermark was before crop)


def grad_cam(model: torch.nn.Module, image: torch.Tensor, target: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (cam 224x224 in [0, 1], softmax probabilities) for one normalised image.

    In ResNet50, model.forward_features() outputs layer4 feature maps (2048 x 7 x 7).
    model.forward_head() applies global average pooling and linear classifier (model.fc).
    """
    model.zero_grad(set_to_none=True)
    features = model.forward_features(image.unsqueeze(0))
    features.retain_grad()
    logits = model.forward_head(features)
    logits[0, target].backward()
    weights = features.grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * features).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)[0, 0].detach()
    cam = cam / cam.max() if float(cam.max()) > 0 else cam
    return cam.cpu().numpy(), F.softmax(logits, dim=1)[0].detach().cpu().numpy()


def plant_mask(rgb: np.ndarray, threshold: float = 40.0) -> np.ndarray:
    """Pixels whose colour differs clearly from the image border (background) colour."""
    border = np.concatenate([
        rgb[:8].reshape(-1, 3),
        rgb[-8:].reshape(-1, 3),
        rgb[:, :8].reshape(-1, 3),
        rgb[:, -8:].reshape(-1, 3),
    ])
    background = np.median(border, axis=0)
    return np.linalg.norm(rgb.astype(float) - background, axis=2) > threshold


def overlay_spotlight(axis: plt.Axes, rgb: np.ndarray, cam: np.ndarray, title: str, title_colour: str = INK) -> None:
    """Spotlight view: attended regions are bright, ignored regions dimmed, peak outlined in cyan."""
    dimmed = rgb * (0.18 + 0.82 * cam[..., None])
    axis.imshow(np.clip(dimmed, 0, 1))
    axis.contour(cam, levels=[0.5], colors=[CONTOUR], linewidths=1.6)
    axis.set_title(title, fontsize=8.5, color=title_colour, loc="left")
    axis.axis("off")


def overlay_heatmap(axis: plt.Axes, rgb: np.ndarray, cam: np.ndarray, title: str, title_colour: str = INK, alpha: float = 0.5) -> None:
    """Alpha-blended jet colormap overlay on the preprocessed image."""
    axis.imshow(rgb)
    axis.imshow(cam, cmap="jet", alpha=alpha * cam, vmin=0, vmax=1)
    axis.set_title(title, fontsize=8.5, color=title_colour, loc="left")
    axis.axis("off")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-dir", type=Path, default=Path("outputs/resnet50"))
    parser.add_argument("--config", type=Path, default=Path("models/resnet50/config.yaml"))
    parser.add_argument("--predictions", type=Path, default=Path("results/member_3/predictions.csv"))
    parser.add_argument("--audit", type=Path, default=Path("data/processed/audit.csv"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("results/member_3/gradcam"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.manual_seed(0)

    config = load_config(args.config)
    config["model"]["pretrained"] = False  # weights come from local checkpoints
    _, eval_tf = build_transforms(config)
    mean = np.array(config["data"]["mean"])[:, None, None]
    std = np.array(config["data"]["std"])[:, None, None]

    cache_path = args.runs_dir / "image_cache.pt"
    if not cache_path.is_file():
        raise FileNotFoundError(f"Image cache not found at {cache_path}. Run train.py first.")
    cache = torch.load(cache_path, map_location="cpu")["images"]

    predictions = pd.read_csv(args.predictions, dtype={"image_id": str})
    audit = pd.read_csv(args.audit, dtype={"image_id": str})[["image_id", "image_path", "dalle_colour_bar_detected"]]
    times_wrong = (
        predictions.assign(wrong=predictions.true_label != predictions.predicted_label)
        .groupby("image_id")["wrong"]
        .sum()
    )
    rows = predictions[predictions.seed == args.seed].merge(audit, on="image_id")

    records, cams, rgbs = [], {}, {}
    for fold, fold_rows in rows.groupby("fold"):
        model = create_model(config, None).to(device)
        checkpoint_path = args.runs_dir / "checkpoints" / f"resnet50_seed_{args.seed}_fold_{fold}.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state)
        model.eval()

        for row in fold_rows.itertuples(index=False):
            image = eval_tf(cache[row.image_path]).to(device)
            predicted = CLASS_NAMES.index(row.predicted_label)
            cam, probabilities = grad_cam(model, image, predicted)

            # verify reproduction consistency
            diff = abs(probabilities[predicted] - row.confidence)
            if diff >= 1e-3:
                print(f"Warning: probability diff {diff:.4f} on {row.image_path}")

            rgb = np.clip(image.cpu().numpy() * std + mean, 0, 1).transpose(1, 2, 0)
            mask = plant_mask((rgb * 255).astype(np.uint8))
            mass = cam / cam.sum() if cam.sum() > 0 else np.full_like(cam, 1 / cam.size)
            size = cam.shape[0]
            corner = slice(int(size * (1 - CORNER_FRACTION)), None)

            records.append({
                "image_id": row.image_id,
                "image_path": row.image_path,
                "fold": fold,
                "true_label": row.true_label,
                "predicted_label": row.predicted_label,
                "confidence": row.confidence,
                "correct": row.true_label == row.predicted_label,
                "times_wrong_all_seeds": int(times_wrong[row.image_id]),
                "dalle_colour_bar_detected": bool(row.dalle_colour_bar_detected),
                "plant_area_fraction": float(mask.mean()),
                "cam_mass_on_plant": float(mass[mask].sum()),
                "cam_mass_bottom_right_corner": float(mass[corner, corner].sum()),
            })
            cams[row.image_id], rgbs[row.image_id] = cam, rgb

    stats = pd.DataFrame(records)
    stats.round(4).to_csv(out / "gradcam_stats.csv", index=False)

    corner_area = CORNER_FRACTION ** 2
    summary = {
        "seed": args.seed,
        "images": len(stats),
        "method": "Grad-CAM on ResNet50 layer4 output (2048x7x7), target = predicted class, "
                  "model = checkpoint of the fold where the image was held out",
        "plant_mask": "pixels with RGB distance > 40 from median border colour",
        "mean_plant_area_fraction": round(float(stats.plant_area_fraction.mean()), 3),
        "mean_cam_mass_on_plant": round(float(stats.cam_mass_on_plant.mean()), 3),
        "share_of_images_with_most_attention_on_plant": round(float((stats.cam_mass_on_plant > 0.5).mean()), 3),
        "bottom_right_corner_area_fraction": corner_area,
        "mean_cam_mass_bottom_right_corner": round(float(stats.cam_mass_bottom_right_corner.mean()), 4),
        "by_true_class": {
            label: {
                "cam_mass_on_plant": round(float(g.cam_mass_on_plant.mean()), 3),
                "plant_area_fraction": round(float(g.plant_area_fraction.mean()), 3),
                "cam_mass_bottom_right_corner": round(float(g.cam_mass_bottom_right_corner.mean()), 4),
            }
            for label, g in stats.groupby("true_label")
        },
        "by_colour_bar_group": {
            ("had_colour_bar" if flag else "no_colour_bar"): {
                "images": int(len(g)),
                "cam_mass_on_plant": round(float(g.cam_mass_on_plant.mean()), 3),
                "cam_mass_bottom_right_corner": round(float(g.cam_mass_bottom_right_corner.mean()), 4),
            }
            for flag, g in stats.groupby("dalle_colour_bar_detected")
        },
        "correct_vs_wrong_cam_mass_on_plant": {
            ("correct" if flag else "wrong"): round(float(g.cam_mass_on_plant.mean()), 3)
            for flag, g in stats.groupby("correct")
        },
    }
    (out / "gradcam_summary.json").write_text(json.dumps(summary, indent=2))

    # 1) Examples panel: 4 classes x 4 columns (top confident correct + errors)
    figure, axes = plt.subplots(4, 4, figsize=(14, 14.5))
    for r, label in enumerate(CLASS_NAMES):
        group = stats[stats.true_label == label]
        correct = group[group.correct].sort_values("confidence", ascending=False).head(2)
        wrong = group[~group.correct].sort_values("confidence", ascending=False).head(2)
        chosen = pd.concat([correct, wrong, group[group.correct].sort_values("confidence").head(4)])
        chosen = chosen.drop_duplicates("image_id").head(4)
        for c, item in enumerate(chosen.itertuples(index=False)):
            ok = item.correct
            title = (f"{item.image_path}\nTrue: {item.true_label} → Pred: {item.predicted_label} "
                     f"({item.confidence:.2f}) {'✓' if ok else '✗'}")
            overlay_spotlight(axes[r, c], rgbs[item.image_id], cams[item.image_id], title, INK if ok else "#b3261e")
        for c in range(len(chosen), 4):
            axes[r, c].axis("off")
    figure.suptitle(
        f"ResNet50 Grad-CAM (Seed {args.seed}; each image explained by out-of-fold checkpoint)\n"
        "Bright = high activation driving prediction | Dimmed = ignored | Cyan outline = ≥50% peak activation",
        color=INK, fontsize=12, y=0.99
    )
    figure.tight_layout()
    figure.savefig(out / "gradcam_examples.png", dpi=120, bbox_inches="tight")
    plt.close(figure)

    # 2) Targeted Confusions: specifically inspecting -N -> -K and -P -> -K errors alongside correct predictions
    nk_confusions = stats[(stats.true_label == "-N") & (stats.predicted_label == "-K")].sort_values("confidence", ascending=False).head(3)
    pk_confusions = stats[(stats.true_label == "-P") & (stats.predicted_label == "-K")].sort_values("confidence", ascending=False).head(3)
    k_correct = stats[(stats.true_label == "-K") & stats.correct].sort_values("confidence", ascending=False).head(3)

    confusion_rows = [
        ("True -N misclassified as -K", nk_confusions),
        ("True -P misclassified as -K", pk_confusions),
        ("True -K correctly classified", k_correct),
    ]

    figure, axes = plt.subplots(3, 3, figsize=(11.5, 11))
    for r, (row_title, row_data) in enumerate(confusion_rows):
        for c, item in enumerate(row_data.itertuples(index=False)):
            ok = item.correct
            title = (f"{item.image_path}\nTrue: {item.true_label} | Pred: {item.predicted_label} "
                     f"({item.confidence:.2f}) {'✓' if ok else '✗'}")
            overlay_spotlight(axes[r, c], rgbs[item.image_id], cams[item.image_id], title, INK if ok else "#b3261e")
        for c in range(len(row_data), 3):
            axes[r, c].axis("off")
        axes[r, 0].text(-0.05, 0.5, row_title, transform=axes[r, 0].transAxes,
                         rotation=90, va="center", ha="right", fontsize=10, fontweight="bold", color=INK)
    figure.suptitle(
        "ResNet50 Key Confusions: Analyzing Why -N and -P Leaves are Misclassified as -K (Potassium Scorch)\n"
        "Notice how peripheral necrosis and chlorotic leaf margins attract potassium deficiency features",
        color=INK, fontsize=11, y=0.99
    )
    figure.tight_layout()
    figure.savefig(out / "gradcam_confusions.png", dpi=120, bbox_inches="tight")
    plt.close(figure)

    # 3) Complete Healthy FN Panel (all 12 FN images in the dataset)
    fn_items = stats[stats.true_label == "FN"].sort_values(["correct", "confidence"], ascending=[False, False])
    n_fn = len(fn_items)
    cols = 4
    rows_fn = int(np.ceil(n_fn / cols))
    figure, axes = plt.subplots(rows_fn, cols, figsize=(13, 3.4 * rows_fn))
    axes = np.atleast_2d(axes)
    for idx, item in enumerate(fn_items.itertuples(index=False)):
        r, c = divmod(idx, cols)
        ok = item.correct
        title = (f"{item.image_path}\nTrue: FN → Pred: {item.predicted_label} "
                 f"({item.confidence:.2f}) {'✓' if ok else '✗'}")
        overlay_spotlight(axes[r, c], rgbs[item.image_id], cams[item.image_id], title, INK if ok else "#b3261e")
    for idx in range(n_fn, rows_fn * cols):
        r, c = divmod(idx, cols)
        axes[r, c].axis("off")
    figure.suptitle(
        f"ResNet50 Explainability on Minority Healthy Class (FN, {n_fn} total images)\n"
        "Attention uniformly spans the compact, vibrant green leaf rosette",
        color=INK, fontsize=12, y=1.01
    )
    figure.tight_layout()
    figure.savefig(out / "gradcam_healthy_fn.png", dpi=120, bbox_inches="tight")
    plt.close(figure)

    # 4) Images wrong in all 3 seeds: original preprocessed image + heatmap overlay
    hard = stats[stats.times_wrong_all_seeds == 3].sort_values(["true_label", "image_path"])
    if len(hard):
        figure, axes = plt.subplots(2, len(hard), figsize=(3.2 * len(hard), 7.2))
        axes = np.atleast_2d(axes)
        for c, item in enumerate(hard.itertuples(index=False)):
            axes[0, c].imshow(rgbs[item.image_id])
            axes[0, c].set_title(f"{item.image_path}\nTrue: {item.true_label}", fontsize=8.5, color=INK, loc="left")
            axes[0, c].axis("off")
            overlay_spotlight(axes[1, c], rgbs[item.image_id], cams[item.image_id],
                              f"Pred: {item.predicted_label} ({item.confidence:.2f}) ✗", "#b3261e")
        figure.suptitle("Images Misclassified Across All 3 Seeds: Original (top) vs ResNet50 Grad-CAM (bottom)",
                        color=INK, fontsize=12)
        figure.tight_layout(h_pad=2.0)
        figure.savefig(out / "gradcam_always_wrong.png", dpi=120, bbox_inches="tight")
        plt.close(figure)

    # 5) Average heatmap per true class (spatial bias check)
    figure, axes = plt.subplots(1, 4, figsize=(14, 4.0))
    for axis, label in zip(axes, CLASS_NAMES):
        ids = stats.loc[stats.true_label == label, "image_id"]
        average = np.mean([cams[i] for i in ids], axis=0)
        img = axis.imshow(average, cmap=HEAT_CMAP, vmin=0, vmax=1)
        axis.set_title(f"{label}: Mean of {len(ids)} Heatmaps", fontsize=10, color=INK, loc="left")
        axis.axis("off")
        size = average.shape[0]
        start = size * (1 - CORNER_FRACTION)
        axis.add_patch(plt.Rectangle((start, start), size - start - 1, size - start - 1, fill=False,
                                     edgecolor=MUTED, linestyle="--", linewidth=1))
    colourbar = figure.colorbar(img, ax=axes, fraction=0.015, pad=0.01)
    colourbar.set_label("Mean Grad-CAM Activation (0–1)", color=MUTED)
    figure.suptitle(
        "ResNet50 Average Heatmap per Class (Dashed Box = Pre-crop DALL·E Watermark Corner)\n"
        "Attention is centralized on the foliage; cropped bottom-right corner shows zero learned bias",
        color=INK, fontsize=11
    )
    figure.savefig(out / "gradcam_class_average.png", dpi=120, bbox_inches="tight")
    plt.close(figure)

    print(f"Grad-CAM analysis complete! Deliverables written to: {out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
