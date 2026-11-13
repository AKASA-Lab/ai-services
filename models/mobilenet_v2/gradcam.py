"""Grad-CAM for the MobileNetV2 runs (Person 1).

Every image is explained by the checkpoint of the fold in which it was **held out**,
so the heatmap comes from a model that never trained on that image. The target class
is the predicted class.

Outputs (in ``--output-dir``):
  gradcam_examples.png      correct and wrong examples for every class
  gradcam_always_wrong.png  the images misclassified in all three seeds
  gradcam_class_average.png mean heatmap per true class (is attention parked somewhere fixed?)
  gradcam_stats.csv         per-image attention statistics
  gradcam_summary.json      aggregated statistics

    python -m models.mobilenet_v2.gradcam --runs-dir outputs/mobilenet_v2 --seed 42
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
from models.mobilenet_v2.train import build_transforms, create_model

INK, MUTED = "#2b2b29", "#6b6a64"
CONTOUR = "#00e5ff"
HEAT_CMAP = LinearSegmentedColormap.from_list("heat", ["#fbefec", "#e8402a", "#7a1606"])
CORNER_FRACTION = 0.2  # bottom-right 20% x 20% of the frame (4% of the area)


def grad_cam(model: torch.nn.Module, image: torch.Tensor, target: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (cam 224x224 in [0, 1], softmax probabilities) for one normalised image."""
    model.zero_grad(set_to_none=True)
    features = model.forward_features(image.unsqueeze(0))
    features.retain_grad()
    logits = model.forward_head(features)
    logits[0, target].backward()
    weights = features.grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * features).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)[0, 0].detach()
    cam = cam / cam.max() if float(cam.max()) > 0 else cam
    return cam.numpy(), F.softmax(logits, dim=1)[0].detach().numpy()


def plant_mask(rgb: np.ndarray, threshold: float = 40.0) -> np.ndarray:
    """Pixels whose colour differs clearly from the image's own border (background) colour."""
    border = np.concatenate([rgb[:8].reshape(-1, 3), rgb[-8:].reshape(-1, 3),
                             rgb[:, :8].reshape(-1, 3), rgb[:, -8:].reshape(-1, 3)])
    background = np.median(border, axis=0)
    return np.linalg.norm(rgb.astype(float) - background, axis=2) > threshold


def overlay(axis: plt.Axes, rgb: np.ndarray, cam: np.ndarray, title: str, title_colour: str = INK) -> None:
    """Spotlight view: regions the model ignored are dimmed, the strongest region is outlined.

    Leaf colours here span green, yellow, red and purple, so a coloured heat overlay is hard to
    read. Brightness (attended = bright) and a cyan outline at 50% of the peak stay readable.
    """
    axis.imshow(rgb * (0.18 + 0.82 * cam[..., None]))
    axis.contour(cam, levels=[0.5], colors=[CONTOUR], linewidths=1.6)
    axis.set_title(title, fontsize=8.5, color=title_colour, loc="left")
    axis.axis("off")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-dir", type=Path, default=Path("outputs/mobilenet_v2"))
    parser.add_argument("--config", type=Path, default=Path("models/mobilenet_v2/config.yaml"))
    parser.add_argument("--predictions", type=Path, default=Path("results/member_1/predictions.csv"))
    parser.add_argument("--audit", type=Path, default=Path("data/processed/audit.csv"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("results/member_1/gradcam"))
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    config = load_config(args.config)
    config["model"]["pretrained"] = False  # weights come from our checkpoints
    _, eval_tf = build_transforms(config)
    mean = np.array(config["data"]["mean"])[:, None, None]
    std = np.array(config["data"]["std"])[:, None, None]
    cache = torch.load(args.runs_dir / "image_cache.pt")["images"]

    predictions = pd.read_csv(args.predictions, dtype={"image_id": str})
    audit = pd.read_csv(args.audit, dtype={"image_id": str})[["image_id", "image_path", "dalle_colour_bar_detected"]]
    times_wrong = (predictions.assign(wrong=predictions.true_label != predictions.predicted_label)
                   .groupby("image_id")["wrong"].sum())
    rows = predictions[predictions.seed == args.seed].merge(audit, on="image_id")

    records, cams, rgbs = [], {}, {}
    for fold, fold_rows in rows.groupby("fold"):
        model = create_model(config, None)
        state = torch.load(args.runs_dir / "checkpoints" / f"mobilenet_v2_seed_{args.seed}_fold_{fold}.pt")
        model.load_state_dict(state)
        model.eval()
        for row in fold_rows.itertuples(index=False):
            image = eval_tf(cache[row.image_path])
            predicted = CLASS_NAMES.index(row.predicted_label)
            cam, probabilities = grad_cam(model, image, predicted)
            assert abs(probabilities[predicted] - row.confidence) < 1e-3, "checkpoint does not reproduce saved prediction"
            rgb = np.clip(image.numpy() * std + mean, 0, 1).transpose(1, 2, 0)
            mask = plant_mask((rgb * 255).astype(np.uint8))
            mass = cam / cam.sum() if cam.sum() > 0 else np.full_like(cam, 1 / cam.size)
            size = cam.shape[0]
            corner = slice(int(size * (1 - CORNER_FRACTION)), None)
            records.append({
                "image_id": row.image_id, "image_path": row.image_path, "fold": fold,
                "true_label": row.true_label, "predicted_label": row.predicted_label,
                "confidence": row.confidence, "correct": row.true_label == row.predicted_label,
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
        "method": "Grad-CAM on the 1280x7x7 output of conv_head+bn2, target = predicted class, "
                  "model = checkpoint of the fold where the image was held out",
        "plant_mask": "pixels with RGB distance > 40 from the median border colour",
        "mean_plant_area_fraction": round(float(stats.plant_area_fraction.mean()), 3),
        "mean_cam_mass_on_plant": round(float(stats.cam_mass_on_plant.mean()), 3),
        "share_of_images_with_most_attention_on_plant": round(float((stats.cam_mass_on_plant > 0.5).mean()), 3),
        "bottom_right_corner_area_fraction": corner_area,
        "mean_cam_mass_bottom_right_corner": round(float(stats.cam_mass_bottom_right_corner.mean()), 4),
        "by_true_class": {
            label: {"cam_mass_on_plant": round(float(g.cam_mass_on_plant.mean()), 3),
                    "plant_area_fraction": round(float(g.plant_area_fraction.mean()), 3),
                    "cam_mass_bottom_right_corner": round(float(g.cam_mass_bottom_right_corner.mean()), 4)}
            for label, g in stats.groupby("true_label")
        },
        "by_colour_bar_group": {
            ("had_colour_bar" if flag else "no_colour_bar"): {
                "images": int(len(g)),
                "cam_mass_on_plant": round(float(g.cam_mass_on_plant.mean()), 3),
                "cam_mass_bottom_right_corner": round(float(g.cam_mass_bottom_right_corner.mean()), 4)}
            for flag, g in stats.groupby("dalle_colour_bar_detected")
        },
        "correct_vs_wrong_cam_mass_on_plant": {
            ("correct" if flag else "wrong"): round(float(g.cam_mass_on_plant.mean()), 3)
            for flag, g in stats.groupby("correct")
        },
    }
    (out / "gradcam_summary.json").write_text(json.dumps(summary, indent=2))

    # 1) Examples: per class, the two most confident correct predictions and up to two errors.
    figure, axes = plt.subplots(4, 4, figsize=(12, 12.8))
    for r, label in enumerate(CLASS_NAMES):
        group = stats[stats.true_label == label]
        correct = group[group.correct].sort_values("confidence", ascending=False).head(2)
        wrong = group[~group.correct].sort_values("confidence", ascending=False).head(2)
        chosen = pd.concat([correct, wrong, group[group.correct].sort_values("confidence").head(4)])
        chosen = chosen.drop_duplicates("image_id").head(4)
        for c, item in enumerate(chosen.itertuples(index=False)):
            ok = item.correct
            title = (f"{item.image_path}\ntrue {item.true_label} → pred {item.predicted_label} "
                     f"({item.confidence:.2f}) {'✓' if ok else '✗'}")
            overlay(axes[r, c], rgbs[item.image_id], cams[item.image_id], title, INK if ok else "#b3261e")
        for c in range(len(chosen), 4):
            axes[r, c].axis("off")
    figure.suptitle(f"MobileNetV2 Grad-CAM (seed {args.seed}; each image explained by the model that never trained on it)\n"
                    "bright = regions that drove the prediction, dimmed = ignored, cyan line = strongest region (≥ 50% of peak); ✗ = misclassified", color=INK, fontsize=12)
    figure.tight_layout()
    figure.savefig(out / "gradcam_examples.png", dpi=95, bbox_inches="tight")
    plt.close(figure)

    # 2) Images wrong in all three seeds.
    hard = stats[stats.times_wrong_all_seeds == 3].sort_values(["true_label", "image_path"])
    if len(hard):
        figure, axes = plt.subplots(2, len(hard), figsize=(3.1 * len(hard), 7.4))
        axes = np.atleast_2d(axes)
        for c, item in enumerate(hard.itertuples(index=False)):
            axes[0, c].imshow(rgbs[item.image_id])
            axes[0, c].set_title(f"{item.image_path}\ntrue {item.true_label}", fontsize=8.5, color=INK, loc="left")
            axes[0, c].axis("off")
            overlay(axes[1, c], rgbs[item.image_id], cams[item.image_id],
                    f"pred {item.predicted_label} ({item.confidence:.2f})", "#b3261e")
        figure.suptitle("Images misclassified in all 3 seeds: original (top) and Grad-CAM for the wrong class (bottom)",
                        color=INK, fontsize=12)
        figure.tight_layout(h_pad=2.2)
        figure.savefig(out / "gradcam_always_wrong.png", dpi=95, bbox_inches="tight")
        plt.close(figure)

    # 3) Average heatmap per true class: would reveal attention fixed on one spot (e.g. a corner).
    figure, axes = plt.subplots(1, 4, figsize=(14, 3.9))
    for axis, label in zip(axes, CLASS_NAMES):
        ids = stats.loc[stats.true_label == label, "image_id"]
        average = np.mean([cams[i] for i in ids], axis=0)
        image = axis.imshow(average, cmap=HEAT_CMAP, vmin=0, vmax=1)
        axis.set_title(f"{label}: mean of {len(ids)} heatmaps", fontsize=10, color=INK, loc="left")
        axis.axis("off")
        size = average.shape[0]
        start = size * (1 - CORNER_FRACTION)
        axis.add_patch(plt.Rectangle((start, start), size - start - 1, size - start - 1, fill=False,
                                     edgecolor=MUTED, linestyle="--", linewidth=1))
    colourbar = figure.colorbar(image, ax=axes, fraction=0.015, pad=0.01)
    colourbar.set_label("mean Grad-CAM (0–1)", color=MUTED)
    figure.suptitle("Average Grad-CAM per class (dashed box = bottom-right corner, where the DALL·E bar was before cropping)",
                    color=INK, fontsize=11)
    figure.savefig(out / "gradcam_class_average.png", dpi=95, bbox_inches="tight")
    plt.close(figure)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
