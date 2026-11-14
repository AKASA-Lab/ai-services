"""Turn ResNet50 run outputs into the results/member_3 deliverables.

Metrics are recomputed from the saved prediction CSV with the shared evaluator;
nothing is copied from training logs.

    python -m models.resnet50.summarize --runs-dir outputs/resnet50 --output-dir results/member_3
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

from lettuce_benchmark import CLASS_NAMES
from lettuce_benchmark.config import load_config
from lettuce_benchmark.evaluation import calculate_metrics, evaluate_predictions, save_evaluation
from lettuce_benchmark.reproducibility import software_versions

SEED_STYLE = {42: ("#2a78d6", "-"), 123: ("#eb6834", "--"), 2026: ("#1baf7a", ":")}
INK, MUTED, GRID = "#2b2b29", "#6b6a64", "#e4e3de"


def _style(axis: plt.Axes, title: str, ylabel: str) -> None:
    axis.set_title(title, loc="left", fontsize=11, color=INK)
    axis.set_xlabel("Epoch", color=MUTED)
    axis.set_ylabel(ylabel, color=MUTED)
    axis.grid(True, color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(GRID)
    axis.tick_params(colors=MUTED)


def training_curves(runs_dir: Path, destination: Path, frozen_epochs: int) -> None:
    panels = [("train_loss", "Training loss", "Weighted CE"), ("val_loss", "Validation loss", "Weighted CE"),
              ("val_macro_f1", "Validation macro F1", "Macro F1")]
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for history_path in sorted(runs_dir.glob("runs/seed_*/fold_*/history.csv")):
        seed = int(history_path.parent.parent.name.split("_")[1])
        history = pd.read_csv(history_path)
        colour, linestyle = SEED_STYLE[seed]
        for axis, (column, _, _) in zip(axes, panels):
            axis.plot(history["epoch"], history[column], color=colour, linestyle=linestyle, linewidth=1.6, alpha=0.85)
    for axis, (_, title, ylabel) in zip(axes, panels):
        _style(axis, title, ylabel)
        axis.axvspan(0.5, frozen_epochs + 0.5, color="#f2f1ed", zorder=0)
    axes[2].set_ylim(0, 1.02)
    axes[0].text(1, axes[0].get_ylim()[1], "backbone frozen", color=MUTED, fontsize=8, va="top")
    handles = [plt.Line2D([], [], color=c, linestyle=s, linewidth=2, label=f"seed {seed} (4 folds)")
               for seed, (c, s) in SEED_STYLE.items()]
    figure.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    figure.suptitle("ResNet50: one line per run (3 seeds × 4 folds)", y=1.08, color=INK, fontsize=12)
    figure.tight_layout()
    figure.savefig(destination, dpi=150, bbox_inches="tight")
    plt.close(figure)


def combined_confusion_matrix(predictions: pd.DataFrame, destination: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    for axis, (seed, frame) in zip(axes, predictions.groupby("seed")):
        matrix = np.array(calculate_metrics(frame["true_label"].tolist(), frame["predicted_label"].tolist())["confusion_matrix"])
        row_share = matrix / matrix.sum(axis=1, keepdims=True)
        axis.imshow(row_share, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(CLASS_NAMES)):
            for j in range(len(CLASS_NAMES)):
                axis.text(j, i, f"{matrix[i, j]}", ha="center", va="center", fontsize=11,
                          color="white" if row_share[i, j] > 0.6 else INK)
        axis.set_xticks(range(4), CLASS_NAMES)
        axis.set_yticks(range(4), CLASS_NAMES)
        axis.set_xlabel("Predicted", color=MUTED)
        axis.set_ylabel("True", color=MUTED)
        axis.set_title(f"Seed {seed}: out-of-fold, {len(frame)} images", loc="left", fontsize=11, color=INK)
        axis.tick_params(colors=MUTED, length=0)
        for spine in axis.spines.values():
            spine.set_visible(False)
    figure.suptitle("ResNet50 confusion matrices (counts; shading = share of the true class)", color=INK, fontsize=12)
    figure.tight_layout()
    figure.savefig(destination, dpi=150, bbox_inches="tight")
    plt.close(figure)


def write_error_analysis(predictions: pd.DataFrame, results: dict, output_path: Path) -> None:
    """Generate a markdown error analysis from the prediction results."""
    summary = results["summary"]
    lines = [
        "# ResNet50 — Error Analysis\n",
        "## Overall Performance\n",
        f"- **Macro F1:** {summary['macro_f1']['mean']:.3f} ± {summary['macro_f1']['std']:.3f}",
        f"- **Accuracy:** {summary['accuracy']['mean']:.3f} ± {summary['accuracy']['std']:.3f}",
        f"- **Balanced Accuracy:** {summary['balanced_accuracy']['mean']:.3f} ± {summary['balanced_accuracy']['std']:.3f}\n",
        "## Per-Class Recall\n",
        "| Class | Recall (mean ± std) |",
        "|---|---|",
    ]
    for label in CLASS_NAMES:
        recall = summary["per_class"][label]["recall"]
        lines.append(f"| `{label}` | {recall['mean']:.3f} ± {recall['std']:.3f} |")

    # Identify consistently misclassified images
    predictions = predictions.copy()
    predictions["correct"] = predictions["true_label"] == predictions["predicted_label"]
    per_image = predictions.groupby(["image_id", "true_label"]).agg(
        times_tested=("correct", "count"),
        times_wrong=("correct", lambda s: int((~s).sum())),
    ).reset_index()
    always_wrong = per_image[per_image["times_wrong"] == per_image["times_tested"]]
    sometimes_wrong = per_image[(per_image["times_wrong"] > 0) & (per_image["times_wrong"] < per_image["times_tested"])]

    lines.extend([
        "\n## Misclassification Patterns\n",
        f"- **Always misclassified** (wrong in all seeds): {len(always_wrong)} images",
        f"- **Sometimes misclassified** (wrong in some seeds): {len(sometimes_wrong)} images",
        f"- **Always correct**: {len(per_image) - len(always_wrong) - len(sometimes_wrong)} images\n",
    ])

    # Confusion patterns: which classes are confused with each other
    wrong = predictions[~predictions["correct"]]
    if len(wrong) > 0:
        lines.append("## Most Common Confusions\n")
        lines.append("| True | Predicted As | Count |")
        lines.append("|---|---|---:|")
        confusion_pairs = wrong.groupby(["true_label", "predicted_label"]).size().sort_values(ascending=False)
        for (true, pred), count in confusion_pairs.head(10).items():
            lines.append(f"| `{true}` | `{pred}` | {count} |")

    lines.extend([
        "\n## Limitations\n",
        "- The `FN` (healthy) class has only ~12 images — per-class metrics for this class are inherently noisy.",
        "- Image-level splitting does not guarantee plant-independent generalisation.",
        "- Results are from the lettuce benchmark dataset and do not transfer to fenugreek.",
    ])

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=Path("outputs/resnet50"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/member_3"))
    parser.add_argument("--config", type=Path, default=Path("models/resnet50/config.yaml"))
    parser.add_argument("--audit", type=Path, default=Path("data/processed/audit.csv"))
    parser.add_argument("--metadata", type=Path, default=Path("data/processed/metadata.csv"))
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_csv(args.runs_dir / "predictions.csv", dtype={"image_id": str})
    runs = pd.read_csv(args.runs_dir / "runs.csv")
    results = evaluate_predictions(predictions)
    save_evaluation(predictions, results, out)
    predictions.to_csv(out / "predictions.csv", index=False)
    runs.to_csv(out / "runs.csv", index=False)

    config = load_config(args.config)
    first_run = json.loads(next(args.runs_dir.glob("runs/seed_*/fold_*/run.json")).read_text())
    with (out / "config.json").open("w", encoding="utf-8") as stream:
        json.dump({"model": "ResNet50 (timm resnet50.a1_in1k, ImageNet-1k)", "config": config,
                   "hardware": first_run["hardware"], "software_versions": software_versions(),
                   "seeds": sorted(runs["seed"].unique().tolist()), "folds": [1, 2, 3, 4]}, stream, indent=2)

    training_curves(args.runs_dir, out / "training_curves.png", config["training"]["frozen_backbone_epochs"])
    combined_confusion_matrix(predictions, out / "confusion_matrix.png")

    efficiency = {
        "device": first_run["hardware"]["device"],
        "torch_threads": int(runs["torch_threads"].iloc[0]),
        "parameter_count": int(runs["parameter_count"].iloc[0]),
        "checkpoint_megabytes": round(float(runs["checkpoint_bytes"].iloc[0]) / 1e6, 2),
        "latency_ms_batch1_mean": round(float(runs["latency_ms_batch1"].mean()), 2),
        "latency_ms_batch1_std": round(float(runs["latency_ms_batch1"].std(ddof=1)), 2),
        "training_seconds_per_run_mean": round(float(runs["training_seconds"].mean()), 1),
        "training_seconds_total": round(float(runs["training_seconds"].sum()), 1),
        "peak_gpu_memory": "not applicable (CPU run)" if first_run["hardware"]["device"] == "cpu" else "see run logs",
        "timing_method": "time.perf_counter, batch size 1, 224x224, 10 warm-up + 50 timed forward passes, inference_mode",
    }
    (out / "efficiency.json").write_text(json.dumps(efficiency, indent=2))

    # Error analysis
    write_error_analysis(predictions, results, out / "error_analysis.md")

    # Misclassified images CSV (with colour bar subgroup if audit available)
    if args.audit.is_file():
        audit = pd.read_csv(args.audit, dtype={"image_id": str})[["image_id", "image_path", "dalle_colour_bar_detected"]]
        joined = predictions.merge(audit, on="image_id", how="left")
        joined["correct"] = joined["true_label"] == joined["predicted_label"]
        per_image = joined.groupby(["image_id", "image_path", "true_label", "dalle_colour_bar_detected"]).agg(
            times_wrong=("correct", lambda s: int((~s).sum())),
            predicted_labels=("predicted_label", lambda s: ";".join(s)),
            mean_confidence=("confidence", "mean"),
        ).reset_index().sort_values(["times_wrong", "true_label"], ascending=[False, True])
        per_image[per_image["times_wrong"] > 0].to_csv(out / "misclassified_images.csv", index=False)
        subgroup = joined.groupby(["seed", "dalle_colour_bar_detected"])["correct"].agg(["mean", "count"]).reset_index()
        subgroup.to_csv(out / "accuracy_by_colour_bar.csv", index=False)

    summary = results["summary"]
    print("macro F1 %.3f ± %.3f | accuracy %.3f ± %.3f | balanced acc %.3f ± %.3f" % (
        summary["macro_f1"]["mean"], summary["macro_f1"]["std"], summary["accuracy"]["mean"], summary["accuracy"]["std"],
        summary["balanced_accuracy"]["mean"], summary["balanced_accuracy"]["std"]))
    for label in CLASS_NAMES:
        recall = summary["per_class"][label]["recall"]
        print(f"  recall {label}: {recall['mean']:.3f} ± {recall['std']:.3f}")
    print(json.dumps(efficiency, indent=2))


if __name__ == "__main__":
    main()
