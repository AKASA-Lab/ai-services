"""Shared classification metrics and repeated out-of-fold prediction aggregation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from . import CLASS_NAMES, SEEDS

PREDICTION_COLUMNS = [
    "image_id",
    "seed",
    "fold",
    "true_label",
    "predicted_label",
    "confidence",
]


def calculate_metrics(
    true_labels: list[str] | np.ndarray,
    predicted_labels: list[str] | np.ndarray,
) -> dict[str, Any]:
    """Calculate aggregate and per-class metrics in canonical label order."""
    if len(true_labels) == 0:
        raise ValueError("Cannot calculate metrics for empty predictions")
    if len(true_labels) != len(predicted_labels):
        raise ValueError("true_labels and predicted_labels must have equal lengths")
    unknown = (set(true_labels) | set(predicted_labels)) - set(CLASS_NAMES)
    if unknown:
        raise ValueError(f"Unknown prediction labels: {sorted(unknown)}")
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels,
        predicted_labels,
        labels=list(CLASS_NAMES),
        zero_division=0,
    )
    matrix = confusion_matrix(true_labels, predicted_labels, labels=list(CLASS_NAMES))
    return {
        "accuracy": float(accuracy_score(true_labels, predicted_labels)),
        "balanced_accuracy": float(balanced_accuracy_score(true_labels, predicted_labels)),
        "macro_precision": float(precision_recall_fscore_support(
            true_labels, predicted_labels, labels=list(CLASS_NAMES), average="macro", zero_division=0
        )[0]),
        "macro_recall": float(precision_recall_fscore_support(
            true_labels, predicted_labels, labels=list(CLASS_NAMES), average="macro", zero_division=0
        )[1]),
        "macro_f1": float(f1_score(
            true_labels, predicted_labels, labels=list(CLASS_NAMES), average="macro", zero_division=0
        )),
        "weighted_f1": float(f1_score(
            true_labels, predicted_labels, labels=list(CLASS_NAMES), average="weighted", zero_division=0
        )),
        "per_class": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(CLASS_NAMES)
        },
        "confusion_matrix": matrix.astype(int).tolist(),
        "class_names": list(CLASS_NAMES),
    }


def aggregate_oof_predictions(predictions: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """Validate and aggregate four held-out folds into one OOF frame per seed."""
    missing = set(PREDICTION_COLUMNS).difference(predictions.columns)
    if missing:
        raise ValueError(f"Predictions are missing columns: {sorted(missing)}")
    frame = predictions[PREDICTION_COLUMNS].copy()
    if frame[PREDICTION_COLUMNS[:-1]].isna().any().any():
        raise ValueError("Prediction identifiers, folds, and labels must not be null")
    if frame["confidence"].isna().any() or not frame["confidence"].between(0, 1).all():
        raise ValueError("confidence must be non-null and between 0 and 1")
    if frame.duplicated(["image_id", "seed", "fold"]).any():
        raise ValueError("Duplicate image prediction within a seed/fold")
    unknown = (set(frame["true_label"]) | set(frame["predicted_label"])) - set(CLASS_NAMES)
    if unknown:
        raise ValueError(f"Unknown labels in predictions: {sorted(unknown)}")

    aggregated: dict[int, pd.DataFrame] = {}
    reference_ids: set[str] | None = None
    for seed, seed_frame in frame.groupby("seed", sort=True):
        if set(seed_frame["fold"].astype(int)) != {1, 2, 3, 4}:
            raise ValueError(f"Seed {seed} must contain test predictions from folds 1 through 4")
        if seed_frame["image_id"].duplicated().any():
            raise ValueError(f"Each image must have one OOF prediction for seed {seed}")
        current_ids = set(seed_frame["image_id"])
        if reference_ids is None:
            reference_ids = current_ids
        elif current_ids != reference_ids:
            raise ValueError("All seeds must contain the same set of image IDs")
        label_counts = seed_frame.groupby("image_id")["true_label"].nunique()
        if (label_counts != 1).any():
            raise ValueError(f"Inconsistent true labels across folds for seed {seed}")
        aggregated[int(seed)] = seed_frame.sort_values("image_id").reset_index(drop=True)
    return aggregated


def evaluate_predictions(predictions: pd.DataFrame) -> dict[str, Any]:
    """Calculate seed-level OOF metrics and across-seed mean/sample std."""
    seed_frames = aggregate_oof_predictions(predictions)
    if set(seed_frames) != set(SEEDS):
        raise ValueError(f"Evaluation requires predictions for exactly these seeds: {list(SEEDS)}")
    per_seed = {
        seed: calculate_metrics(frame["true_label"].tolist(), frame["predicted_label"].tolist())
        for seed, frame in seed_frames.items()
    }
    metric_names = [
        "accuracy",
        "balanced_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
    ]
    summary = {}
    for metric in metric_names:
        values = [per_seed[seed][metric] for seed in sorted(per_seed)]
        summary[metric] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        }
    summary["per_class"] = {}
    for label in CLASS_NAMES:
        summary["per_class"][label] = {}
        for metric in ("precision", "recall", "f1"):
            values = [per_seed[seed]["per_class"][label][metric] for seed in sorted(per_seed)]
            summary["per_class"][label][metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            }
    return {
        "primary_metric": "macro_f1",
        "per_seed": {str(seed): metrics for seed, metrics in per_seed.items()},
        "summary": summary,
    }


def save_evaluation(
    predictions: pd.DataFrame,
    results: dict[str, Any],
    output_dir: str | Path,
) -> None:
    """Write metrics JSON/CSV and per-seed confusion matrix images."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(results, stream, indent=2)

    metric_names = list(results["summary"].keys())
    rows = []
    for seed, metrics in results["per_seed"].items():
        row = {"seed": int(seed)}
        row.update({name: metrics[name] for name in metric_names if name in metrics})
        for label in CLASS_NAMES:
            for metric in ("precision", "recall", "f1"):
                row[f"{label}_{metric}"] = metrics["per_class"][label][metric]
        rows.append(row)
    for statistic in ("mean", "std"):
        row = {"seed": statistic}
        for metric in metric_names:
            if metric == "per_class":
                continue
            row[metric] = results["summary"][metric][statistic]
        for label in CLASS_NAMES:
            for metric in ("precision", "recall", "f1"):
                row[f"{label}_{metric}"] = results["summary"]["per_class"][label][metric][statistic]
        rows.append(row)
    pd.DataFrame(rows).to_csv(destination / "metrics.csv", index=False)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    seed_frames = aggregate_oof_predictions(predictions)
    for seed, frame in seed_frames.items():
        matrix = calculate_metrics(
            frame["true_label"].tolist(), frame["predicted_label"].tolist()
        )["confusion_matrix"]
        figure, axis = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES,
            ax=axis,
        )
        axis.set(xlabel="Predicted label", ylabel="True label", title=f"OOF confusion matrix — seed {seed}")
        figure.tight_layout()
        figure.savefig(destination / f"confusion_matrix_seed_{seed}.png", dpi=160)
        plt.close(figure)
