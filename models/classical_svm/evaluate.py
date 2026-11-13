"""Evaluation helpers backed by the shared benchmark metrics implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from lettuce_benchmark import CLASS_NAMES, SEEDS
from lettuce_benchmark.evaluation import (
    evaluate_predictions,
    save_evaluation,
)

METRIC_NAMES = (
    "accuracy",
    "balanced_accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_f1",
)


def build_metric_row(
    experiment: str,
    level: str,
    seed: int | str,
    fold: int | None,
    metrics: dict[str, Any],
    confusion_matrix_file: str | None = None,
) -> dict[str, Any]:
    """Flatten shared aggregate and per-class metric output into one CSV row."""
    row: dict[str, Any] = {
        "experiment": experiment,
        "level": level,
        "seed": seed,
        "fold": fold,
        "confusion_matrix_file": confusion_matrix_file,
    }
    row.update({name: metrics[name] for name in METRIC_NAMES if name in metrics})
    for label in CLASS_NAMES:
        class_metrics = metrics.get("per_class", {}).get(label, {})
        for metric in ("precision", "recall", "f1", "support"):
            if metric in class_metrics:
                row[f"{label}_{metric}"] = class_metrics[metric]
    return row


def write_confusion_matrix_csv(
    matrix: list[list[int]],
    output_path: str | Path,
) -> None:
    """Write a labeled confusion matrix as a CSV file."""
    pd.DataFrame(matrix, index=CLASS_NAMES, columns=CLASS_NAMES).rename_axis(
        "true_label"
    ).to_csv(output_path)


def summarize_oof_predictions(
    predictions: pd.DataFrame,
    experiment: str,
    confusion_dir: str | Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Create per-seed OOF and across-seed metric rows and confusion matrices."""
    result = evaluate_predictions(predictions)
    destination = Path(confusion_dir)
    rows = []
    for seed in SEEDS:
        metrics = result["per_seed"][str(seed)]
        matrix_path = destination / f"{experiment}_seed_{seed}_oof.csv"
        write_confusion_matrix_csv(metrics["confusion_matrix"], matrix_path)
        rows.append(
            build_metric_row(
                experiment,
                "seed_oof",
                seed,
                None,
                metrics,
                str(Path("confusion_matrices") / matrix_path.name),
            )
        )

    summary = result["summary"]
    for statistic in ("mean", "std"):
        aggregate_metrics: dict[str, Any] = {
            name: summary[name][statistic] for name in METRIC_NAMES
        }
        aggregate_metrics["per_class"] = {
            label: {
                metric: summary["per_class"][label][metric][statistic]
                for metric in ("precision", "recall", "f1")
            }
            for label in CLASS_NAMES
        }
        rows.append(
            build_metric_row(
                experiment,
                f"seed_{statistic}",
                statistic,
                None,
                aggregate_metrics,
            )
        )
    return rows, result


def evaluate_and_save(
    predictions: pd.DataFrame,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Aggregate saved-schema out-of-fold predictions and write metrics."""
    results = evaluate_predictions(predictions)
    save_evaluation(predictions, results, output_dir)
    return results
