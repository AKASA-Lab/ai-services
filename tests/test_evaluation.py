import pandas as pd
import pytest

from lettuce_benchmark.evaluation import (
    aggregate_oof_predictions,
    calculate_metrics,
    evaluate_predictions,
    save_evaluation,
)


def make_predictions(seeds=(42, 123, 2026)) -> pd.DataFrame:
    rows = []
    labels = ("FN", "-N", "-P", "-K")
    for seed in seeds:
        for fold in range(1, 5):
            label = labels[fold - 1]
            rows.append({
                "image_id": f"image_{fold}",
                "seed": seed,
                "fold": fold,
                "true_label": label,
                "predicted_label": label,
                "confidence": 0.9,
            })
    return pd.DataFrame(rows)


def test_metrics_include_per_class_and_confusion_matrix() -> None:
    metrics = calculate_metrics(
        ["FN", "-N", "-P", "-K"],
        ["FN", "-N", "-K", "-K"],
    )
    assert metrics["accuracy"] == 0.75
    assert metrics["macro_f1"] < 1
    assert metrics["per_class"]["-P"]["recall"] == 0
    assert len(metrics["confusion_matrix"]) == 4


def test_oof_aggregation_and_seed_summary() -> None:
    predictions = make_predictions()
    aggregated = aggregate_oof_predictions(predictions)
    assert set(aggregated) == {42, 123, 2026}
    assert all(len(frame) == 4 for frame in aggregated.values())
    result = evaluate_predictions(predictions)
    assert result["primary_metric"] == "macro_f1"
    assert result["summary"]["macro_f1"] == {"mean": 1.0, "std": 0.0}


def test_duplicate_oof_prediction_is_rejected() -> None:
    predictions = make_predictions(seeds=(42,))
    predictions.loc[1, "image_id"] = predictions.loc[0, "image_id"]
    with pytest.raises(ValueError, match="one OOF prediction"):
        aggregate_oof_predictions(predictions)


def test_missing_fold_is_rejected() -> None:
    predictions = make_predictions(seeds=(42,)).iloc[:3]
    with pytest.raises(ValueError, match="folds 1 through 4"):
        aggregate_oof_predictions(predictions)


def test_evaluation_saves_machine_readable_results_and_plots(tmp_path) -> None:
    predictions = make_predictions()
    result = evaluate_predictions(predictions)
    save_evaluation(predictions, result, tmp_path)
    assert (tmp_path / "metrics.json").is_file()
    assert (tmp_path / "metrics.csv").is_file()
    assert (tmp_path / "confusion_matrix_seed_42.png").is_file()
    saved = pd.read_csv(tmp_path / "metrics.csv")
    assert set(saved["seed"].astype(str)) == {"42", "123", "2026", "mean", "std"}
