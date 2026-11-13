"""Validation-selected classical ML experiments using the shared split manifests."""

from __future__ import annotations

import json
import pickle
import platform
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import sklearn
import skimage

from lettuce_benchmark import CLASS_NAMES, N_SPLITS, SEEDS
from lettuce_benchmark.evaluation import (
    PREDICTION_COLUMNS as SHARED_PREDICTION_COLUMNS,
    calculate_metrics,
)
from lettuce_benchmark.splits import validate_split_collection

from .evaluate import (
    build_metric_row,
    summarize_oof_predictions,
    write_confusion_matrix_csv,
)
from .feature_extraction import (
    DEFAULT_IMAGE_SIZE,
    FEATURE_DIMENSIONS,
    FEATURE_GROUPS,
    extract_feature_groups,
)
from .predict import predict_with_confidence
from .train import (
    EXPERIMENTS,
    PRIMARY_METRIC,
    RANDOM_FOREST_CANDIDATES,
    SVM_CANDIDATES,
    build_pipeline,
    select_hyperparameters,
)

PREDICTION_COLUMNS = [
    "image_id",
    "seed",
    "fold",
    "experiment",
    "true_label",
    "predicted_label",
    "confidence",
    "partition",
]


def _collect_splits(splits_dir: Path) -> dict[tuple[int, int], pd.DataFrame]:
    """Load and validate all twelve official split files without modifying them."""
    splits: dict[tuple[int, int], pd.DataFrame] = {}
    for seed in SEEDS:
        for fold in range(1, N_SPLITS + 1):
            path = splits_dir / f"seed_{seed}" / f"fold_{fold}.csv"
            if not path.is_file():
                raise FileNotFoundError(f"Shared split manifest not found: {path}")
            frame = pd.read_csv(path)
            if set(("image_id", "image_path", "label", "partition", "seed", "fold")) - set(frame):
                raise ValueError(f"Split manifest is missing required columns: {path}")
            if set(frame["seed"].astype(int)) != {seed} or set(frame["fold"].astype(int)) != {fold}:
                raise ValueError(f"Manifest seed/fold values do not match its path: {path}")
            splits[(seed, fold)] = frame
    validate_split_collection(splits)
    return splits


def _image_index(
    splits: dict[tuple[int, int], pd.DataFrame],
) -> pd.DataFrame:
    """Build the unique image index and reject inconsistent manifest metadata."""
    rows = pd.concat(
        [frame[["image_id", "image_path", "label"]] for frame in splits.values()],
        ignore_index=True,
    ).drop_duplicates()
    if rows["image_id"].duplicated().any():
        raise ValueError("Shared manifests assign inconsistent path or label to an image_id")
    if set(rows["label"]) != set(CLASS_NAMES):
        raise ValueError("Shared manifests do not contain the expected four classes")
    return rows.sort_values("image_id").reset_index(drop=True)


def _feature_cache(
    images: pd.DataFrame,
    image_root: Path,
    image_size: int,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, float], float]:
    """Extract each unique image once and report total wall-clock feature time."""
    cache: dict[str, dict[str, np.ndarray]] = {}
    per_image_seconds: dict[str, float] = {}
    started = time.perf_counter()
    for row in images.itertuples(index=False):
        image_path = image_root / Path(row.image_path)
        if not image_path.is_file():
            raise FileNotFoundError(f"Manifest image not found: {image_path}")
        image_started = time.perf_counter()
        cache[row.image_id] = extract_feature_groups(image_path, image_size=image_size)
        per_image_seconds[row.image_id] = time.perf_counter() - image_started
    return cache, per_image_seconds, time.perf_counter() - started


def _feature_matrix(
    cache: dict[str, dict[str, np.ndarray]],
    image_ids: pd.Series,
    groups: tuple[str, ...],
) -> np.ndarray:
    """Select ordered feature groups for an ordered sequence of image IDs."""
    matrix = np.vstack(
        [
            np.concatenate([cache[image_id][group] for group in groups])
            for image_id in image_ids
        ]
    )
    if not np.isfinite(matrix).all():
        raise ValueError("Feature matrix contains non-finite values")
    return matrix


def _dump_json(path: Path, value: dict[str, Any]) -> None:
    """Write a JSON object with stable formatting."""
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)


def make_test_prediction_frame(
    test: pd.DataFrame,
    experiment: str,
    seed: int,
    fold: int,
    predicted_labels: np.ndarray,
    confidence: list[float],
) -> pd.DataFrame:
    """Build test-partition predictions using the requested output schema."""
    if len(test) != len(predicted_labels) or len(test) != len(confidence):
        raise ValueError("Test rows, predicted labels, and confidences must have equal lengths")
    return pd.DataFrame(
        {
            "image_id": test["image_id"].to_numpy(),
            "seed": seed,
            "fold": fold,
            "experiment": experiment,
            "true_label": test["label"].to_numpy(),
            "predicted_label": predicted_labels,
            "confidence": confidence,
            "partition": "test",
        },
        columns=PREDICTION_COLUMNS,
    )


def run_experiments(
    image_root: str | Path = "data/raw",
    splits_dir: str | Path = "data/splits",
    output_dir: str | Path = "results/classical_ml",
    image_size: int = DEFAULT_IMAGE_SIZE,
    experiment_names: tuple[str, ...] | None = None,
) -> dict[str, dict[str, Any]]:
    """Run selected experiments on every shared fold and save predictions/metrics.

    Feature extraction is cached once per unique image. Candidate models fit on
    training rows and are selected using validation macro-F1. The selected
    pipeline is then refit on train plus validation before one final test pass.
    """
    root = Path(image_root).expanduser().resolve()
    manifest_root = Path(splits_dir).expanduser().resolve()
    results_root = Path(output_dir).expanduser().resolve()
    if image_size < 1:
        raise ValueError("image_size must be positive")

    selected_experiments = EXPERIMENTS
    if experiment_names is not None:
        unknown = set(experiment_names).difference(item.name for item in EXPERIMENTS)
        if unknown:
            raise ValueError(f"Unknown experiment names: {sorted(unknown)}")
        selected_experiments = tuple(
            item for item in EXPERIMENTS if item.name in experiment_names
        )
        if not selected_experiments:
            raise ValueError("At least one experiment must be selected")

    splits = _collect_splits(manifest_root)
    images = _image_index(splits)
    features, per_image_extraction_seconds, extraction_seconds = _feature_cache(
        images, root, image_size
    )
    results_root.mkdir(parents=True, exist_ok=True)
    confusion_dir = results_root / "confusion_matrices"
    confusion_dir.mkdir(parents=True, exist_ok=True)
    extraction_summary = {
        "unique_images": len(images),
        "feature_groups": list(FEATURE_GROUPS),
        "feature_dimensions": FEATURE_DIMENSIONS,
        "image_size": image_size,
        "total_wall_seconds": extraction_seconds,
        "seconds_per_image": extraction_seconds / len(images),
        "scope": "all feature groups extracted once per unique image and reused",
    }
    prediction_frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    efficiency_rows: list[dict[str, Any]] = [
        {
            "record_type": "shared_feature_cache",
            "experiment": "ALL",
            "seed": None,
            "fold": None,
            "image_count": len(images),
            "feature_extraction_seconds": extraction_seconds,
            "feature_extraction_ms_per_image": (
                extraction_seconds * 1000 / len(images)
            ),
            "hyperparameter_selection_seconds": None,
            "classifier_training_seconds": None,
            "prediction_latency_seconds": None,
            "prediction_latency_ms_per_image": None,
            "model_size_bytes": None,
            "feature_extraction_scope": extraction_summary["scope"],
        }
    ]
    all_results: dict[str, dict[str, Any]] = {}
    experiment_configurations: dict[str, Any] = {}
    for experiment in selected_experiments:
        experiment_prediction_frames: list[pd.DataFrame] = []
        experiment_runs: list[dict[str, Any]] = []

        for (seed, fold), split in sorted(splits.items()):
            partitions = {
                name: split.loc[split["partition"] == name].reset_index(drop=True)
                for name in ("train", "validation", "test")
            }
            train, validation, test = (
                partitions["train"],
                partitions["validation"],
                partitions["test"],
            )
            train_x = _feature_matrix(features, train["image_id"], experiment.feature_groups)
            validation_x = _feature_matrix(
                features, validation["image_id"], experiment.feature_groups
            )
            test_x = _feature_matrix(features, test["image_id"], experiment.feature_groups)
            train_y = train["label"].to_numpy()
            validation_y = validation["label"].to_numpy()
            test_y = test["label"].to_numpy()

            selection_started = time.perf_counter()
            best_parameters, validation_macro_f1 = select_hyperparameters(
                experiment,
                train_x,
                train_y,
                validation_x,
                validation_y,
                seed,
            )
            selection_seconds = time.perf_counter() - selection_started

            development_x = np.vstack((train_x, validation_x))
            development_y = np.concatenate((train_y, validation_y))
            model = build_pipeline(experiment, best_parameters, seed)
            training_started = time.perf_counter()
            model.fit(development_x, development_y)
            training_seconds = time.perf_counter() - training_started

            predicted_labels, confidence, prediction_seconds = predict_with_confidence(
                model, test_x, experiment.classifier
            )
            fold_predictions = make_test_prediction_frame(
                test,
                experiment.name,
                seed,
                fold,
                predicted_labels,
                confidence,
            )
            experiment_prediction_frames.append(fold_predictions)
            prediction_frames.append(fold_predictions)
            test_metrics = calculate_metrics(test_y.tolist(), predicted_labels.tolist())
            confusion_name = f"{experiment.name}_seed_{seed}_fold_{fold}_test.csv"
            write_confusion_matrix_csv(
                test_metrics["confusion_matrix"],
                confusion_dir / confusion_name,
            )
            metric_rows.append(
                build_metric_row(
                    experiment.name,
                    "fold_test",
                    seed,
                    fold,
                    test_metrics,
                    str(Path("confusion_matrices") / confusion_name),
                )
            )
            test_feature_seconds = sum(
                per_image_extraction_seconds[image_id]
                for image_id in test["image_id"]
            )
            model_size_bytes = len(
                pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)
            )
            efficiency_rows.append(
                {
                    "record_type": "test_run",
                    "experiment": experiment.name,
                    "seed": seed,
                    "fold": fold,
                    "image_count": len(test),
                    "feature_extraction_seconds": test_feature_seconds,
                    "feature_extraction_ms_per_image": (
                        test_feature_seconds * 1000 / len(test)
                    ),
                    "hyperparameter_selection_seconds": selection_seconds,
                    "classifier_training_seconds": training_seconds,
                    "prediction_latency_seconds": prediction_seconds,
                    "prediction_latency_ms_per_image": (
                        prediction_seconds * 1000 / len(test)
                    ),
                    "model_size_bytes": model_size_bytes,
                    "feature_extraction_scope": (
                        "sum of measured per-image full-feature extraction times "
                        "for this test fold; features are shared from one cache"
                    ),
                }
            )
            experiment_runs.append(
                {
                    "seed": seed,
                    "fold": fold,
                    "training_images_for_selection": len(train),
                    "validation_images": len(validation),
                    "final_fit_images": len(development_y),
                    "test_images": len(test),
                    "selected_parameters": best_parameters,
                    "validation_macro_f1": validation_macro_f1,
                    "hyperparameter_selection_seconds": selection_seconds,
                    "classifier_training_seconds": training_seconds,
                    "prediction_latency_seconds": prediction_seconds,
                    "model_size_bytes": model_size_bytes,
                }
            )

        experiment_predictions = pd.concat(
            experiment_prediction_frames, ignore_index=True
        )
        oof_rows, results = summarize_oof_predictions(
            experiment_predictions[SHARED_PREDICTION_COLUMNS],
            experiment.name,
            confusion_dir,
        )
        metric_rows.extend(oof_rows)
        all_results[experiment.name] = results
        experiment_configurations[experiment.name] = {
            "feature_groups": list(experiment.feature_groups),
            "feature_dimension": sum(
                FEATURE_DIMENSIONS[group] for group in experiment.feature_groups
            ),
            "classifier": experiment.classifier,
            "runs": experiment_runs,
        }

    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions.to_csv(
        results_root / "predictions.csv",
        index=False,
        columns=PREDICTION_COLUMNS,
    )
    pd.DataFrame(metric_rows).to_csv(results_root / "metrics.csv", index=False)
    pd.DataFrame(efficiency_rows).to_csv(
        results_root / "efficiency.csv", index=False
    )
    _dump_json(
        results_root / "experiment_configurations.json",
        {
            "selection_metric": PRIMARY_METRIC,
            "validation_protocol": (
                "Fit each candidate scaler/classifier on train only; select by "
                "validation macro-F1; refit selected pipeline on train+validation; "
                "evaluate test once."
            ),
            "confidence": (
                "SVM predicted-class softmax score (uncalibrated); Random Forest "
                "predicted-class probability."
            ),
            "seeds": list(SEEDS),
            "folds_per_seed": N_SPLITS,
            "class_names": list(CLASS_NAMES),
            "prediction_columns": PREDICTION_COLUMNS,
            "feature_extraction": extraction_summary,
            "search_spaces": {
                "svm": {
                    "kernel": "rbf",
                    "class_weight": "balanced",
                    "candidates": list(SVM_CANDIDATES),
                },
                "random_forest": {
                    "class_weight": "balanced",
                    "candidates": list(RANDOM_FOREST_CANDIDATES),
                },
            },
            "versions": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "opencv": cv2.__version__,
                "scikit_image": skimage.__version__,
                "scikit_learn": sklearn.__version__,
            },
            "experiments": experiment_configurations,
        },
    )
    return all_results


def main() -> None:
    """Run experiments from command-line arguments."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", default="data/raw")
    parser.add_argument("--splits-dir", default="data/splits")
    parser.add_argument("--output-dir", default="results/classical_ml")
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=[experiment.name for experiment in EXPERIMENTS],
        default=None,
    )
    arguments = parser.parse_args()
    run_experiments(
        image_root=arguments.image_root,
        splits_dir=arguments.splits_dir,
        output_dir=arguments.output_dir,
        image_size=arguments.image_size,
        experiment_names=(
            tuple(arguments.experiments) if arguments.experiments else None
        ),
    )


if __name__ == "__main__":
    main()
