"""Training pipelines and validation-only hyperparameter selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from lettuce_benchmark.evaluation import calculate_metrics

from .feature_extraction import FEATURE_GROUPS


@dataclass(frozen=True)
class Experiment:
    """Feature groups and estimator family for one named experiment."""

    name: str
    feature_groups: tuple[str, ...]
    classifier: str


EXPERIMENTS = (
    Experiment("E1", ("color_stats",), "svm"),
    Experiment("E2", ("color_histograms",), "svm"),
    Experiment("E3", ("glcm", "lbp"), "svm"),
    Experiment("E4", FEATURE_GROUPS, "svm"),
    Experiment("E5", FEATURE_GROUPS, "random_forest"),
)
SVM_CANDIDATES = (
    {"C": 0.1, "gamma": "scale"},
    {"C": 0.1, "gamma": 0.01},
    {"C": 0.1, "gamma": 0.001},
    {"C": 1.0, "gamma": "scale"},
    {"C": 1.0, "gamma": 0.01},
    {"C": 1.0, "gamma": 0.001},
    {"C": 10.0, "gamma": "scale"},
    {"C": 10.0, "gamma": 0.01},
    {"C": 10.0, "gamma": 0.001},
)
RANDOM_FOREST_CANDIDATES = tuple(
    {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
        "max_features": "sqrt",
    }
    for n_estimators in (100, 300)
    for max_depth in (None, 20)
    for min_samples_leaf in (1, 2)
)
PRIMARY_METRIC = "macro_f1"


def build_pipeline(
    experiment: Experiment,
    parameters: dict[str, Any],
    seed: int,
) -> Pipeline:
    """Construct a scaler-first pipeline and the requested classifier."""
    if experiment.classifier == "svm":
        classifier = SVC(
            kernel="rbf",
            class_weight="balanced",
            decision_function_shape="ovr",
            random_state=seed,
            **parameters,
        )
    else:
        classifier = RandomForestClassifier(
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
            **parameters,
        )
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", classifier),
        ]
    )


def select_hyperparameters(
    experiment: Experiment,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    seed: int,
) -> tuple[dict[str, Any], float]:
    """Select parameters by validation macro-F1 using training rows to fit."""
    candidates = (
        SVM_CANDIDATES
        if experiment.classifier == "svm"
        else RANDOM_FOREST_CANDIDATES
    )
    best_parameters: dict[str, Any] | None = None
    best_score = -1.0
    for parameters in candidates:
        model = build_pipeline(experiment, parameters, seed)
        model.fit(train_features, train_labels)
        validation_predictions = model.predict(validation_features)
        score = calculate_metrics(
            validation_labels.tolist(),
            validation_predictions.tolist(),
        )[PRIMARY_METRIC]
        if score > best_score:
            best_parameters = dict(parameters)
            best_score = float(score)
    if best_parameters is None:
        raise RuntimeError(f"No hyperparameter candidates for {experiment.name}")
    return best_parameters, best_score
