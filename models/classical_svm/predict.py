"""Prediction and confidence-score helpers for classical classifiers."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from sklearn.pipeline import Pipeline


def predict_with_confidence(
    model: Pipeline,
    features: np.ndarray,
    classifier: str,
) -> tuple[np.ndarray, list[float], float]:
    """Predict labels, matching confidence scores, and elapsed inference time."""
    if classifier not in {"svm", "random_forest"}:
        raise ValueError(f"Unknown classifier: {classifier}")
    started = time.perf_counter()
    predicted_labels = model.predict(features)
    if classifier == "svm":
        decision_scores = model.decision_function(features)
        shifted_scores = decision_scores - decision_scores.max(axis=1, keepdims=True)
        exponentiated_scores = np.exp(shifted_scores)
        confidence_scores = exponentiated_scores / exponentiated_scores.sum(
            axis=1, keepdims=True
        )
    else:
        confidence_scores = model.predict_proba(features)
    prediction_seconds = time.perf_counter() - started

    classes = model.named_steps["classifier"].classes_
    class_positions = {label: index for index, label in enumerate(classes)}
    confidence = [
        float(confidence_scores[row, class_positions[label]])
        for row, label in enumerate(predicted_labels)
    ]
    if not np.isfinite(confidence).all() or not np.isfinite(confidence_scores).all():
        raise ValueError("Non-finite prediction confidence")
    return predicted_labels, confidence, prediction_seconds
