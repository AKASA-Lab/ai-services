"""Tests for classical feature extraction dimensions and determinism."""

import numpy as np
import pandas as pd
from PIL import Image

from models.classical_svm.feature_extraction import (
    FEATURE_DIMENSIONS,
    extract_feature_groups,
    extract_image_features,
)
from models.classical_svm.pipeline import (
    PREDICTION_COLUMNS,
    make_test_prediction_frame,
)
from models.classical_svm.train import (
    RANDOM_FOREST_CANDIDATES,
    SVM_CANDIDATES,
)


def test_feature_group_dimensions_values_and_histogram_normalization() -> None:
    image = np.zeros((1024, 1024, 3), dtype=np.uint8)
    image[:, :, 0] = np.arange(1024, dtype=np.uint16)[None, :] % 256
    image[:, :, 1] = np.arange(1024, dtype=np.uint16)[:, None] % 256
    groups = extract_feature_groups(Image.fromarray(image), image_size=32)

    assert {name: values.size for name, values in groups.items()} == FEATURE_DIMENSIONS
    assert all(np.isfinite(values).all() for values in groups.values())
    assert np.isclose(groups["lbp"].sum(), 1.0)
    assert np.isclose(groups["color_histograms"][:16].sum(), 1.0)


def test_feature_extraction_is_consistent_for_same_image() -> None:
    image = Image.new("RGB", (1024, 1024), color=(72, 131, 46))
    first = extract_image_features(image, image_size=24)
    second = extract_image_features(image, image_size=24)

    assert first.shape == (sum(FEATURE_DIMENSIONS.values()),)
    assert np.isfinite(first).all()
    np.testing.assert_array_equal(first, second)


def test_runner_prediction_frame_uses_requested_schema_and_test_partition() -> None:
    test = pd.DataFrame(
        {
            "image_id": ["image-a", "image-b"],
            "label": ["FN", "-N"],
        }
    )
    predictions = make_test_prediction_frame(
        test,
        experiment="E4",
        seed=42,
        fold=1,
        predicted_labels=np.asarray(["FN", "-K"]),
        confidence=[0.8, 0.6],
    )

    assert list(predictions.columns) == PREDICTION_COLUMNS
    assert predictions["experiment"].tolist() == ["E4", "E4"]
    assert predictions["partition"].tolist() == ["test", "test"]
    assert predictions["confidence"].between(0, 1).all()


def test_runner_rejects_mismatched_prediction_counts() -> None:
    test = pd.DataFrame({"image_id": ["image-a"], "label": ["FN"]})

    with np.testing.assert_raises_regex(ValueError, "must have equal lengths"):
        make_test_prediction_frame(
            test,
            experiment="E4",
            seed=42,
            fold=1,
            predicted_labels=np.asarray(["FN", "-N"]),
            confidence=[0.8],
        )


def test_runner_search_spaces_cover_requested_svm_and_forest_parameters() -> None:
    assert len(SVM_CANDIDATES) == 9
    assert {candidate["C"] for candidate in SVM_CANDIDATES} == {0.1, 1.0, 10.0}
    assert {candidate["gamma"] for candidate in SVM_CANDIDATES} == {
        "scale",
        0.01,
        0.001,
    }

    assert len(RANDOM_FOREST_CANDIDATES) == 8
    assert {candidate["n_estimators"] for candidate in RANDOM_FOREST_CANDIDATES} == {
        100,
        300,
    }
    assert {candidate["max_depth"] for candidate in RANDOM_FOREST_CANDIDATES} == {
        None,
        20,
    }
    assert {
        candidate["min_samples_leaf"] for candidate in RANDOM_FOREST_CANDIDATES
    } == {1, 2}
