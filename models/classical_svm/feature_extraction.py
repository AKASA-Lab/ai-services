"""Watermark-safe color and texture feature extraction."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image

from lettuce_benchmark.preprocessing import watermark_safe_crop

from .color_features import (
    COLOR_HISTOGRAMS_DIMENSION,
    COLOR_STATISTICS_DIMENSION,
    color_histograms,
    color_statistics,
)
from .texture_features import (
    GLCM_DIMENSION,
    LBP_DIMENSION,
    glcm_features,
    lbp_histogram,
)

FEATURE_DIMENSIONS = {
    "color_stats": COLOR_STATISTICS_DIMENSION,
    "color_histograms": COLOR_HISTOGRAMS_DIMENSION,
    "glcm": GLCM_DIMENSION,
    "lbp": LBP_DIMENSION,
}
FEATURE_GROUPS = tuple(FEATURE_DIMENSIONS)
DEFAULT_IMAGE_SIZE = 256


def _read_rgb_image(image: Image.Image | np.ndarray | str | Path) -> Image.Image:
    """Load an image as RGB without bypassing the shared safe crop."""
    if isinstance(image, (str, Path)):
        with Image.open(image) as opened:
            return opened.convert("RGB")
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("Image arrays must have shape (height, width, 3) in RGB order")
    if array.dtype != np.uint8:
        if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
            raise ValueError("Image arrays must contain finite numeric values")
        if array.min() < 0 or array.max() > 255:
            raise ValueError("Image array values must be in the range [0, 255]")
        array = array.astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def _prepare_image(
    image: Image.Image | np.ndarray | str | Path,
    image_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the shared crop, resize, and return RGB and grayscale arrays."""
    if image_size < 1:
        raise ValueError("image_size must be positive")
    cropped = watermark_safe_crop(_read_rgb_image(image))
    rgb = cv2.resize(
        np.asarray(cropped),
        (image_size, image_size),
        interpolation=cv2.INTER_AREA,
    )
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return rgb, gray


def extract_feature_groups(
    image: Image.Image | np.ndarray | str | Path,
    image_size: int = DEFAULT_IMAGE_SIZE,
) -> dict[str, np.ndarray]:
    """Extract all configured feature groups after applying the shared safe crop."""
    rgb, gray = _prepare_image(image, image_size)
    features = {
        "color_stats": color_statistics(rgb),
        "color_histograms": color_histograms(rgb),
        "glcm": glcm_features(gray),
        "lbp": lbp_histogram(gray),
    }
    for name, vector in features.items():
        expected = FEATURE_DIMENSIONS[name]
        if vector.shape != (expected,) or not np.isfinite(vector).all():
            raise ValueError(f"Invalid {name} feature vector: shape={vector.shape}")
    return features


def extract_image_features(
    image: Image.Image | np.ndarray | str | Path,
    feature_groups: Iterable[str] = FEATURE_GROUPS,
    image_size: int = DEFAULT_IMAGE_SIZE,
) -> np.ndarray:
    """Extract selected feature groups into one finite, one-dimensional vector."""
    selected = tuple(feature_groups)
    unknown = set(selected).difference(FEATURE_DIMENSIONS)
    if unknown:
        raise ValueError(f"Unknown feature groups: {sorted(unknown)}")
    if not selected:
        raise ValueError("At least one feature group must be selected")
    groups = extract_feature_groups(image, image_size=image_size)
    return np.concatenate([groups[group] for group in selected])
