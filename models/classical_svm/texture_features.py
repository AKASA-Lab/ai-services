"""GLCM and uniform LBP texture features."""

from __future__ import annotations

import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

GLCM_DIMENSION = 4
LBP_POINTS = 8
LBP_DIMENSION = LBP_POINTS + 2
GLCM_LEVELS = 32
GLCM_DISTANCES = (1, 2)
GLCM_ANGLES = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)


def glcm_features(gray: np.ndarray) -> np.ndarray:
    """Average contrast, homogeneity, energy, and correlation over angles/distances."""
    quantized = (gray.astype(np.uint16) * GLCM_LEVELS // 256).astype(np.uint8)
    matrix = graycomatrix(
        quantized,
        distances=list(GLCM_DISTANCES),
        angles=list(GLCM_ANGLES),
        levels=GLCM_LEVELS,
        symmetric=True,
        normed=True,
    )
    properties = ("contrast", "homogeneity", "energy", "correlation")
    return np.asarray(
        [graycoprops(matrix, prop).mean() for prop in properties],
        dtype=np.float64,
    )


def lbp_histogram(gray: np.ndarray) -> np.ndarray:
    """Return a normalized 8-neighbor, radius-1 uniform LBP histogram."""
    lbp = local_binary_pattern(gray, P=LBP_POINTS, R=1, method="uniform")
    histogram, _ = np.histogram(
        lbp,
        bins=LBP_DIMENSION,
        range=(-0.5, LBP_POINTS + 1.5),
    )
    normalized = histogram.astype(np.float64)
    normalized /= normalized.sum()
    return normalized
