"""Color statistics and histogram features for RGB lettuce images."""

from __future__ import annotations

import cv2
import numpy as np

COLOR_STATISTICS_DIMENSION = 18
COLOR_HISTOGRAMS_DIMENSION = 96


def color_statistics(rgb: np.ndarray) -> np.ndarray:
    """Compute channel-wise means and standard deviations in RGB, HSV, and LAB."""
    channels = (
        rgb,
        cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV),
        cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB),
    )
    return np.concatenate(
        [
            np.concatenate((values.mean(axis=(0, 1)), values.std(axis=(0, 1))))
            for values in channels
        ]
    ).astype(np.float64)


def color_histograms(rgb: np.ndarray) -> np.ndarray:
    """Compute normalized 16-bin histograms for RGB and HSV channels."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    histograms = []
    for image, ranges in ((rgb, (0, 256)), (hsv, (0, 256))):
        for channel in range(3):
            channel_range = (0, 180) if image is hsv and channel == 0 else ranges
            histogram = cv2.calcHist([image], [channel], None, [16], channel_range)
            histogram = histogram.reshape(-1).astype(np.float64)
            histogram /= histogram.sum()
            histograms.append(histogram)
    return np.concatenate(histograms)
