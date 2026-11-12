"""Random seed initialization and experiment metadata helpers."""

from __future__ import annotations

import importlib.metadata
import platform
import random
from typing import Any

import numpy as np


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch (including CUDA when available)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def software_versions() -> dict[str, str]:
    """Return installed versions for benchmark dependencies."""
    packages = (
        "numpy",
        "pandas",
        "scikit-learn",
        "torch",
        "torchvision",
        "ultralytics",
        "opencv-python-headless",
        "scikit-image",
        "Pillow",
        "matplotlib",
        "seaborn",
        "PyYAML",
    )
    versions = {"python": platform.python_version()}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def experiment_metadata(
    model_name: str,
    seed: int,
    fold: int,
    training_config: dict[str, Any],
) -> dict[str, Any]:
    """Capture model/run configuration, hardware details, and software versions."""
    hardware = {"platform": platform.platform(), "processor": platform.processor()}
    try:
        import torch
    except ImportError:
        hardware.update({"device": "unavailable", "cuda_available": False})
    else:
        cuda_available = torch.cuda.is_available()
        hardware.update({
            "device": "cuda" if cuda_available else "cpu",
            "cuda_available": cuda_available,
        })
        if cuda_available:
            hardware["gpu_name"] = torch.cuda.get_device_name(0)
    return {
        "model_name": model_name,
        "seed": int(seed),
        "fold": int(fold),
        "hardware": hardware,
        "training_config": training_config,
        "software_versions": software_versions(),
    }
