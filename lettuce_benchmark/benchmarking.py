"""Model and inference efficiency measurement helpers."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable


def parameter_count(model: Any) -> int:
    """Count all parameters in a PyTorch model."""
    return sum(parameter.numel() for parameter in model.parameters())


def checkpoint_size_bytes(path: str | Path) -> int:
    """Return the size of an existing checkpoint file in bytes."""
    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    return checkpoint.stat().st_size


def _synchronize(device: Any) -> None:
    import torch

    if getattr(device, "type", str(device).split(":")[0]) == "cuda":
        torch.cuda.synchronize(device)


def inference_latency(
    model: Any,
    sample: Any,
    device: Any,
    warmup: int = 10,
    iterations: int = 100,
) -> dict[str, float]:
    """Measure batch-one inference latency with warm-up and CUDA synchronization."""
    if warmup < 0 or iterations < 1:
        raise ValueError("warmup must be non-negative and iterations must be positive")
    import torch

    model = model.to(device).eval()
    sample = sample.to(device)
    if sample.shape[0] != 1:
        raise ValueError("Inference latency must be measured with batch size 1")
    with torch.inference_mode():
        for _ in range(warmup):
            model(sample)
        _synchronize(device)
        started = time.perf_counter()
        for _ in range(iterations):
            model(sample)
        _synchronize(device)
    total = time.perf_counter() - started
    return {
        "mean_seconds": total / iterations,
        "mean_milliseconds": total * 1000 / iterations,
        "iterations": float(iterations),
        "warmup_iterations": float(warmup),
    }


def measure_callable(function: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[Any, float]:
    """Run a callable once and return its result and elapsed wall-clock seconds."""
    started = time.perf_counter()
    result = function(*args, **kwargs)
    return result, time.perf_counter() - started


def measure_training(training_function: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[Any, float]:
    """Run one training procedure and return its result and wall-clock duration."""
    return measure_callable(training_function, *args, **kwargs)


def measure_classical_prediction(
    feature_extractor: Callable[..., Any],
    classifier: Any,
    images: Any,
) -> dict[str, Any]:
    """Time classical feature extraction and classifier prediction separately."""
    features, extraction_seconds = measure_callable(feature_extractor, images)
    predictions, prediction_seconds = measure_callable(classifier.predict, features)
    return {
        "predictions": predictions,
        "feature_extraction_seconds": extraction_seconds,
        "classifier_prediction_seconds": prediction_seconds,
    }


def peak_gpu_memory_bytes(device: Any) -> int | None:
    """Return peak CUDA allocation since reset, or ``None`` for non-CUDA devices."""
    import torch

    if getattr(device, "type", str(device).split(":")[0]) != "cuda" or not torch.cuda.is_available():
        return None
    torch.cuda.synchronize(device)
    return int(torch.cuda.max_memory_allocated(device))
