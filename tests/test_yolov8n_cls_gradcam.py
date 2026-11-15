from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from lettuce_benchmark import CLASS_NAMES
from models.yolov8n_cls.gradcam import (
    YOLOv8GradCAM,
    find_image_holdout_fold,
    load_holdout_model_for_image,
    overlay,
    plant_mask,
    resolve_checkpoint_path,
    select_gradcam_samples,
)


class DummyHead(nn.Module):
    """Synthetic classification head simulating Ultralytics Classify head."""

    def __init__(self, in_channels: int = 16, num_classes: int = 4) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(0.0)
        self.linear = nn.Linear(in_channels, num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))
        return features.softmax(1), features


class DummyYOLOModel(nn.Module):
    """Synthetic model simulating YOLOv8n-cls without requiring trained weights."""

    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.Conv2d(8, 16, 3, padding=1),  # acts as model[-2]
            DummyHead(16, 4),                 # acts as model[-1]
        )
        # Native Ultralytics alphabetical order
        self.names = {0: "-K", 1: "-N", 2: "-P", 3: "FN"}

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(x)


def test_gradcam_shape_and_range() -> None:
    torch.manual_seed(42)
    model = DummyYOLOModel().eval()
    cam_engine = YOLOv8GradCAM(model)

    x = torch.randn(1, 3, 224, 224)
    cam, probs, pred_idx = cam_engine(x, target_class="FN")

    assert isinstance(cam, np.ndarray)
    assert cam.shape == (224, 224)
    assert 0.0 <= float(cam.min()) <= float(cam.max()) <= 1.0
    assert len(probs) == 4
    assert np.isclose(probs.sum(), 1.0, atol=1e-5)
    assert 0 <= pred_idx < 4

    cam_engine.remove_hooks()


def test_gradcam_class_order_mapping() -> None:
    model = DummyYOLOModel().eval()
    # Force linear weights to bias toward native index 0 ('-K')
    with torch.no_grad():
        model.model[-1].linear.weight.zero_()
        model.model[-1].linear.bias.copy_(torch.tensor([10.0, 0.0, 0.0, 0.0]))

    cam_engine = YOLOv8GradCAM(model)
    x = torch.randn(1, 3, 224, 224)
    cam, probs, pred_idx = cam_engine(x)

    # In canonical CLASS_NAMES = ['FN', '-N', '-P', '-K'], '-K' is index 3
    assert CLASS_NAMES[pred_idx] == "-K"
    assert probs[CLASS_NAMES.index("-K")] == pytest.approx(probs.max(), abs=1e-4)

    cam_engine.remove_hooks()


def test_sample_selection_is_deterministic(tmp_path: Path) -> None:
    # Build synthetic predictions and audit CSVs
    preds_rows = []
    audit_rows = []
    image_counter = 0

    for label in CLASS_NAMES:
        for f in range(1, 5):
            for i in range(3):
                image_id = f"img_{label}_{f}_{i}"
                conf = 0.95 - 0.1 * i
                preds_rows.append({
                    "image_id": image_id, "seed": 42, "fold": f,
                    "true_label": label, "predicted_label": label if i < 2 else "-K",
                    "confidence": conf,
                })
                audit_rows.append({
                    "image_id": image_id, "image_path": f"{label}/{image_id}.png",
                    "label": label, "width": 1024 if label == "FN" else 1023,
                    "height": 1024 if label == "FN" else 1005,
                    "dalle_colour_bar_detected": label == "FN",
                })
                image_counter += 1

    preds_file = tmp_path / "predictions.csv"
    audit_file = tmp_path / "audit.csv"
    pd.DataFrame(preds_rows).to_csv(preds_file, index=False)
    pd.DataFrame(audit_rows).to_csv(audit_file, index=False)

    run1 = select_gradcam_samples(preds_file, audit_file, preds_file, seed=42)
    run2 = select_gradcam_samples(preds_file, audit_file, preds_file, seed=42)

    assert run1["correct"]["image_id"].tolist() == run2["correct"]["image_id"].tolist()
    assert run1["healthy_fn"]["image_id"].tolist() == run2["healthy_fn"]["image_id"].tolist()
    assert run1["misclassified"]["image_id"].tolist() == run2["misclassified"]["image_id"].tolist()


def test_holdout_fold_lookup() -> None:
    # Use real results/member_4/predictions.csv in the repository
    preds_path = Path("results/member_4/predictions.csv")
    if preds_path.is_file():
        df = pd.read_csv(preds_path, dtype={"image_id": str})
        sample_row = df.iloc[0]
        seed, fold = find_image_holdout_fold(sample_row["image_id"], preds_path, seed=sample_row["seed"])
        assert fold in [1, 2, 3, 4]
        assert seed == sample_row["seed"]

        ckpt_path = resolve_checkpoint_path(seed, fold, "dummy_checkpoints")
        assert ckpt_path.name == f"yolov8n_cls_seed_{seed}_fold_{fold}.pt"


def test_missing_checkpoint_raises_filenotfound() -> None:
    preds_path = Path("results/member_4/predictions.csv")
    if preds_path.is_file():
        df = pd.read_csv(preds_path, dtype={"image_id": str})
        sample_id = str(df.iloc[0]["image_id"])
        with pytest.raises(FileNotFoundError, match="Checkpoint file missing"):
            load_holdout_model_for_image(
                sample_id,
                predictions_path=preds_path,
                checkpoints_dir="nonexistent_dir",
                seed=42,
            )


def test_resolve_image_path_hex_hash() -> None:
    """Regression test: image_id is a hex hash, NOT a file path.

    The old fallback f"{label}/{image_id}.png" (e.g. '-K/a191e9b7b9d70a56.png')
    is wrong because the real path is '-K/k_1.png'. This test ensures
    resolve_image_path() correctly looks up the real path from audit/metadata.
    """
    from models.yolov8n_cls.shortcut_analysis import resolve_image_path

    # Build a synthetic audit DataFrame with real-format hex-hash IDs
    audit_df = pd.DataFrame({
        "image_id": ["a914de3754e50ed7", "b8291fcae0e4d231", "cdef012345678900"],
        "image_path": ["FN/fn1.png", "-K/k_1.png", "-N/n_5.png"],
        "label": ["FN", "-K", "-N"],
    })

    # Successful lookup by hex hash
    assert resolve_image_path("a914de3754e50ed7", audit_or_metadata_df=audit_df) == "FN/fn1.png"
    assert resolve_image_path("b8291fcae0e4d231", audit_or_metadata_df=audit_df) == "-K/k_1.png"
    assert resolve_image_path("cdef012345678900", audit_or_metadata_df=audit_df) == "-N/n_5.png"

    # The old fallback would have returned the WRONG path
    wrong_fallback = f"-K/b8291fcae0e4d231.png"  # hash-based, not the real name
    correct_path = resolve_image_path("b8291fcae0e4d231", audit_or_metadata_df=audit_df)
    assert correct_path != wrong_fallback, (
        f"Path resolution should NOT return the old hash-based fallback. "
        f"Got '{correct_path}', old fallback would be '{wrong_fallback}'"
    )

    # Missing ID raises KeyError
    with pytest.raises(KeyError, match="Cannot resolve image_path"):
        resolve_image_path("0000000000000000", audit_or_metadata_df=audit_df,
                           metadata_path=Path("nonexistent/metadata.csv"))


def test_plant_mask_and_overlay() -> None:
    rgb = np.zeros((224, 224, 3), dtype=np.uint8)
    rgb[50:150, 50:150] = [30, 200, 50]  # green plant region
    mask = plant_mask(rgb)
    assert mask.shape == (224, 224)
    assert mask[100, 100] == True
    assert mask[10, 10] == False

    cam = np.zeros((224, 224), dtype=np.float32)
    cam[50:150, 50:150] = 0.8
    fig, ax = plt.subplots()
    overlay(ax, rgb.astype(float) / 255.0, cam, "test title")
    plt.close(fig)


def test_gradcam_examples_only_contain_correct_samples(tmp_path: Path) -> None:
    """Test that every sample selected for the examples figure has true == predicted."""
    preds_rows = []
    audit_rows = []
    for label in CLASS_NAMES:
        for i in range(5):
            img_id = f"test_{label}_{i}"
            # i == 0: misclassified, i > 0: correct
            pred = "-K" if (i == 0 and label != "-K") else ("-N" if i == 0 else label)
            preds_rows.append({
                "image_id": img_id,
                "seed": 42,
                "fold": 1,
                "true_label": label,
                "predicted_label": pred,
                "confidence": 0.99 - 0.1 * i,
            })
            audit_rows.append({
                "image_id": img_id,
                "image_path": f"{label}/{img_id}.png",
                "label": label,
                "width": 1024,
                "height": 1024,
            })
    preds_file = tmp_path / "preds.csv"
    audit_file = tmp_path / "audit.csv"
    pd.DataFrame(preds_rows).to_csv(preds_file, index=False)
    pd.DataFrame(audit_rows).to_csv(audit_file, index=False)

    samples = select_gradcam_samples(preds_file, audit_file, preds_file, seed=42)
    correct_samples = samples["correct"]
    assert len(correct_samples) > 0
    assert (correct_samples["true_label"] == correct_samples["predicted_label"]).all()

    # Also verify on real predictions if present
    real_preds = Path("results/member_4/predictions.csv")
    real_audit = Path("data/processed/audit.csv")
    if real_preds.is_file() and real_audit.is_file():
        real_samples = select_gradcam_samples(real_preds, real_audit, real_preds, seed=42)
        assert (real_samples["correct"]["true_label"] == real_samples["correct"]["predicted_label"]).all()
