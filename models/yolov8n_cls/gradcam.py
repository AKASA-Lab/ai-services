"""Grad-CAM for YOLOv8n-cls (Person 4).

Layer selection rationale:
In YOLOv8n-cls (Ultralytics classification architecture), the model backbone consists of
9 sequential modules (indices 0 to 8 in model.model), followed by the classification head
(index 9, an instance of ultralytics.nn.modules.head.Classify).
Module index 8 (`model.model.model[-2]` on a loaded YOLO instance, or `model.model[-2]` on
the bare ClassificationModel) is the final C2f cross-stage partial bottleneck block.
It outputs 256 feature channels at 7x7 spatial resolution for a 224x224 input.
We hook this layer because it retains 2D spatial coordinate information at the highest semantic
level immediately before global average pooling collapses spatial dimensions to 1x1 in the
Classify head (`conv -> pool -> drop -> linear`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from lettuce_benchmark import CLASS_NAMES
from models.yolov8n_cls.train import to_canonical_order


class YOLOv8GradCAM:
    """Computes Grad-CAM heatmaps for YOLOv8 classification models.

    Hooks the final feature layer before global pooling (`model.model[-2]`) and captures
    pre-softmax logits from the classifier head so gradients are unattenuated by softmax saturation.
    """

    def __init__(self, model: Any, target_layer: nn.Module | None = None) -> None:
        self.model = model
        self.bare_model = self._extract_bare_model(model)
        names = getattr(model, "names", getattr(self.bare_model, "names", None))
        if names is not None and sorted(names.values()) == sorted(CLASS_NAMES):
            self.names = names
        else:
            # Default Ultralytics alphabetical folder order for the 4 classes
            self.names = {0: "-K", 1: "-N", 2: "-P", 3: "FN"}

        if target_layer is None:
            self.target_layer = self._find_target_layer(self.bare_model)
        else:
            self.target_layer = target_layer

        self.linear_layer = self._find_linear_layer(self.bare_model)

        self._activations: list[torch.Tensor] = []
        self._gradients: list[torch.Tensor] = []
        self._captured_logits: list[torch.Tensor] = []

        self._fwd_hook = self.target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = self.target_layer.register_full_backward_hook(self._save_gradient)
        self._linear_hook = (
            self.linear_layer.register_forward_hook(self._save_logits)
            if self.linear_layer is not None
            else None
        )

    @staticmethod
    def _extract_bare_model(model: Any) -> nn.Module:
        """Extract underlying PyTorch nn.Module from YOLO or ClassificationModel."""
        if hasattr(model, "model") and isinstance(model.model, nn.Module):
            if hasattr(model.model, "model") and isinstance(model.model.model, (nn.Sequential, nn.ModuleList)):
                return model.model
            return model.model
        return model

    @staticmethod
    def _find_target_layer(bare_model: nn.Module) -> nn.Module:
        """Find the last feature extraction layer before the classification head."""
        if hasattr(bare_model, "model") and hasattr(bare_model.model, "__getitem__"):
            # model.model[-2] is the final C2f backbone block
            return bare_model.model[-2]
        # Fallback: scan backwards for the last 2D convolutional/block layer
        modules = [m for m in bare_model.modules() if isinstance(m, (nn.Conv2d, nn.Sequential))]
        if modules:
            return modules[-1]
        raise ValueError("Could not automatically locate the target feature layer in the model.")

    @staticmethod
    def _find_linear_layer(bare_model: nn.Module) -> nn.Module | None:
        """Find the linear classification layer to extract un-softmaxed logits."""
        if hasattr(bare_model, "model") and hasattr(bare_model.model, "__getitem__"):
            head = bare_model.model[-1]
            if hasattr(head, "linear") and isinstance(head.linear, nn.Module):
                return head.linear
        for m in bare_model.modules():
            if isinstance(m, nn.Linear):
                return m
        return None

    def _save_activation(self, module: nn.Module, input: Any, output: torch.Tensor) -> None:
        self._activations.append(output)

    def _save_gradient(self, module: nn.Module, grad_input: Any, grad_output: tuple[torch.Tensor, ...]) -> None:
        self._gradients.append(grad_output[0])

    def _save_logits(self, module: nn.Module, input: Any, output: torch.Tensor) -> None:
        self._captured_logits.append(output)

    def remove_hooks(self) -> None:
        """Remove PyTorch forward/backward hooks."""
        self._fwd_hook.remove()
        self._bwd_hook.remove()
        if self._linear_hook is not None:
            self._linear_hook.remove()

    def __call__(
        self,
        image_tensor: torch.Tensor,
        target_class: int | str | None = None,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Compute Grad-CAM for a normalized input image tensor.

        Args:
            image_tensor: (3, H, W) or (1, 3, H, W) float tensor.
            target_class: Canonical class name ('FN', '-N', '-P', '-K') or canonical index.
                          If None, defaults to the predicted class.

        Returns:
            cam: 2D numpy array (H, W) normalized to [0, 1].
            probabilities: 1D numpy array of class probabilities in canonical order.
            predicted_class: Canonical integer index of the predicted class.
        """
        self._activations.clear()
        self._gradients.clear()
        self._captured_logits.clear()

        if image_tensor.ndim == 3:
            image_tensor = image_tensor.unsqueeze(0)

        device = next(self.bare_model.parameters()).device
        image_tensor = image_tensor.to(device)

        # Ultralytics models freeze gradients in eval mode; enable gradients for Grad-CAM
        with torch.enable_grad():
            self.bare_model.zero_grad(set_to_none=True)
            image_tensor = image_tensor.clone().requires_grad_(True)
            output = self.bare_model(image_tensor)

            # Extract raw logits (pre-softmax)
            if self._captured_logits:
                logits = self._captured_logits[-1]
            elif isinstance(output, (tuple, list)) and len(output) > 1:
                logits = output[1]
            elif isinstance(output, torch.Tensor):
                logits = output
            else:
                raise RuntimeError("Failed to extract logits from model forward pass.")

            softmax_probs = F.softmax(logits, dim=-1)[0].detach().cpu().numpy()
            canonical_probs = to_canonical_order(softmax_probs[np.newaxis, :], self.names)[0]
            predicted_canonical_idx = int(canonical_probs.argmax())

            # Resolve target class index in model's native numbering
            if target_class is None:
                native_target_idx = int(logits[0].argmax().item())
            elif isinstance(target_class, str):
                if target_class not in CLASS_NAMES:
                    raise ValueError(f"Unknown class name: {target_class}. Must be one of {CLASS_NAMES}")
                # Find native index for this class name
                name_to_native = {name: idx for idx, name in self.names.items()}
                native_target_idx = name_to_native[target_class]
            else:
                # target_class is canonical index
                canonical_name = CLASS_NAMES[target_class]
                name_to_native = {name: idx for idx, name in self.names.items()}
                native_target_idx = name_to_native[canonical_name]

            # Backward pass on target logit
            target_score = logits[0, native_target_idx]
            target_score.backward()

        if not self._activations or not self._gradients:
            raise RuntimeError("Hooks failed to capture activations or gradients.")

        activations = self._activations[-1]
        gradients = self._gradients[-1]

        # Global average pooling of gradients -> importance weights
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * activations).sum(dim=1, keepdim=True))

        # Upsample to input image dimensions
        input_size = image_tensor.shape[-2:]
        cam = F.interpolate(cam, size=input_size, mode="bilinear", align_corners=False)[0, 0]
        cam = cam.detach().cpu().numpy()

        # Min-max normalization to [0, 1]
        cam_min, cam_max = float(cam.min()), float(cam.max())
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam, canonical_probs, predicted_canonical_idx


# --------------------------------------------------------------------------- Checkpoint Resolution
def find_image_holdout_fold(
    image_id: str,
    predictions_path: Path | str = Path("results/member_4/predictions.csv"),
    seed: int = 42,
) -> tuple[int, int]:
    """Look up the seed and fold in which ``image_id`` was held out in the test set.

    Args:
        image_id: Image identifier string (e.g. 'fn1', '7275c536f7de381c', etc.).
        predictions_path: Path to the held-out predictions CSV.
        seed: Random seed partition (default 42).

    Returns:
        (seed, fold): Fold is 1-indexed (1..4).
    """
    df = pd.read_csv(predictions_path, dtype={"image_id": str})
    subset = df[(df["image_id"] == str(image_id)) & (df["seed"] == int(seed))]
    if subset.empty:
        raise ValueError(f"Image ID '{image_id}' not found in predictions for seed {seed}")
    fold = int(subset.iloc[0]["fold"])
    return int(seed), fold


def resolve_checkpoint_path(
    seed: int,
    fold: int,
    checkpoints_dir: Path | str = Path("outputs/yolov8n_cls/checkpoints"),
) -> Path:
    """Return expected path to the FP16 checkpoint for a given seed and fold (1..4)."""
    return Path(checkpoints_dir) / f"yolov8n_cls_seed_{seed}_fold_{fold}.pt"


def load_holdout_model_for_image(
    image_id: str,
    predictions_path: Path | str = Path("results/member_4/predictions.csv"),
    checkpoints_dir: Path | str = Path("outputs/yolov8n_cls/checkpoints"),
    seed: int = 42,
) -> Any:
    """Load the fold model that never saw ``image_id`` during training or validation.

    Ensures zero data leakage: each test image is explained strictly by the model trained
    on folds other than its held-out test fold.
    """
    from ultralytics import YOLO

    s, fold = find_image_holdout_fold(image_id, predictions_path=predictions_path, seed=seed)
    checkpoint = resolve_checkpoint_path(s, fold, checkpoints_dir=checkpoints_dir)
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Checkpoint file missing for held-out fold: {checkpoint}. "
            f"Seed {s}, Fold {fold} model must exist to explain image '{image_id}'."
        )
    return YOLO(str(checkpoint))


# --------------------------------------------------------------------------- Sample Selection
def select_gradcam_samples(
    predictions_path: Path | str = Path("results/member_4/predictions.csv"),
    audit_path: Path | str = Path("data/processed/audit.csv"),
    misclassified_path: Path | str = Path("results/member_4/misclassified_images.csv"),
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Deterministically select samples for Grad-CAM inspection.

    Categories:
      (a) correctly classified examples for each class (highest confidence + lowest confidence)
      (b) healthy FN images
      (c) misclassified examples prioritizing -N/-K confusions and -N -> FN

    Returns:
      dict with keys 'correct', 'healthy_fn', 'misclassified'.
    """
    preds = pd.read_csv(predictions_path, dtype={"image_id": str})
    audit = pd.read_csv(audit_path, dtype={"image_id": str})
    preds_seed = preds[preds["seed"] == int(seed)].copy()

    # Merge audit metadata if available
    audit_cols = [c for c in ["image_id", "image_path", "width", "height", "dalle_colour_bar_detected"] if c in audit.columns]
    merged = preds_seed.merge(audit[audit_cols], on="image_id", how="left")

    # (a) Correctly classified examples (highest and lowest confidence per class)
    correct_records = []
    correct_df = merged[merged["true_label"] == merged["predicted_label"]].copy()
    for label in CLASS_NAMES:
        cls_df = correct_df[correct_df["true_label"] == label].sort_values("confidence", ascending=False)
        if len(cls_df) > 0:
            high = cls_df.iloc[0].copy()
            high["selection_reason"] = f"{label} highest confidence"
            correct_records.append(high)
        if len(cls_df) > 1:
            low = cls_df.iloc[-1].copy()
            low["selection_reason"] = f"{label} lower confidence"
            correct_records.append(low)
    df_correct = pd.DataFrame(correct_records).drop_duplicates("image_id").reset_index(drop=True)

    # (b) Healthy FN images (all available in seed, deterministic order)
    df_fn = merged[merged["true_label"] == "FN"].sort_values(["confidence", "image_id"], ascending=[False, True]).reset_index(drop=True)
    df_fn["selection_reason"] = "healthy FN image"

    # (c) Misclassified examples prioritizing -N/-K confusions and -N -> FN
    mis = merged[merged["true_label"] != merged["predicted_label"]].copy()

    def _misclass_priority(row: pd.Series) -> int:
        true_lbl, pred_lbl = str(row["true_label"]), str(row["predicted_label"])
        if (true_lbl == "-N" and pred_lbl == "-K") or (true_lbl == "-K" and pred_lbl == "-N"):
            return 1  # -N/-K confusion
        if true_lbl == "-N" and pred_lbl == "FN":
            return 2  # -N -> FN false positive
        return 3      # other misclassifications

    if not mis.empty:
        mis["priority"] = mis.apply(_misclass_priority, axis=1)
        mis = mis.sort_values(["priority", "confidence"], ascending=[True, False]).reset_index(drop=True)
        mis["selection_reason"] = mis["priority"].map({
            1: "-N/-K confusion",
            2: "-N -> FN confusion",
            3: "other misclassification",
        })
    df_mis = mis

    return {
        "correct": df_correct,
        "healthy_fn": df_fn,
        "misclassified": df_mis,
    }


def format_sample_table(samples: dict[str, pd.DataFrame]) -> str:
    """Format selected samples into an aligned text summary table."""
    lines = ["Category | Image ID | True -> Pred | Conf | Dimensions | Reason",
             "---|---|---|---|---|---"]
    for cat, df in samples.items():
        for _, r in df.iterrows():
            img_id = str(r.get("image_id", ""))[:12]
            true_pred = f"{r.get('true_label', '')} -> {r.get('predicted_label', '')}"
            conf = f"{float(r.get('confidence', 0.0)):.3f}"
            w, h = r.get("width", "?"), r.get("height", "?")
            dim = f"{w}x{h}" if w != "?" else "?"
            reason = str(r.get("selection_reason", cat))
            lines.append(f"{cat} | {img_id} | {true_pred} | {conf} | {dim} | {reason}")
    return "\n".join(lines)


