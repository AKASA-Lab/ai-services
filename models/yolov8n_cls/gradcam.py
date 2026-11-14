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

from typing import Any

import numpy as np
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
