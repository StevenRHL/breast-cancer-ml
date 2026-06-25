"""Hook-based Grad-CAM for explaining a single patch prediction.

Grad-CAM weights a target conv layer's activation maps by the spatially
global-average-pooled gradients of the predicted-class score, producing a coarse
saliency map upsampled to the input resolution and alpha-blended over the patch.

The implementation is non-intrusive: it attaches forward/backward hooks to the
target layer for the duration of one ``generate`` call and removes them in a
``finally`` block, so hooks never accumulate across calls.

Coding discipline follows the p10-coding-rules skill.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from matplotlib import colormaps
from PIL import Image
from torch import nn
from torch.nn import functional as F

from .config import CONFIG
from .utils import tensor_to_pil

logger = logging.getLogger(__name__)

_EPS = 1e-8  # guards divide-by-zero when a CAM is flat


class GradCAM:
    """Grad-CAM saliency generator bound to one model and one target layer.

    At 50×50 input the feature maps are approximately 3×4 pixels. Heatmaps are
    coarse spatial approximations and must not be interpreted as precise lesion
    boundaries. This instance is not thread-safe; create one per inference
    worker.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        """Bind to ``model`` and the ``target_layer`` to hook during ``generate``."""
        assert isinstance(model, nn.Module), "model must be an nn.Module"
        assert isinstance(target_layer, nn.Module), "target_layer must be an nn.Module"
        self._model = model
        self._target_layer = target_layer
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None

    def _save_activations(self, _module, _inp, output: torch.Tensor) -> None:
        """Forward hook: capture the target layer's output activations."""
        self._activations = output.detach()

    def _save_gradients(self, _module, _grad_in, grad_out) -> None:
        """Backward hook: capture gradients flowing into the target layer."""
        self._gradients = grad_out[0].detach()

    def generate(self, input_tensor: torch.Tensor) -> tuple[np.ndarray, Image.Image]:
        """Produce a Grad-CAM heatmap and overlay for ``input_tensor``.

        Args:
            input_tensor: A ``(1, 3, H, W)`` preprocessed (normalised) batch of
                exactly one image.

        Returns:
            ``(heatmap, overlay)`` where ``heatmap`` is a float32 ``(H, W)``
            array normalised to [0, 1] and ``overlay`` is the colourised heatmap
            alpha-blended over the de-normalised input patch.

        Raises:
            ValueError: If ``input_tensor`` is not a single-image batch.
        """
        if input_tensor.dim() != 4 or input_tensor.shape[0] != 1:
            raise ValueError(
                f"expected a (1,3,H,W) single-image batch, got "
                f"{tuple(input_tensor.shape)}"
            )

        fwd = self._target_layer.register_forward_hook(self._save_activations)
        bwd = self._target_layer.register_full_backward_hook(self._save_gradients)
        try:
            cam = self._compute_cam(input_tensor)
        finally:
            # Always remove hooks — never accumulate across calls.
            fwd.remove()
            bwd.remove()
            self._activations = None
            self._gradients = None

        height, width = int(input_tensor.shape[2]), int(input_tensor.shape[3])
        heatmap = self._upsample(cam, height, width)
        overlay = self._overlay(heatmap, input_tensor)
        return heatmap, overlay

    def _compute_cam(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Forward + backward on the predicted class; return the raw ReLU'd CAM."""
        self._model.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(True):
            logits = self._model(input_tensor)
            predicted = int(torch.argmax(logits, dim=1).item())
            score = logits[0, predicted]
            score.backward()

        if self._activations is None or self._gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients")

        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # (1,C,1,1)
        cam = F.relu((weights * self._activations).sum(dim=1, keepdim=True))  # (1,1,h,w)
        logger.debug("grad-cam predicted class=%d cam shape=%s", predicted,
                     tuple(cam.shape))
        return cam

    @staticmethod
    def _upsample(cam: torch.Tensor, height: int, width: int) -> np.ndarray:
        """Bilinearly upsample a CAM to ``(height, width)`` and normalise to [0,1]."""
        up = F.interpolate(cam, size=(height, width), mode="bilinear",
                           align_corners=False)
        flat = up.squeeze().detach().cpu().to(torch.float32)
        lo, hi = float(flat.min()), float(flat.max())
        normalised = (flat - lo) / (hi - lo + _EPS)
        return normalised.numpy().astype(np.float32)

    def _overlay(self, heatmap: np.ndarray, input_tensor: torch.Tensor) -> Image.Image:
        """Colourise ``heatmap`` and alpha-blend it over the input patch."""
        cmap = colormaps[CONFIG.gradcam_colormap]
        coloured = (cmap(heatmap)[..., :3] * 255.0).astype(np.float32)  # (H,W,3)
        base = np.asarray(tensor_to_pil(input_tensor), dtype=np.float32)  # (H,W,3)
        alpha = CONFIG.gradcam_alpha
        blended = alpha * coloured + (1.0 - alpha) * base
        return Image.fromarray(blended.clip(0, 255).astype(np.uint8), mode="RGB")
