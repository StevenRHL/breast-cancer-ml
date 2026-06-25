"""Inference: single-patch (with Grad-CAM) and batch (probabilities only).

The decision threshold is always passed in by the caller — callers pull the
default from ``CONFIG.default_threshold`` (the val-tuned operating point). No
threshold is hardcoded here, so the operating point can never silently diverge
from the one selected in Phase 5.

Coding discipline follows the p10-coding-rules skill.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

from .config import CONFIG
from .gradcam import GradCAM
from .utils import load_image, preprocess

logger = logging.getLogger(__name__)

# Bound batch size so a huge folder can never allocate an unbounded tensor.
MAX_BATCH_SIZE: int = 256


def _device_of(model: nn.Module) -> torch.device:
    """Return the device the model's parameters live on."""
    return next(model.parameters()).device


def _label_for(probability: float, threshold: float) -> str:
    """Map a malignant probability to a label using ``threshold``."""
    is_malignant = probability >= threshold
    return CONFIG.malignant_label if is_malignant else CONFIG.benign_label


@torch.no_grad()
def _malignant_probability(model: nn.Module, tensor: torch.Tensor) -> torch.Tensor:
    """Return per-row P(malignant) for a ``(N,3,H,W)`` batch."""
    return F.softmax(model(tensor), dim=1)[:, 1]


def predict_single(
    image_path: Path,
    model: nn.Module,
    gradcam: GradCAM,
    threshold: float,
) -> dict:
    """Classify one patch and produce a Grad-CAM explanation.

    Args:
        image_path: Path to a single patch image.
        model: A loaded model in eval mode.
        gradcam: A :class:`GradCAM` bound to ``model``.
        threshold: Malignant decision threshold (caller-supplied).

    Returns:
        ``{"label", "probability", "threshold_used", "heatmap", "overlay"}``.

    Raises:
        ValueError: If ``threshold`` is outside (0, 1).
    """
    if not 0.0 < threshold < 1.0:
        raise ValueError(f"threshold must be in (0,1), got {threshold}")

    device = _device_of(model)
    image = load_image(image_path)
    tensor = preprocess(image, (CONFIG.image_size, CONFIG.image_size)).to(device)

    probability = float(_malignant_probability(model, tensor).item())
    heatmap, overlay = gradcam.generate(tensor)
    result = {
        "label": _label_for(probability, threshold),
        "probability": probability,
        "threshold_used": threshold,
        "heatmap": heatmap,
        "overlay": overlay,
    }
    logger.info("single %s -> %s (p=%.4f, t=%.4f)", image_path.name,
                result["label"], probability, threshold)
    return result


def _load_batch_tensor(
    image_paths: list[Path], device: torch.device
) -> torch.Tensor:
    """Preprocess and stack one chunk of images into a single batch tensor."""
    tensors = [
        preprocess(load_image(p), (CONFIG.image_size, CONFIG.image_size))
        for p in image_paths  # bounded: chunk length <= MAX_BATCH_SIZE
    ]
    return torch.cat(tensors, dim=0).to(device)


def predict_batch(
    image_paths: list[Path],
    model: nn.Module,
    threshold: float,
) -> list[dict]:
    """Classify many patches (no Grad-CAM, for throughput).

    Args:
        image_paths: Patch paths to classify.
        model: A loaded model in eval mode.
        threshold: Malignant decision threshold (caller-supplied).

    Returns:
        One dict per patch: ``{"path", "label", "probability"}`` in input order.
        ``path`` is included so :func:`app.aggregator.aggregate_by_patient` can
        group results by patient.

    Raises:
        ValueError: If ``image_paths`` is empty or ``threshold`` is outside (0,1).
    """
    if not image_paths:
        raise ValueError("predict_batch received an empty image_paths list")
    if not 0.0 < threshold < 1.0:
        raise ValueError(f"threshold must be in (0,1), got {threshold}")

    device = _device_of(model)
    results: list[dict] = []
    # Bounded: ceil(len / MAX_BATCH_SIZE) chunks, each at most MAX_BATCH_SIZE.
    for start in range(0, len(image_paths), MAX_BATCH_SIZE):
        chunk = image_paths[start:start + MAX_BATCH_SIZE]
        tensor = _load_batch_tensor(chunk, device)
        probs = _malignant_probability(model, tensor).cpu().numpy()
        for path, prob in zip(chunk, probs):
            probability = float(prob)
            results.append({
                "path": path,
                "label": _label_for(probability, threshold),
                "probability": probability,
            })
    logger.info("batch classified %d patches (t=%.4f)", len(results), threshold)
    return results
