"""Architecture-aware model loading and Grad-CAM target-layer resolution.

This is the *only* module that knows how to turn an architecture name into a
loaded ``nn.Module``. New architectures are added here (and in
``src/models.build_model``) and nowhere else.

Coding discipline follows the p10-coding-rules skill.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch
from torch import nn

from .config import CONFIG

logger = logging.getLogger(__name__)


def _build_bare_model(arch: str) -> nn.Module:
    """Construct an un-trained model for ``arch`` using the shared src factory.

    ``src`` is added to ``sys.path`` on demand so the app reuses the exact same
    architecture definitions the training/eval code used (no duplication).
    Weights are NOT pretrained here — they are overwritten by the checkpoint.
    """
    src_dir = str(CONFIG.src_dir)
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    try:
        from models import build_model  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            f"could not import src/models.py from {src_dir!r}: {exc}"
        ) from exc
    return build_model(arch, pretrained=False)


def load_model(
    model_name: str, checkpoint_path: Path, device: torch.device
) -> nn.Module:
    """Load ``model_name``'s trained weights from ``checkpoint_path`` onto ``device``.

    Args:
        model_name: A registered model name (e.g. "ResNet18", "SmallCNN").
        checkpoint_path: Path to the ``.pt`` checkpoint (must contain
            ``state_dict``).
        device: Target torch device.

    Returns:
        The model in ``eval()`` mode with weights loaded.

    Raises:
        ValueError: If ``model_name`` is not registered (never a silent fallback).
        FileNotFoundError: If the checkpoint file does not exist.
        KeyError: If the checkpoint has no ``state_dict``.
    """
    if model_name not in CONFIG.model_specs:
        raise ValueError(
            f"unknown model {model_name!r}; expected one of "
            f"{sorted(CONFIG.model_specs)}"
        )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    spec = CONFIG.model_specs[model_name]
    model = _build_bare_model(spec.arch).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "state_dict" not in checkpoint:
        raise KeyError(f"checkpoint {checkpoint_path} has no 'state_dict' key")
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    logger.info(
        "loaded %s from %s (epoch=%s) onto %s",
        model_name, checkpoint_path, checkpoint.get("epoch", "?"), device,
    )
    return model


def get_target_layer(model: nn.Module, model_name: str) -> nn.Module:
    """Resolve the Grad-CAM target layer for ``model_name`` from config.

    The dotted path (e.g. "layer4.1", "features.12") is read from the model spec
    and walked against ``model`` — attribute access for names, indexing for
    integers. Fails loudly if the path does not resolve.
    """
    if model_name not in CONFIG.model_specs:
        raise ValueError(f"unknown model {model_name!r}")
    path = CONFIG.model_specs[model_name].target_layer
    module: nn.Module = model
    for token in path.split("."):  # bounded: short, fixed dotted path
        try:
            module = module[int(token)] if token.isdigit() else getattr(module, token)
        except (AttributeError, IndexError, TypeError) as exc:
            raise ValueError(
                f"target_layer path {path!r} for {model_name} failed at "
                f"token {token!r}: {exc}"
            ) from exc
    return module
