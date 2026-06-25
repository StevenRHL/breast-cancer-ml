"""Interactive decision-support app for the breast IDC classifier (Phase 6).

Public API re-exported for convenience and smoke testing. The Gradio UI lives in
``app.ui`` and is intentionally not imported here, so importing this package
never requires Gradio to be installed.

Coding discipline follows the p10-coding-rules skill.
"""

from __future__ import annotations

from .aggregator import aggregate_by_patient
from .disclaimer import DISCLAIMER_TEXT
from .gradcam import GradCAM
from .inference import predict_batch, predict_single
from .model_loader import load_model

__all__ = [
    "load_model",
    "GradCAM",
    "predict_single",
    "predict_batch",
    "aggregate_by_patient",
    "DISCLAIMER_TEXT",
]
