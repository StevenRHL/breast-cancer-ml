"""Single source of truth for the interactive app (Phase 6).

Every configurable value lives here: the active architecture, checkpoint paths,
image sizes, the val-tuned decision threshold, label names, the patient-ID regex,
Gradio server settings, the Grad-CAM colormap, and the Grad-CAM target layer per
architecture. No other module hardcodes any of these — switching the deployed
model is a one-line change to ``ACTIVE_MODEL`` (or the ``ACTIVE_MODEL`` env var).

Paths are :class:`pathlib.Path`. Environment-specific values (server host/port,
share flag, active model) read from env vars with sensible defaults so nothing
secret or machine-specific is baked into source.

Coding discipline follows the p10-coding-rules skill.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

# --- Repository layout (no absolute paths; resolved from this file) ---------

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
SRC_DIR: Path = PROJECT_ROOT / "src"
CHECKPOINTS_DIR: Path = PROJECT_ROOT / "checkpoints"
ARCHIVE_DIR: Path = PROJECT_ROOT / "archive"

# --- Class labels (0 = benign, 1 = malignant) -------------------------------

BENIGN_LABEL: str = "Benign"
MALIGNANT_LABEL: str = "Malignant"
LABEL_NAMES: dict[int, str] = {0: BENIGN_LABEL, 1: MALIGNANT_LABEL}

# --- Normalisation (must match training; see src/data.py) -------------------

IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

# --- Patient-ID parsing -----------------------------------------------------
# Patch filenames look like: 10253_idx5_x1001_y1001_class0.png
# The patient ID is the leading run of digits before "_idx".
PATIENT_ID_PATTERN: re.Pattern[str] = re.compile(r"^(\d+)_idx\d+")


@dataclass(frozen=True)
class ModelSpec:
    """Everything needed to load and explain one architecture.

    Attributes:
        arch: Key passed to ``src/models.build_model`` ("resnet18"/"smallcnn").
        checkpoint: Path to the best-checkpoint ``.pt`` file.
        image_size: Square input size the model was trained at.
        tuned_threshold: Malignant decision threshold tuned on the VALIDATION
            split under a precision floor (Phase 5), applied unchanged at
            inference. Never re-tuned on test data.
        target_layer: Dotted module path (resolved against the model) of the
            convolutional layer used as the Grad-CAM target.
    """

    arch: str
    checkpoint: Path
    image_size: int
    tuned_threshold: float
    target_layer: str


# Registry of supported architectures. New architectures are registered here and
# in src/models.build_model; nothing else in the app needs to change.
MODEL_SPECS: dict[str, ModelSpec] = {
    "ResNet18": ModelSpec(
        arch="resnet18",
        checkpoint=CHECKPOINTS_DIR / "resnet18_best.pt",
        image_size=128,
        tuned_threshold=0.3162,  # val-tuned, precision floor 0.70 (Phase 5)
        target_layer="layer4.1",  # == layer4[-1], last BasicBlock
    ),
    "SmallCNN": ModelSpec(
        arch="smallcnn",
        checkpoint=CHECKPOINTS_DIR / "smallcnn_best.pt",
        image_size=50,
        tuned_threshold=0.4192,  # val-tuned, precision floor 0.70 (Phase 5)
        target_layer="features.12",  # last conv layer before global pool
    ),
}

# Phase 5 finding: ImageNet-transfer ResNet18 beat the from-scratch SmallCNN on
# tuned-threshold test recall, PR-AUC, and ROC-AUC. ResNet18 is the active model.
DEFAULT_ACTIVE_MODEL: str = "ResNet18"


def _resolve_active_model() -> str:
    """Read the active model name from env, validating against the registry.

    Fails loudly on an unknown name rather than silently falling back, so a
    typo in ``ACTIVE_MODEL`` can never deploy the wrong architecture.
    """
    name = os.environ.get("ACTIVE_MODEL", DEFAULT_ACTIVE_MODEL)
    if name not in MODEL_SPECS:
        raise ValueError(
            f"ACTIVE_MODEL={name!r} is not a registered model; "
            f"expected one of {sorted(MODEL_SPECS)}"
        )
    return name


def _env_int(name: str, default: int) -> int:
    """Parse a positive int from env, raising on a malformed value."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    if not raw.isdigit():
        raise ValueError(f"env {name}={raw!r} is not a non-negative integer")
    return int(raw)


@dataclass(frozen=True)
class Config:
    """Immutable application configuration, built once from env + registry."""

    project_root: Path
    src_dir: Path
    checkpoints_dir: Path
    archive_dir: Path
    active_model: str
    model_specs: dict[str, ModelSpec]
    label_names: dict[int, str]
    benign_label: str
    malignant_label: str
    imagenet_mean: tuple[float, float, float]
    imagenet_std: tuple[float, float, float]
    patient_id_pattern: re.Pattern[str]
    gradcam_colormap: str
    gradcam_alpha: float
    gradio_server_name: str
    gradio_server_port: int
    gradio_share: bool
    # Derived: the active architecture's loadable spec and operating point.
    active_spec: ModelSpec = field(init=False)
    default_threshold: float = field(init=False)
    image_size: int = field(init=False)

    def __post_init__(self) -> None:
        # Frozen dataclass: set derived fields via object.__setattr__.
        spec = self.model_specs[self.active_model]
        object.__setattr__(self, "active_spec", spec)
        object.__setattr__(self, "default_threshold", spec.tuned_threshold)
        object.__setattr__(self, "image_size", spec.image_size)


def load_config() -> Config:
    """Construct the :class:`Config` singleton from environment + registry."""
    return Config(
        project_root=PROJECT_ROOT,
        src_dir=SRC_DIR,
        checkpoints_dir=CHECKPOINTS_DIR,
        archive_dir=ARCHIVE_DIR,
        active_model=_resolve_active_model(),
        model_specs=MODEL_SPECS,
        label_names=LABEL_NAMES,
        benign_label=BENIGN_LABEL,
        malignant_label=MALIGNANT_LABEL,
        imagenet_mean=IMAGENET_MEAN,
        imagenet_std=IMAGENET_STD,
        patient_id_pattern=PATIENT_ID_PATTERN,
        gradcam_colormap=os.environ.get("GRADCAM_COLORMAP", "jet"),
        gradcam_alpha=0.45,
        gradio_server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
        gradio_server_port=_env_int("GRADIO_SERVER_PORT", 7860),
        gradio_share=os.environ.get("GRADIO_SHARE", "").lower() in {"1", "true", "yes"},
    )


# Module-level singleton other modules import.
CONFIG: Config = load_config()
