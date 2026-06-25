"""Pure image I/O and tensor helpers for inference and display.

No global state, no model knowledge — every function is a deterministic
transform. Normalisation matches training (see ``src/data.py``) so checkpoint
weights see inputs distributed exactly as they were trained on.

Coding discipline follows the p10-coding-rules skill.
"""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from .config import CONFIG


def load_image(path: Path) -> Image.Image:
    """Load an image file and return it as RGB.

    Args:
        path: Path to an image file.

    Returns:
        The image converted to RGB mode.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file cannot be opened/decoded as an image.
    """
    if not path.is_file():
        raise FileNotFoundError(f"image not found: {path}")
    try:
        with Image.open(path) as image:
            return image.convert("RGB")
    except Exception as exc:  # Pillow raises various types on bad files
        raise ValueError(f"could not read image {path}: {exc}") from exc


def preprocess(image: Image.Image, size: tuple[int, int]) -> torch.Tensor:
    """Resize, normalise, and add a batch dim, ready for the model.

    Args:
        image: An RGB PIL image.
        size: Target ``(height, width)``.

    Returns:
        A ``(1, 3, H, W)`` float tensor normalised with the ImageNet stats.

    Raises:
        ValueError: If ``image`` is not RGB or ``size`` is non-positive.
    """
    if image.mode != "RGB":
        raise ValueError(f"expected RGB image, got mode={image.mode!r}")
    height, width = size
    if height <= 0 or width <= 0:
        raise ValueError(f"size must be positive, got {size}")
    pipeline = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=CONFIG.imagenet_mean, std=CONFIG.imagenet_std),
    ])
    return pipeline(image).unsqueeze(0)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Invert :func:`preprocess` to a displayable RGB image.

    Accepts a ``(1, 3, H, W)`` or ``(3, H, W)`` normalised tensor.

    Raises:
        ValueError: If the tensor does not have 3 channels.
    """
    work = tensor.detach().cpu()
    if work.dim() == 4:
        work = work[0]
    if work.dim() != 3 or work.shape[0] != 3:
        raise ValueError(f"expected (3,H,W) or (1,3,H,W), got {tuple(tensor.shape)}")
    mean = torch.tensor(CONFIG.imagenet_mean).view(3, 1, 1)
    std = torch.tensor(CONFIG.imagenet_std).view(3, 1, 1)
    denorm = (work * std + mean).clamp(0.0, 1.0)
    array = (denorm.permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")
    return Image.fromarray(array, mode="RGB")
