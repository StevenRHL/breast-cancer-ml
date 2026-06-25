"""Model definitions for the IDC patch classifier.

Two architectures are compared on identical data:
  * ``resnet18`` — ImageNet-pretrained ResNet18, final layer replaced with a
    2-class head; fed upsampled patches (50 -> image_size).
  * ``smallcnn`` — a lightweight CNN trained from scratch on native 50x50.

Both emit 2 logits so a single weighted CrossEntropyLoss and a single
softmax-probability path serve both, keeping the comparison fair.

Coding discipline follows the p10-coding-rules skill.
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models
from torchvision.models import ResNet18_Weights

NUM_CLASSES = 2
ARCHITECTURES = ("resnet18", "smallcnn")


def build_resnet18(pretrained: bool = True) -> nn.Module:
    """ImageNet-pretrained ResNet18 with a fresh 2-class head."""
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model


class SmallCNN(nn.Module):
    """Compact CNN for native 50x50 RGB patches.

    Four conv blocks (3->32->64->128->128) with BatchNorm + max-pool, then a
    global-average-pooled linear head. Small enough to train from scratch fast.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, p_drop: float = 0.3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            *self._block(3, 32),    # 50 -> 25
            *self._block(32, 64),   # 25 -> 12
            *self._block(64, 128),  # 12 -> 6
            *self._block(128, 128),  # 6 -> 3
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p_drop),
            nn.Linear(128, num_classes),
        )

    @staticmethod
    def _block(in_ch: int, out_ch: int) -> list[nn.Module]:
        return [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def build_model(arch: str, pretrained: bool = True) -> nn.Module:
    """Factory: ``arch`` in :data:`ARCHITECTURES`."""
    if arch == "resnet18":
        return build_resnet18(pretrained)
    if arch == "smallcnn":
        return SmallCNN()
    raise ValueError(f"unknown arch {arch!r}; expected one of {ARCHITECTURES}")
