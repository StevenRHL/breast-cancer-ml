"""Phase 5 evaluation: load a PR-AUC-selected checkpoint, tune the malignant
decision threshold for recall under a precision floor, and report the full
metric suite at both the default (0.5) and tuned thresholds.

The threshold is tuned on the VALIDATION split only, then applied unchanged to
TEST — the held-out set never influences the chosen operating point.

Coding discipline follows the p10-coding-rules skill.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from sklearn.metrics import precision_recall_curve

from data import build_dataloader
from models import ARCHITECTURES, build_model
from train import (
    CKPT_DIR,
    DEFAULT_IMAGE_SIZE,
    collect_predictions,
    compute_metrics,
    get_device,
)

DEFAULT_PRECISION_FLOOR = 0.70


def load_checkpoint(arch: str, device: torch.device) -> tuple[torch.nn.Module, dict]:
    """Rebuild ``arch`` and load its best checkpoint weights."""
    path = os.path.join(CKPT_DIR, f"{arch}_best.pt")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(arch, pretrained=False).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def tune_threshold(
    y_true: np.ndarray, prob: np.ndarray, precision_floor: float
) -> tuple[float, bool]:
    """Pick the threshold maximising recall among those whose precision is at
    least ``precision_floor``. Returns (threshold, floor_met).

    If no threshold reaches the floor, fall back to the highest-precision
    threshold and flag ``floor_met=False``.
    """
    assert 0.0 < precision_floor < 1.0, f"bad floor: {precision_floor}"
    precision, recall, thresholds = precision_recall_curve(y_true, prob)
    # precision/recall have one more entry than thresholds; align to thresholds.
    prec, rec = precision[:-1], recall[:-1]
    eligible = prec >= precision_floor
    if eligible.any():
        idx_among = int(np.argmax(np.where(eligible, rec, -1.0)))
        return float(thresholds[idx_among]), True
    # Floor unreachable — choose the most precise operating point available.
    return float(thresholds[int(np.argmax(prec))]), False


def _print_metrics(tag: str, m: dict) -> None:
    c = m["confusion"]
    print(
        f"  {tag:<11} recall={m['recall']:.4f} precision={m['precision']:.4f} "
        f"f1={m['f1']:.4f} spec={m['specificity']:.4f} acc={m['accuracy']:.4f} "
        f"roc_auc={m['roc_auc']:.4f} pr_auc={m['pr_auc']:.4f}  "
        f"[tn={c['tn']} fp={c['fp']} fn={c['fn']} tp={c['tp']}]"
    )


def evaluate_checkpoint(arch: str, precision_floor: float, num_workers: int) -> dict:
    """Full Phase 5 report for one architecture's best checkpoint."""
    device = get_device()
    model, ckpt = load_checkpoint(arch, device)
    image_size = ckpt.get("image_size", DEFAULT_IMAGE_SIZE[arch])
    print(f"[{arch}] checkpoint epoch={ckpt.get('epoch')} image_size={image_size} "
          f"device={device}")

    val_loader = build_dataloader("val", 512, image_size, num_workers, shuffle=False)
    test_loader = build_dataloader("test", 512, image_size, num_workers, shuffle=False)
    y_val, p_val = collect_predictions(model, val_loader, device)
    y_test, p_test = collect_predictions(model, test_loader, device)

    threshold, floor_met = tune_threshold(y_val, p_val, precision_floor)
    print(f"\n[{arch}] threshold tuned on VAL for max recall with precision "
          f">= {precision_floor:.2f}: t={threshold:.4f}"
          f"{'' if floor_met else '  (FLOOR NOT REACHABLE — fell back)'}")

    print(f"\n[{arch}] PR-AUC-selected checkpoint @ default threshold 0.50:")
    _print_metrics("VAL", compute_metrics(y_val, p_val, 0.5))
    _print_metrics("TEST", compute_metrics(y_test, p_test, 0.5))

    val_t = compute_metrics(y_val, p_val, threshold)
    test_t = compute_metrics(y_test, p_test, threshold)
    print(f"\n[{arch}] @ tuned threshold {threshold:.4f}:")
    _print_metrics("VAL", val_t)
    _print_metrics("TEST", test_t)
    print(f"\n[{arch}] CHOSEN operating point (test): "
          f"precision={test_t['precision']:.4f} recall={test_t['recall']:.4f} "
          f"f1={test_t['f1']:.4f}")
    return {"arch": arch, "threshold": threshold, "floor_met": floor_met,
            "test_tuned": test_t, "val_tuned": val_t}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained checkpoint.")
    p.add_argument("--arch", required=True, choices=ARCHITECTURES)
    p.add_argument("--precision-floor", dest="precision_floor", type=float,
                   default=DEFAULT_PRECISION_FLOOR)
    p.add_argument("--num-workers", dest="num_workers", type=int, default=6)
    return p.parse_args()


if __name__ == "__main__":
    _args = parse_args()
    evaluate_checkpoint(_args.arch, _args.precision_floor, _args.num_workers)
