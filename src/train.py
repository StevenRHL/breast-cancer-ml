"""Phase 3/4 training entrypoint for the IDC patch classifier.

Trains one architecture (``resnet18`` or ``smallcnn``) on MPS (CPU fallback),
checkpointing on best validation PR-AUC. Every epoch it logs the full metric
suite (precision, recall, specificity, F1, ROC-AUC, PR-AUC, accuracy) for BOTH
val and test so the two architectures are directly comparable.

PR-AUC (threshold-independent, robust to imbalance) replaced raw recall as the
selection metric: recall alone was gamed by trivial high-recall/low-precision
epochs. The decision threshold is tuned separately in Phase 5, under a precision
floor, so recall is still optimised where it matters — at the operating point.

Honesty note: the held-out TEST set is never used for any decision — checkpoint
selection, early stopping, and LR scheduling key off VALIDATION PR-AUC only. Test
metrics are logged for observation; the winner is chosen on validation.

Class weights are computed from the TRAIN split only.

Coding discipline follows the p10-coding-rules skill.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time

import numpy as np
import torch
from torch import nn
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_class_weight

from data import build_dataloader, PROJECT_ROOT
from models import ARCHITECTURES, build_model

LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
CKPT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
DEFAULT_IMAGE_SIZE = {"resnet18": 128, "smallcnn": 50}
DECISION_THRESHOLD = 0.5  # tuned for recall in Phase 5; fixed here for comparison
# Checkpoint/early-stop/LR-schedule key off this val metric. PR-AUC is
# threshold-independent and robust to imbalance, so it cannot be gamed by a
# trivially high-recall, low-precision operating point (unlike raw recall).
SELECTION_METRIC = "pr_auc"


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and torch (incl. MPS) RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_device() -> torch.device:
    """MPS when available, else CPU. Op-level CPU fallback is enabled via the
    PYTORCH_ENABLE_MPS_FALLBACK env var set by the caller."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_class_weights(loader, device: torch.device) -> torch.Tensor:
    """Balanced class weights from the TRAIN split labels only."""
    labels = np.array([label for _, label in loader.dataset.samples])
    weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=labels)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    """Run one training epoch; return mean loss."""
    model.train()
    running, n = 0.0, 0
    for images, labels in loader:  # bounded by dataset size
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        running += loss.item() * images.size(0)
        n += images.size(0)
    assert n > 0, "empty training loader"
    return running / n


@torch.no_grad()
def collect_predictions(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    """Return (y_true, prob_malignant) over a loader."""
    model.eval()
    ys, ps = [], []
    for images, labels in loader:  # bounded by dataset size
        prob = torch.softmax(model(images.to(device)), dim=1)[:, 1]
        ps.append(prob.cpu().numpy())
        ys.append(labels.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def compute_metrics(
    y_true: np.ndarray, prob: np.ndarray, threshold: float = DECISION_THRESHOLD
) -> dict:
    """Full metric suite for the malignant (positive) class."""
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "accuracy": float((tp + tn) / len(y_true)),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "roc_auc": float(roc_auc_score(y_true, prob)),
        "pr_auc": float(average_precision_score(y_true, prob)),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def evaluate(model, loader, device, threshold: float = DECISION_THRESHOLD) -> dict:
    y_true, prob = collect_predictions(model, loader, device)
    return compute_metrics(y_true, prob, threshold)


def _fmt(m: dict) -> str:
    return (
        f"rec={m['recall']:.4f} prec={m['precision']:.4f} f1={m['f1']:.4f} "
        f"spec={m['specificity']:.4f} roc={m['roc_auc']:.4f} pr={m['pr_auc']:.4f} "
        f"acc={m['accuracy']:.4f}"
    )


def run_training(args: argparse.Namespace) -> dict:
    """Full training run for one architecture. Returns final best-checkpoint
    metrics for val and test."""
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)
    device = get_device()
    image_size = args.image_size or DEFAULT_IMAGE_SIZE[args.arch]
    print(f"[{args.arch}] device={device} image_size={image_size} "
          f"batch={args.batch_size} epochs={args.epochs}", flush=True)

    train_loader = build_dataloader(
        "train", args.batch_size, image_size, args.num_workers)
    val_loader = build_dataloader(
        "val", args.eval_batch_size, image_size, args.num_workers, shuffle=False)
    test_loader = build_dataloader(
        "test", args.eval_batch_size, image_size, args.num_workers, shuffle=False)

    model = build_model(args.arch).to(device)
    weights = train_class_weights(train_loader, device)
    print(f"[{args.arch}] class weights (benign, malignant) = "
          f"{weights.tolist()}", flush=True)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=1)

    log_path = os.path.join(LOG_DIR, f"phase3_{args.arch}.jsonl")
    ckpt_path = os.path.join(CKPT_DIR, f"{args.arch}_best.pt")
    best_score, best = -1.0, {}
    epochs_no_improve = 0
    with open(log_path, "w", encoding="utf-8") as log:
        for epoch in range(args.epochs):  # bounded by args.epochs
            t0 = time.time()
            loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_m = evaluate(model, val_loader, device)
            test_m = evaluate(model, test_loader, device)
            score = val_m[SELECTION_METRIC]  # selection signal: val PR-AUC only
            scheduler.step(score)
            record = {
                "arch": args.arch, "epoch": epoch, "train_loss": loss,
                "lr": optimizer.param_groups[0]["lr"],
                "secs": round(time.time() - t0, 1), "val": val_m, "test": test_m,
            }
            log.write(json.dumps(record) + "\n")
            log.flush()
            print(f"[{args.arch}] e{epoch} loss={loss:.4f} "
                  f"({record['secs']}s)\n   VAL  {_fmt(val_m)}\n   TEST {_fmt(test_m)}",
                  flush=True)

            if score > best_score:
                best_score = score
                best = {"epoch": epoch, "val": val_m, "test": test_m}
                torch.save({"state_dict": model.state_dict(), "arch": args.arch,
                            "image_size": image_size, "epoch": epoch,
                            "val": val_m, "test": test_m}, ckpt_path)
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= args.patience:
                    print(f"[{args.arch}] early stop at epoch {epoch}", flush=True)
                    break

    print(f"[{args.arch}] BEST (val-{SELECTION_METRIC}, epoch {best['epoch']}):\n"
          f"   VAL  {_fmt(best['val'])}\n   TEST {_fmt(best['test'])}", flush=True)
    return best


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train an IDC patch classifier.")
    p.add_argument("--arch", required=True, choices=ARCHITECTURES)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", dest="batch_size", type=int, default=256)
    p.add_argument("--eval-batch-size", dest="eval_batch_size", type=int, default=512)
    p.add_argument("--image-size", dest="image_size", type=int, default=0,
                   help="0 = arch default (resnet18:128, smallcnn:50)")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--num-workers", dest="num_workers", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.image_size == 0:
        args.image_size = 0  # resolved in run_training to arch default
    return args


if __name__ == "__main__":
    # Seed the torch global RNG (and friends) in the entrypoint, under the
    # __main__ guard so DataLoader worker subprocesses don't re-run it.
    _args = parse_args()
    set_seed(_args.seed)
    run_training(_args)
