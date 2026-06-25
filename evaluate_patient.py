"""Patient-level evaluation on the held-out TEST set (the headline metric).

Per-patch accuracy is for debugging; what matters clinically is per-PATIENT
performance — "would this tool have flagged this patient's slide?". This script
rolls the per-patch malignant probabilities up to one decision per patient and
reports recall / precision / F1 under two aggregation strategies.

Ground truth: a patient is malignant iff they have >=1 label-1 (IDC) patch in the
test set; benign iff all their patches are label 0.

Strategy A (max probability): patient score = max patch P(malignant); flag if it
meets the val-tuned threshold from config.py. The most clinically conservative
rule — a single suspicious patch flags the patient.

Strategy B (proportion): patient score = fraction of patches called malignant at
0.50; flag if that fraction exceeds 0.10. A separate comparison, not a swap-in.

Correctness guarantees (see CLAUDE.md / BRAIN.md):
  * TEST split only — val and train are never read for inference.
  * Patient IDs are asserted disjoint across train/val/test before any inference;
    overlap stops the run rather than being papered over.
  * The decision threshold is taken unchanged from config.py — never retuned on
    test data.

Coding discipline follows the p10-coding-rules skill.

Run: ``PYTORCH_ENABLE_MPS_FALLBACK=1 python evaluate_patient.py``
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
# src/ holds the shared dataloader + inference helpers; reuse them, don't fork.
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from app.config import CONFIG  # noqa: E402  (path set up above)
from app.model_loader import load_model  # noqa: E402
from data import SPLITS_PATH, build_dataloader  # type: ignore  # noqa: E402
from train import collect_predictions, get_device  # type: ignore  # noqa: E402

logger = logging.getLogger(__name__)

EVAL_BATCH_SIZE = 512
NUM_WORKERS = 6
PROPORTION_FLAG = 0.10  # Strategy B: flag patient if >10% of patches called malignant
PATCH_DECISION = 0.50   # Strategy B per-patch call threshold
N_EXAMPLES = 5          # regex cross-check sample size


def load_split_assignment(splits_path: str = SPLITS_PATH) -> dict[str, str]:
    """Load the persisted ``patient_id -> split`` map."""
    path = Path(splits_path)
    if not path.is_file():
        raise FileNotFoundError(f"splits file not found: {path}")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)["assignment"]


def assert_no_split_overlap(assignment: dict[str, str]) -> None:
    """Fail loudly if any patient appears in more than one split.

    Patient leakage across splits is the single biggest source of fake accuracy
    on this dataset, so this is a hard precondition, not a warning.
    """
    splits: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    for pid, split in assignment.items():  # bounded by patient count
        if split not in splits:
            raise ValueError(f"patient {pid} has unknown split {split!r}")
        splits[split].add(pid)
    train, val, test = splits["train"], splits["val"], splits["test"]
    overlap = (train & val) | (train & test) | (val & test)
    if overlap:
        raise AssertionError(f"patient overlap across splits: {sorted(overlap)}")
    logger.info("split sizes: train=%d val=%d test=%d (disjoint)",
                len(train), len(val), len(test))


def patient_id_from_path(path: str) -> str:
    """Derive the patient ID from the ``archive/<pid>/<class>/file.png`` layout."""
    return Path(path).parent.parent.name


def group_by_patient(
    samples: list[tuple[str, int]], probabilities: np.ndarray
) -> dict[str, dict]:
    """Group per-patch (label, P(malignant)) by patient from the path structure.

    Returns ``pid -> {"truth": int, "probs": list[float]}`` where ``truth`` is 1
    iff the patient has at least one malignant patch.
    """
    # Explicit raise (not assert): this length check is the alignment invariant
    # that patch i's probability matches patch i's label. A silent mismatch would
    # corrupt every patient metric, so the guard must survive `python -O`.
    if len(samples) != len(probabilities):
        raise ValueError(
            f"samples/probs length mismatch: {len(samples)} != {len(probabilities)}"
        )
    grouped: dict[str, dict] = {}
    for (path, label), prob in zip(samples, probabilities):  # bounded by N patches
        pid = patient_id_from_path(path)
        bucket = grouped.setdefault(pid, {"truth": 0, "probs": []})
        bucket["probs"].append(float(prob))
        if label == 1:
            bucket["truth"] = 1
    return grouped


def _binary_metrics(preds: dict[str, int], truth: dict[str, int]) -> dict:
    """Recall / precision / F1 and class counts for a patient-level prediction map."""
    tp = fp = fn = tn = 0
    for pid, true_label in truth.items():  # bounded by patient count
        pred = preds[pid]
        if pred == 1 and true_label == 1:
            tp += 1
        elif pred == 1 and true_label == 0:
            fp += 1
        elif pred == 0 and true_label == 1:
            fn += 1
        else:
            tn += 1
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"recall": recall, "precision": precision, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def strategy_a_predictions(grouped: dict[str, dict], threshold: float) -> dict[str, int]:
    """Max-probability rule: flag patient if their highest patch prob >= threshold."""
    return {
        pid: int(max(bucket["probs"]) >= threshold)
        for pid, bucket in grouped.items()  # bounded by patient count
    }


def strategy_b_predictions(grouped: dict[str, dict]) -> dict[str, int]:
    """Proportion rule: flag patient if >10% of patches are malignant at 0.50."""
    predictions: dict[str, int] = {}
    for pid, bucket in grouped.items():  # bounded by patient count
        probs = np.asarray(bucket["probs"])
        fraction = float((probs >= PATCH_DECISION).mean())
        predictions[pid] = int(fraction > PROPORTION_FLAG)
    return predictions


def regex_crosscheck(samples: list[tuple[str, int]]) -> list[str]:
    """Confirm the config patient-ID regex agrees with the folder-derived ID.

    Samples ``N_EXAMPLES`` patches evenly spaced across the dataset (so the
    examples span different patients, not one patient's contiguous block) and
    verifies the regex-parsed ID matches the archive folder structure. Raises on
    any disagreement.
    """
    if not samples:
        raise ValueError("no samples to cross-check")
    step = max(1, len(samples) // N_EXAMPLES)
    indices = [min(i * step, len(samples) - 1) for i in range(N_EXAMPLES)]
    examples: list[str] = []
    for idx in indices:  # bounded: N_EXAMPLES
        path = samples[idx][0]
        folder_id = patient_id_from_path(path)
        match = CONFIG.patient_id_pattern.search(Path(path).name)
        if match is None:
            raise ValueError(f"regex {CONFIG.patient_id_pattern.pattern!r} did not "
                             f"match filename {Path(path).name!r}")
        if match.group(1) != folder_id:
            raise ValueError(
                f"regex ID {match.group(1)!r} != folder ID {folder_id!r} for {path}"
            )
        examples.append(match.group(1))
    return examples


def count_zero_patch_patients(
    assignment: dict[str, str], grouped: dict[str, dict]
) -> int:
    """How many TEST patients have no usable patches after the 50x50 filter."""
    test_ids = {pid for pid, split in assignment.items() if split == "test"}
    return sum(1 for pid in test_ids if pid not in grouped)  # bounded by patients


def _report(grouped: dict[str, dict], examples: list[str], zero_patch: int) -> None:
    """Print the required patient-level evaluation report."""
    truth = {pid: bucket["truth"] for pid, bucket in grouped.items()}
    total = len(truth)
    malignant = sum(truth.values())
    benign = total - malignant

    threshold = CONFIG.default_threshold
    metrics_a = _binary_metrics(strategy_a_predictions(grouped, threshold), truth)
    metrics_b = _binary_metrics(strategy_b_predictions(grouped), truth)

    print("\n=== Patient-Level Evaluation (Test Set) ===")
    print(f"Total patients: {total}  |  Malignant: {malignant}  |  Benign: {benign}")
    print(f"\nStrategy A (max probability, threshold={threshold:.4f}):")
    print(f"  Patient Recall: {metrics_a['recall']:.3f}  "
          f"Precision: {metrics_a['precision']:.3f}  F1: {metrics_a['f1']:.3f}  "
          f"[tp={metrics_a['tp']} fp={metrics_a['fp']} fn={metrics_a['fn']} "
          f"tn={metrics_a['tn']}]")
    print(f"\nStrategy B (malignant patch proportion > {PROPORTION_FLAG:.2f}):")
    print(f"  Patient Recall: {metrics_b['recall']:.3f}  "
          f"Precision: {metrics_b['precision']:.3f}  F1: {metrics_b['f1']:.3f}  "
          f"[tp={metrics_b['tp']} fp={metrics_b['fp']} fn={metrics_b['fn']} "
          f"tn={metrics_b['tn']}]")
    print(f"\n{N_EXAMPLES} example patient IDs (confirm regex is working): {examples}")
    print(f"Zero-patch test patients (should be 0): {zero_patch}")

    if benign == 0:
        # Honesty guard: with no benign patients, precision/specificity are
        # vacuously 1.0 and carry no information. Recall is the only meaningful
        # patient-level number here. This is expected for IDC whole-slide data —
        # every patient has at least one cancerous region — and must be reported
        # as such, never as a genuine perfect-precision result.
        print("\n[!] DEGENERATE PATIENT LABELS: every test patient has >=1 "
              "malignant patch, so there are 0 benign patients. Patient-level "
              "PRECISION/SPECIFICITY are uninformative; only RECALL is meaningful.")


def main() -> None:
    """Run the full patient-level evaluation on the held-out test set."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    assignment = load_split_assignment()
    assert_no_split_overlap(assignment)
    print("Patient ID split overlap: NONE CONFIRMED")

    device = get_device()
    model = load_model(CONFIG.active_model, CONFIG.active_spec.checkpoint, device)

    # TEST split only — val/train are never loaded here.
    # shuffle=False is REQUIRED here: collect_predictions returns probabilities in
    # loader iteration order, which equals dataset index order only without
    # shuffling. That ordering is what aligns probabilities[i] with samples[i].
    loader = build_dataloader("test", EVAL_BATCH_SIZE, CONFIG.image_size,
                              NUM_WORKERS, shuffle=False)
    samples: list[tuple[str, int]] = loader.dataset.samples  # index order preserved
    _, probabilities = collect_predictions(model, loader, device)

    grouped = group_by_patient(samples, probabilities)
    examples = regex_crosscheck(samples)
    zero_patch = count_zero_patch_patients(assignment, grouped)
    _report(grouped, examples, zero_patch)


if __name__ == "__main__":
    main()
