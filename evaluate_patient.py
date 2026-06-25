# NOTE ON IDC DATASET STRUCTURE
# Every patient in this dataset is a breast cancer patient. Labels 0/1 are
# patch-level region labels (does this 50x50 patch contain IDC?), not patient
# diagnosis labels. Patient-level "recall" in the binary classification sense
# is undefined here (no true negatives). The meaningful patient-level metric
# is per-patient patch recall: for each patient, what fraction of their IDC-
# positive patches does the model correctly identify?
"""Patient-level evaluation on the held-out TEST set (the headline metric).

The clinically meaningful question for this dataset is NOT "is this patient
malignant" (every patient is) but: *for each cancer patient, what fraction of
their IDC-positive tissue regions does the model correctly flag?* — i.e.
**per-patient patch recall**, with per-patient patch precision alongside it.
We report the mean, std, and distribution (min/p25/median/p75/max) across
patients.

This script also runs a VAL-only threshold search optimised for mean per-patient
patch recall under a precision floor, then applies the chosen threshold to TEST
exactly once.

Correctness guarantees (see CLAUDE.md / BRAIN.md):
  * Threshold search reads the VAL split only; TEST is touched once at the end.
  * Patient IDs are asserted disjoint across train/val/test before any inference.
  * Patient grouping comes from the ``archive/<pid>/<class>/`` folder structure,
    never trusted from the filename regex (which is only cross-checked).

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
N_EXAMPLES = 5                 # regex cross-check sample size
PRECISION_FLOOR = 0.70        # patient-level threshold-search precision floor
# Threshold sweep grid for the patient-level search: 0.05..0.95 step 0.01.
THRESHOLD_GRID: tuple[float, ...] = tuple(round(0.05 + 0.01 * i, 2) for i in range(91))
# Materiality cut-off for adopting a separate patient threshold (see Step 1).
THRESHOLD_DELTA_CUTOFF = 0.05


# --- Split + grouping -------------------------------------------------------

def load_split_assignment(splits_path: str = SPLITS_PATH) -> dict[str, str]:
    """Load the persisted ``patient_id -> split`` map."""
    path = Path(splits_path)
    if not path.is_file():
        raise FileNotFoundError(f"splits file not found: {path}")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)["assignment"]


def assert_no_split_overlap(assignment: dict[str, str]) -> None:
    """Fail loudly if any patient appears in more than one split."""
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


def collect_grouped(split: str, model, device) -> dict[str, dict]:
    """Run inference on ``split`` and group (prob, label) per patient.

    Returns ``pid -> {"probs": np.ndarray, "labels": np.ndarray}``. Uses the
    shared dataloader with ``shuffle=False`` so ``collect_predictions`` output
    aligns index-for-index with ``dataset.samples``.
    """
    loader = build_dataloader(split, EVAL_BATCH_SIZE, CONFIG.image_size,
                              NUM_WORKERS, shuffle=False)
    samples: list[tuple[str, int]] = loader.dataset.samples
    _, probabilities = collect_predictions(model, loader, device)
    # Explicit raise (survives -O): this is the prob<->label alignment invariant.
    if len(samples) != len(probabilities):
        raise ValueError(
            f"[{split}] samples/probs length mismatch: "
            f"{len(samples)} != {len(probabilities)}"
        )
    acc: dict[str, dict] = {}
    for (path, label), prob in zip(samples, probabilities):  # bounded by N patches
        pid = patient_id_from_path(path)
        bucket = acc.setdefault(pid, {"probs": [], "labels": []})
        bucket["probs"].append(float(prob))
        bucket["labels"].append(int(label))
    for bucket in acc.values():  # bounded by patient count
        bucket["probs"] = np.asarray(bucket["probs"], dtype=np.float64)
        bucket["labels"] = np.asarray(bucket["labels"], dtype=np.int64)
    return acc


# --- Per-patient patch recall / precision -----------------------------------

def per_patient_recall_precision(
    grouped: dict[str, dict], threshold: float
) -> tuple[list[float], list[float]]:
    """Per-patient patch recall and precision at ``threshold``.

    For each patient: recall = TP / (#malignant patches), precision =
    TP / (#patches called malignant). A patient's recall is undefined (and
    omitted) if they have no malignant patch; precision is undefined (and
    omitted) if the model calls none of their patches malignant.

    Note: because undefined values are omitted, mean recall and mean precision
    can be averaged over different patient subsets at high thresholds (where some
    patients get zero positive calls). At the operating thresholds used here all
    patients have both defined, so the two means cover the same 45 patients.
    """
    recalls: list[float] = []
    precisions: list[float] = []
    for bucket in grouped.values():  # bounded by patient count
        labels = bucket["labels"]
        predicted = bucket["probs"] >= threshold
        true_pos = int(np.sum(predicted & (labels == 1)))
        n_malignant = int(np.sum(labels == 1))
        n_called = int(np.sum(predicted))
        if n_malignant > 0:
            recalls.append(true_pos / n_malignant)
        if n_called > 0:
            precisions.append(true_pos / n_called)
    return recalls, precisions


def distribution(values: list[float]) -> dict[str, float]:
    """Mean/std and the min/p25/median/p75/max distribution of ``values``."""
    if not values:
        raise ValueError("cannot summarise an empty distribution")
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()), "std": float(arr.std()),
        "min": float(arr.min()), "p25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)), "p75": float(np.percentile(arr, 75)),
        "max": float(arr.max()), "n": int(arr.size),
    }


def _mean_patient_rp(grouped: dict[str, dict], threshold: float) -> tuple[float, float]:
    """Mean per-patient patch recall and precision at ``threshold``."""
    recalls, precisions = per_patient_recall_precision(grouped, threshold)
    mean_recall = float(np.mean(recalls)) if recalls else 0.0
    mean_precision = float(np.mean(precisions)) if precisions else 0.0
    return mean_recall, mean_precision


# --- VAL-only threshold search (Step 1) -------------------------------------

def search_patient_threshold(
    val_patch_probs: dict[str, list[float]],
    val_patch_labels: dict[str, list[int]],
    precision_floor: float = PRECISION_FLOOR,
) -> float:
    """Threshold maximising mean per-patient patch recall on val.

    Sweeps :data:`THRESHOLD_GRID`; among thresholds whose mean per-patient patch
    precision is at least ``precision_floor``, returns the one with the highest
    mean per-patient patch recall. If none meet the floor, falls back to the
    highest-precision threshold (and warns). VAL ONLY — never call with test.
    """
    if not val_patch_probs:
        raise ValueError("empty val_patch_probs")
    grouped = {
        pid: {"probs": np.asarray(val_patch_probs[pid], dtype=np.float64),
              "labels": np.asarray(val_patch_labels[pid], dtype=np.int64)}
        for pid in val_patch_probs  # bounded by patient count
    }
    best_t, best_recall = None, -1.0
    fallback_t, fallback_prec = THRESHOLD_GRID[0], -1.0
    for threshold in THRESHOLD_GRID:  # bounded: 91 candidates
        mean_recall, mean_precision = _mean_patient_rp(grouped, threshold)
        if mean_precision >= precision_floor and mean_recall > best_recall:
            best_recall, best_t = mean_recall, threshold
        if mean_precision > fallback_prec:
            fallback_prec, fallback_t = mean_precision, threshold
    if best_t is None:
        logger.warning("no threshold met precision floor %.2f on val; falling back "
                       "to highest-precision t=%.2f", precision_floor, fallback_t)
        return float(fallback_t)
    return float(best_t)


# --- Cross-checks -----------------------------------------------------------

def regex_crosscheck(samples: list[tuple[str, int]]) -> list[str]:
    """Confirm the config patient-ID regex agrees with the folder-derived ID."""
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
    assignment: dict[str, str], grouped: dict[str, dict], split: str
) -> int:
    """How many ``split`` patients have no usable patches after the 50x50 filter."""
    ids = {pid for pid, s in assignment.items() if s == split}
    return sum(1 for pid in ids if pid not in grouped)  # bounded by patients


# --- Reporting --------------------------------------------------------------

def _fmt_dist(label: str, dist: dict[str, float]) -> str:
    """One-line distribution summary."""
    return (f"  {label:<10} mean={dist['mean']:.3f} (std={dist['std']:.3f}) | "
            f"min={dist['min']:.3f} p25={dist['p25']:.3f} median={dist['median']:.3f} "
            f"p75={dist['p75']:.3f} max={dist['max']:.3f}  [n={dist['n']}]")


def report_per_patient(grouped: dict[str, dict], threshold: float, tag: str) -> None:
    """Print the per-patient patch recall/precision distributions for one split."""
    recalls, precisions = per_patient_recall_precision(grouped, threshold)
    print(f"\n-- Per-patient patch recall/precision ({tag}, threshold={threshold:.4f}) --")
    print(_fmt_dist("recall", distribution(recalls)))
    if precisions:
        print(_fmt_dist("precision", distribution(precisions)))
    else:
        print("  precision  undefined (model called no patch malignant for any patient)")


def report_vacuous_binary(grouped: dict[str, dict]) -> None:
    """Print the old binary patient classification, clearly labelled vacuous."""
    truths = [int(np.any(b["labels"] == 1)) for b in grouped.values()]
    total = len(truths)
    malignant = int(sum(truths))
    print("\n-- Binary patient classification: VACUOUS — no benign patients --")
    print(f"  {malignant}/{total} patients are malignant, {total - malignant} benign. "
          "With no true negatives, patient-level recall/precision/F1 are trivially "
          "1.0 and carry NO information. Use per-patient patch recall above instead.")


# --- Main -------------------------------------------------------------------

def _run_threshold_search(val_grouped: dict, test_grouped: dict) -> None:
    """Step 1: VAL-only threshold search, then report val/test operating points."""
    patch_threshold = CONFIG.default_threshold  # existing patch-level threshold
    val_probs = {pid: b["probs"].tolist() for pid, b in val_grouped.items()}
    val_labels = {pid: b["labels"].tolist() for pid, b in val_grouped.items()}
    t_star = search_patient_threshold(val_probs, val_labels, PRECISION_FLOOR)

    val_r_star, val_p_star = _mean_patient_rp(val_grouped, t_star)
    val_r_old, val_p_old = _mean_patient_rp(val_grouped, patch_threshold)
    test_r_star, test_p_star = _mean_patient_rp(test_grouped, t_star)
    delta = abs(t_star - patch_threshold)

    print("\n=== Patient-Level Threshold Search (VAL only) ===")
    print(f"Patient-optimized threshold t* = {t_star:.2f} "
          f"(precision floor {PRECISION_FLOOR:.2f})")
    print(f"VAL  @ t*={t_star:.2f}      : mean recall={val_r_star:.3f}  "
          f"mean precision={val_p_star:.3f}")
    print(f"VAL  @ patch={patch_threshold:.4f} : mean recall={val_r_old:.3f}  "
          f"mean precision={val_p_old:.3f}")
    print(f"TEST @ t*={t_star:.2f}      : mean recall={test_r_star:.3f}  "
          f"mean precision={test_p_star:.3f}  (applied once, no further tuning)")
    if delta > THRESHOLD_DELTA_CUTOFF:
        print(f"Decision: |t* - patch| = {delta:.3f} > {THRESHOLD_DELTA_CUTOFF} -> "
              "adopt a separate PATIENT_THRESHOLD in config.py.")
    else:
        print(f"Decision: |t* - patch| = {delta:.3f} <= {THRESHOLD_DELTA_CUTOFF} -> "
              "no patient-level retuning needed; keep the single threshold.")


def main() -> None:
    """Run the corrected patient-level evaluation and threshold search."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    assignment = load_split_assignment()
    assert_no_split_overlap(assignment)
    print("Patient ID split overlap: NONE CONFIRMED")

    device = get_device()
    model = load_model(CONFIG.active_model, CONFIG.active_checkpoint, device)

    # VAL is used only for the threshold search; TEST is the held-out report.
    val_grouped = collect_grouped("val", model, device)
    test_grouped = collect_grouped("test", model, device)

    patch_threshold = CONFIG.default_threshold
    print("\n=== Patient-Level Evaluation (Test Set) ===")
    print("Headline metric = per-patient patch recall (see NOTE at top of file).")
    # Report at both operating points: the patch threshold (single-patch view) and
    # the deployed patient threshold (per-patient/batch view, what the app uses).
    report_per_patient(test_grouped, patch_threshold, "TEST @ patch threshold")
    report_per_patient(test_grouped, CONFIG.patient_threshold,
                       "TEST @ patient threshold")

    test_loader_samples_pid = regex_crosscheck(
        build_dataloader("test", EVAL_BATCH_SIZE, CONFIG.image_size,
                         NUM_WORKERS, shuffle=False).dataset.samples
    )
    zero_patch = count_zero_patch_patients(assignment, test_grouped, "test")
    print(f"\n{N_EXAMPLES} example patient IDs (confirm regex): {test_loader_samples_pid}")
    print(f"Zero-patch test patients (should be 0): {zero_patch}")

    report_vacuous_binary(test_grouped)
    _run_threshold_search(val_grouped, test_grouped)


if __name__ == "__main__":
    main()
