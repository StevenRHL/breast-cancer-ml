"""Whole-slide / folder inference pipeline (Phase 8).

On this dataset a patient folder *is* a tiled whole-slide image: every patch is
a non-overlapping 50×50 crop whose filename carries the slide coordinate and the
region label (e.g. ``10253_idx5_x1001_y1051_class0.png`` →
``patient=10253, x=1001, y=1051, class=0``; x/y step is 50px). This module walks
such a folder, runs the patch model on every patch, rolls the results up to a
per-patient malignant burden and — when ground-truth labels are present — the
headline **per-patient patch recall** (identical definition to
``evaluate_patient.py``), and emits a structured report (two CSVs + a JSON).

It reuses the existing ``app`` inference stack (``predict_batch``, ``load_model``)
rather than forking it, so the operating point and preprocessing can never
silently diverge from the deployed app.

Design and scope are documented in ``BLUEPRINT.md`` (Phase 8). Raw ``.tif``/
``.svs`` whole-slide files are intentionally out of scope for this iteration
(no such file exists in the dataset to verify against); the documented stride =
50 / zero-overlap tiling can be added later as a front-end that emits the same
patch list this pipeline already consumes.

Coding discipline follows the p10-coding-rules skill.

Run: ``python -m src.wsi_inference --input <path> --output <dir> [--threshold T]``
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from torch import nn

from app.config import CONFIG
from app.inference import predict_batch
from app.model_loader import load_model, select_device  # shared, not forked

logger = logging.getLogger(__name__)

# Bounded enumeration: refuse to scan an unbounded number of patches so a
# mis-pointed --input can never exhaust memory before inference even starts.
# The full archive is ~277k patches, comfortably under this ceiling.
MAX_PATCHES: int = 500_000

# Patch filename grammar: patient id, tile x, tile y, and the 0/1 region class.
# Stricter than CONFIG.patient_id_pattern (which only captures the patient id);
# used to recover slide coordinates for the stitched heatmap.
_PATCH_NAME_PATTERN: re.Pattern[str] = re.compile(
    r"^(\d+)_idx\d+_x(\d+)_y(\d+)_class([01])"
)

# Native crop size of the dataset's patches (used only to lay out the heatmap
# grid; model input resizing is handled by app.utils.preprocess).
PATCH_SIZE_PX: int = 50

_CLASS_DIR_NAMES: tuple[str, str] = ("0", "1")


# --- Patch discovery --------------------------------------------------------


@dataclass(frozen=True)
class PatchRef:
    """One patch to score, with the metadata needed for report + heatmap.

    Attributes:
        path: Absolute path to the patch image.
        patient_id: Patient the patch belongs to.
        x: Tile x-coordinate on the slide (None if unparseable from filename).
        y: Tile y-coordinate on the slide (None if unparseable).
        true_label: Ground-truth region class (0 benign / 1 malignant), or None
            when no ground truth is available (flat folder without a class token).
    """

    path: Path
    patient_id: str
    x: int | None
    y: int | None
    true_label: int | None


def _parse_patch_name(name: str) -> tuple[str | None, int | None, int | None, int | None]:
    """Parse ``(patient_id, x, y, class)`` from a patch filename.

    Falls back to the patient-id-only config regex when the full grammar does
    not match, returning ``None`` for the coordinates/class in that case.
    """
    match = _PATCH_NAME_PATTERN.search(name)
    if match is not None:
        return match.group(1), int(match.group(2)), int(match.group(3)), int(match.group(4))
    pid_match = CONFIG.patient_id_pattern.search(name)
    pid = pid_match.group(1) if pid_match is not None else None
    return pid, None, None, None


def _is_patient_dir(path: Path) -> bool:
    """True if ``path`` is a directory holding a ``0/`` and/or ``1/`` class dir."""
    return path.is_dir() and any(
        (path / name).is_dir() for name in _CLASS_DIR_NAMES  # bounded: 2
    )


def _collect_patient(patient_dir: Path) -> list[PatchRef]:
    """Collect patches for one ``<pid>/<class>/*.png`` patient folder.

    Ground truth comes from the ``0``/``1`` subfolder name (authoritative). The
    filename's patient id, when present, is cross-checked and must agree — a
    mismatch is a corrupt layout and fails loudly rather than mislabelling.
    """
    refs: list[PatchRef] = []
    for label_str in _CLASS_DIR_NAMES:  # bounded: 2
        subdir = patient_dir / label_str
        if not subdir.is_dir():
            continue
        for png in sorted(subdir.glob("*.png")):  # bounded by file count
            file_pid, x, y, _ = _parse_patch_name(png.name)
            if file_pid is not None and file_pid != patient_dir.name:
                raise ValueError(
                    f"filename patient id {file_pid!r} disagrees with folder "
                    f"{patient_dir.name!r} for {png}"
                )
            refs.append(PatchRef(png, patient_dir.name, x, y, int(label_str)))
    return refs


def _collect_flat(input_dir: Path) -> list[PatchRef]:
    """Collect patches from a flat ``<dir>/*.png`` folder (id parsed from name)."""
    refs: list[PatchRef] = []
    unparsed = 0
    for png in sorted(input_dir.glob("*.png")):  # bounded by file count
        pid, x, y, cls = _parse_patch_name(png.name)
        if pid is None:
            unparsed += 1
            continue
        refs.append(PatchRef(png, pid, x, y, cls))
    if unparsed:
        logger.warning("%d file(s) in %s had no parseable patient id and were "
                       "skipped", unparsed, input_dir)
    return refs


def _enforce_ceiling(count: int, input_dir: Path) -> None:
    """Fail loudly if the patch count exceeds the memory-safety ceiling."""
    if count > MAX_PATCHES:
        raise ValueError(
            f"{input_dir} yields > {MAX_PATCHES} patches ({count}); refusing to "
            "scan to protect memory. Point --input at a single patient or raise "
            "MAX_PATCHES deliberately."
        )


def discover_patches(input_dir: Path) -> list[PatchRef]:
    """Enumerate patches under ``input_dir``, auto-detecting the folder layout.

    Supports a single patient folder (``<dir>/<class>/*.png``), a multi-patient
    root (``<dir>/<pid>/<class>/*.png``), or a flat folder of patches
    (``<dir>/*.png``). Bounded by :data:`MAX_PATCHES`.

    Raises:
        NotADirectoryError: If ``input_dir`` is not a directory.
        ValueError: If no patches are found, or the ceiling is exceeded.
    """
    if not input_dir.is_dir():
        raise NotADirectoryError(f"--input is not a directory: {input_dir}")

    if _is_patient_dir(input_dir):
        refs = _collect_patient(input_dir)
    else:
        child_patients = sorted(
            child for child in input_dir.iterdir() if _is_patient_dir(child)
        )
        if child_patients:
            refs = []
            for patient_dir in child_patients:  # bounded by child-dir count
                refs.extend(_collect_patient(patient_dir))
                _enforce_ceiling(len(refs), input_dir)
        else:
            refs = _collect_flat(input_dir)

    if not refs:
        raise ValueError(f"no .png patches found under {input_dir}")
    _enforce_ceiling(len(refs), input_dir)
    logger.info("discovered %d patches across %d patient(s) under %s",
                len(refs), len({r.patient_id for r in refs}), input_dir)
    return refs


# --- Inference + per-patient roll-up ----------------------------------------


def score_patches(
    refs: list[PatchRef], model: nn.Module, threshold: float
) -> np.ndarray:
    """Return P(malignant) per patch, aligned 1:1 with ``refs`` order.

    Delegates to ``app.inference.predict_batch`` (chunked + ``no_grad``), so
    preprocessing and batching match the deployed app exactly.
    """
    if not refs:
        raise ValueError("score_patches received no patches")
    paths = [ref.path for ref in refs]  # bounded by len(refs)
    predictions = predict_batch(paths, model, threshold)
    if len(predictions) != len(refs):
        raise ValueError(
            f"prediction/patch count mismatch: {len(predictions)} != {len(refs)}"
        )
    return np.asarray([p["probability"] for p in predictions], dtype=np.float64)


@dataclass(frozen=True)
class PatientSummary:
    """Per-patient roll-up of patch predictions and (optional) recall."""

    patient_id: str
    total_patches: int
    malignant_count: int          # patches the model called malignant
    malignant_pct: float
    n_malignant_true: int         # ground-truth malignant patches (0 if unknown)
    n_benign_true: int
    patch_recall: float | None    # None when no true-malignant patches
    patch_precision: float | None  # None when model called none malignant


def _summarise_one(
    patient_id: str, refs: list[PatchRef], probs: np.ndarray, threshold: float
) -> PatientSummary:
    """Build the summary for a single patient's patches."""
    predicted_malignant = probs >= threshold
    true_labels = np.asarray([r.true_label for r in refs], dtype=object)
    has_truth = np.asarray([r.true_label is not None for r in refs])
    true_malignant = has_truth & (true_labels == 1)
    true_benign = has_truth & (true_labels == 0)

    total = len(refs)
    mal_count = int(predicted_malignant.sum())
    n_true_mal = int(true_malignant.sum())
    true_pos = int((predicted_malignant & true_malignant).sum())
    recall = (true_pos / n_true_mal) if n_true_mal > 0 else None
    precision = (true_pos / mal_count) if mal_count > 0 and n_true_mal > 0 else None
    return PatientSummary(
        patient_id=patient_id,
        total_patches=total,
        malignant_count=mal_count,
        malignant_pct=100.0 * mal_count / total,
        n_malignant_true=n_true_mal,
        n_benign_true=int(true_benign.sum()),
        patch_recall=recall,
        patch_precision=precision,
    )


def summarise_patients(
    refs: list[PatchRef], probs: np.ndarray, threshold: float
) -> list[PatientSummary]:
    """Group patches by patient and summarise each (sorted by patient id)."""
    if len(refs) != len(probs):
        raise ValueError(f"refs/probs length mismatch: {len(refs)} != {len(probs)}")
    by_patient: dict[str, list[int]] = {}
    for index, ref in enumerate(refs):  # bounded by len(refs)
        by_patient.setdefault(ref.patient_id, []).append(index)
    summaries: list[PatientSummary] = []
    for patient_id in sorted(by_patient):  # bounded by patient count
        idx = by_patient[patient_id]
        summaries.append(_summarise_one(
            patient_id, [refs[i] for i in idx], probs[idx], threshold))
    return summaries


def recall_distribution(summaries: list[PatientSummary]) -> dict[str, float] | None:
    """Mean/std + min/p25/median/p75/max of defined per-patient patch recalls.

    Returns None when no patient has a defined recall (no ground truth). Mirrors
    ``evaluate_patient.distribution`` so numbers are comparable across scripts.
    """
    recalls = [s.patch_recall for s in summaries if s.patch_recall is not None]
    if not recalls:
        return None
    arr = np.asarray(recalls, dtype=np.float64)
    return {
        "mean": float(arr.mean()), "std": float(arr.std()),
        "min": float(arr.min()), "p25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)), "p75": float(np.percentile(arr, 75)),
        "max": float(arr.max()), "n": int(arr.size),
    }


# --- Report writers ---------------------------------------------------------


def _fmt_opt(value: float | None) -> str:
    """Format an optional float for CSV ('' when undefined)."""
    return "" if value is None else f"{value:.6f}"


def write_patch_csv(
    out_path: Path, refs: list[PatchRef], probs: np.ndarray, threshold: float
) -> None:
    """Write one row per patch: id, coords, ground truth, prob, prediction."""
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["patient_id", "x", "y", "true_label",
                         "malignant_prob", "pred_label", "threshold"])
        for ref, prob in zip(refs, probs):  # bounded by len(refs)
            writer.writerow([
                ref.patient_id,
                "" if ref.x is None else ref.x,
                "" if ref.y is None else ref.y,
                "" if ref.true_label is None else ref.true_label,
                f"{float(prob):.6f}",
                int(prob >= threshold),
                f"{threshold:.4f}",
            ])


def write_patient_csv(out_path: Path, summaries: list[PatientSummary]) -> None:
    """Write one row per patient (the per-patient summary)."""
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["patient_id", "total_patches", "malignant_count",
                         "malignant_pct", "n_malignant_true", "n_benign_true",
                         "patch_recall", "patch_precision"])
        for s in summaries:  # bounded by patient count
            writer.writerow([
                s.patient_id, s.total_patches, s.malignant_count,
                f"{s.malignant_pct:.2f}", s.n_malignant_true, s.n_benign_true,
                _fmt_opt(s.patch_recall), _fmt_opt(s.patch_precision),
            ])


def write_summary_json(
    out_path: Path, meta: dict, distribution: dict[str, float] | None
) -> None:
    """Write run metadata + the per-patient recall distribution as JSON."""
    payload = dict(meta)
    payload["ground_truth_available"] = distribution is not None
    payload["per_patient_patch_recall"] = distribution
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


# --- Orchestration ----------------------------------------------------------


def _resolve_checkpoint(model_name: str) -> Path:
    """Checkpoint path for ``model_name`` (honours HF override for the active one)."""
    if model_name == CONFIG.active_model:
        return CONFIG.active_checkpoint  # may be HF_CHECKPOINT_PATH override
    return CONFIG.model_specs[model_name].checkpoint


@dataclass(frozen=True)
class Analysis:
    """Everything one inference pass produces, reused by CLI and Gradio tab.

    Holding the patch refs + probabilities lets callers build a heatmap or write
    the report without re-running the model.
    """

    refs: list[PatchRef]
    probs: np.ndarray
    summaries: list[PatientSummary]
    threshold: float
    model_name: str
    distribution: dict[str, float] | None


def analyse(
    input_dir: Path,
    threshold: float | None = None,
    model_name: str | None = None,
    model: nn.Module | None = None,
) -> Analysis:
    """Discover patches, run inference once, and roll up per patient.

    Args:
        input_dir: Folder of patches (see :func:`discover_patches` for layouts).
        threshold: Malignant decision threshold in (0, 1); defaults to
            ``CONFIG.patient_threshold`` (the frozen per-patient operating point).
        model_name: Registered model name; defaults to ``CONFIG.active_model``.
        model: An already-loaded model (e.g. the running app's). When given it is
            used as-is and ``model_name`` is for reporting only.

    Raises:
        ValueError: If ``threshold`` is outside (0, 1) or ``model_name`` is
            unregistered.
    """
    resolved_threshold = CONFIG.patient_threshold if threshold is None else threshold
    if not 0.0 < resolved_threshold < 1.0:
        raise ValueError(f"threshold must be in (0,1), got {resolved_threshold}")
    resolved_model_name = model_name or CONFIG.active_model
    if resolved_model_name not in CONFIG.model_specs:
        raise ValueError(f"unknown model {resolved_model_name!r}; expected one of "
                         f"{sorted(CONFIG.model_specs)}")

    refs = discover_patches(input_dir)
    if model is None:
        model = load_model(resolved_model_name,
                           _resolve_checkpoint(resolved_model_name), select_device())
    probs = score_patches(refs, model, resolved_threshold)
    summaries = summarise_patients(refs, probs, resolved_threshold)
    return Analysis(
        refs=refs, probs=probs, summaries=summaries,
        threshold=resolved_threshold, model_name=resolved_model_name,
        distribution=recall_distribution(summaries),
    )


def write_report(analysis: Analysis, input_dir: Path, output_dir: Path) -> dict:
    """Write the three report artifacts for ``analysis`` and return its metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    patch_csv = output_dir / "patch_predictions.csv"
    patient_csv = output_dir / "patient_summary.csv"
    summary_json = output_dir / "summary.json"
    write_patch_csv(patch_csv, analysis.refs, analysis.probs, analysis.threshold)
    write_patient_csv(patient_csv, analysis.summaries)

    meta = {
        "model": analysis.model_name,
        "threshold": round(analysis.threshold, 6),
        "input_dir": str(input_dir),
        "total_patches": len(analysis.refs),
        "total_patients": len({r.patient_id for r in analysis.refs}),
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "patch_predictions_csv": str(patch_csv),
        "patient_summary_csv": str(patient_csv),
    }
    write_summary_json(summary_json, meta, analysis.distribution)
    meta["summary_json"] = str(summary_json)
    meta["per_patient_patch_recall"] = analysis.distribution
    logger.info("wrote report for %d patients to %s",
                len(analysis.summaries), output_dir)
    return meta


def run_pipeline(
    input_dir: Path,
    output_dir: Path,
    threshold: float | None = None,
    model_name: str | None = None,
    model: nn.Module | None = None,
) -> dict:
    """Analyse ``input_dir`` and write the three report artifacts (CLI path).

    Returns a metadata dict including artifact paths, counts, and the recall
    distribution (also written to ``summary.json``).
    """
    analysis = analyse(input_dir, threshold, model_name, model)
    return write_report(analysis, input_dir, output_dir)


# --- Heatmap (reused by the Gradio tab) -------------------------------------


def build_patient_heatmap(
    refs: list[PatchRef], probs: np.ndarray, patient_id: str
):
    """Stitch a slide-coordinate malignancy heatmap for one patient.

    Places each patch's P(malignant) at its ``(x, y)`` tile position and colours
    it with the configured Grad-CAM colormap. Returns a PIL image, or None when
    the patient has no coordinate metadata (cannot be laid out spatially).
    """
    from matplotlib import colormaps  # lazy: keep matplotlib off the CLI path
    from PIL import Image

    coords = [(r.x, r.y, float(p)) for r, p in zip(refs, probs)
              if r.patient_id == patient_id and r.x is not None and r.y is not None]
    if not coords:
        return None
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    cols = (max(xs) - min(xs)) // PATCH_SIZE_PX + 1
    rows = (max(ys) - min(ys)) // PATCH_SIZE_PX + 1
    grid = np.full((rows, cols), np.nan, dtype=np.float64)
    for x, y, prob in coords:  # bounded by patch count for this patient
        grid[(y - min(ys)) // PATCH_SIZE_PX, (x - min(xs)) // PATCH_SIZE_PX] = prob

    colormap = colormaps[CONFIG.gradcam_colormap]
    filled = np.nan_to_num(grid, nan=0.0)
    rgba = colormap(filled)
    rgba[np.isnan(grid)] = (1.0, 1.0, 1.0, 1.0)  # white for tissue-free cells
    rgb = (rgba[:, :, :3] * 255.0).round().astype("uint8")
    return Image.fromarray(rgb, mode="RGB")


# --- CLI --------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    """Construct the argparse CLI (kept separate so it is unit-testable)."""
    parser = argparse.ArgumentParser(
        description="Whole-slide / folder IDC inference: per-patient patch "
                    "recall + malignant-burden report (CSV + JSON).")
    parser.add_argument("--input", required=True, type=Path,
                        help="Patch folder: a patient dir, a multi-patient root, "
                             "or a flat folder of patches.")
    parser.add_argument("--output", required=True, type=Path,
                        help="Directory for the report artifacts (created if absent).")
    parser.add_argument("--threshold", type=float, default=None,
                        help=f"Malignant threshold in (0,1). Default: "
                             f"CONFIG.patient_threshold ({CONFIG.patient_threshold}).")
    parser.add_argument("--model", type=str, default=None,
                        choices=sorted(CONFIG.model_specs),
                        help=f"Registered model. Default: {CONFIG.active_model}.")
    return parser


def _print_report(meta: dict) -> None:
    """Print a compact, copy-pasteable run summary to stdout."""
    print("\n=== Whole-Slide / Folder Inference ===")
    print(f"Model={meta['model']}  threshold={meta['threshold']}  "
          f"patients={meta['total_patients']}  patches={meta['total_patches']}")
    dist = meta["per_patient_patch_recall"]
    if dist is None:
        print("Per-patient patch recall: N/A (no ground-truth labels in input).")
    else:
        print(f"Per-patient patch recall: mean={dist['mean']:.3f} "
              f"(std={dist['std']:.3f}) | min={dist['min']:.3f} "
              f"p25={dist['p25']:.3f} median={dist['median']:.3f} "
              f"p75={dist['p75']:.3f} max={dist['max']:.3f}  [n={dist['n']}]")
    print(f"Artifacts: {meta['patch_predictions_csv']}, "
          f"{meta['patient_summary_csv']}, {meta['summary_json']}")


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _build_arg_parser().parse_args(argv)
    meta = run_pipeline(
        input_dir=args.input, output_dir=args.output,
        threshold=args.threshold, model_name=args.model,
    )
    _print_report(meta)


if __name__ == "__main__":
    main()
