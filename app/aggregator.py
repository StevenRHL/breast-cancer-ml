"""Patient-level aggregation of patch predictions.

Per-patient roll-up is the headline metric of this project (see CLAUDE.md): a
pathologist cares about "what does this tool say about this patient's slide",
not about any individual 50×50 patch. Patient IDs are parsed from patch
filenames with the regex owned by ``config.py``.

Coding discipline follows the p10-coding-rules skill.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from .config import CONFIG

logger = logging.getLogger(__name__)


def _patient_id_of(prediction: dict, pattern: re.Pattern[str]) -> str | None:
    """Extract a patient ID from a prediction's filename, or None if unmatched."""
    path = prediction.get("path")
    if path is None:
        return None
    match = pattern.search(Path(path).name)
    return match.group(1) if match else None


def aggregate_by_patient(
    predictions: list[dict], patient_id_pattern: re.Pattern[str]
) -> dict[str, dict]:
    """Group patch predictions by patient and summarise malignant burden.

    Args:
        predictions: Items as returned by ``predict_batch`` — each must carry a
            ``"path"`` and a ``"label"``.
        patient_id_pattern: Compiled regex whose first group is the patient ID
            (typically ``CONFIG.patient_id_pattern``).

    Returns:
        Mapping ``patient_id -> {"patient_id", "total_patches",
        "malignant_count", "malignant_pct"}``.

    Raises:
        ValueError: If ``predictions`` is empty, or if the pattern matches no
            filename at all (a wrong regex must fail loudly, not silently).
    """
    if not predictions:
        raise ValueError("aggregate_by_patient received an empty predictions list")

    summary: dict[str, dict] = {}
    matched = 0
    for prediction in predictions:  # bounded by len(predictions)
        patient_id = _patient_id_of(prediction, patient_id_pattern)
        if patient_id is None:
            continue
        matched += 1
        bucket = summary.setdefault(
            patient_id,
            {"patient_id": patient_id, "total_patches": 0, "malignant_count": 0,
             "malignant_pct": 0.0},
        )
        bucket["total_patches"] += 1
        if prediction.get("label") == CONFIG.malignant_label:
            bucket["malignant_count"] += 1

    if matched == 0:
        raise ValueError(
            f"patient_id_pattern {patient_id_pattern.pattern!r} matched no "
            f"filename in {len(predictions)} predictions — wrong regex?"
        )

    for bucket in summary.values():  # bounded by number of patients
        total = bucket["total_patches"]
        bucket["malignant_pct"] = 100.0 * bucket["malignant_count"] / total

    logger.info("aggregated %d patches into %d patients (%d unmatched)",
                len(predictions), len(summary), len(predictions) - matched)
    return summary
