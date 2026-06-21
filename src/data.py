"""Dataset, dataloaders, and patient-level split for the IDC histopathology
classifier.

Correctness rules baked in here (see CLAUDE.md):
  * Reads ONLY top-level numeric patient folders under archive/. The nested
    archive/IDC_regular_ps50_idx5/ mirror and any non-numeric entry are skipped,
    so a patient can never be globbed twice or leak across splits.
  * Non-50x50 edge patches (~0.8%) are excluded entirely, not padded.
  * The train/val/test split is decided ONCE by ``build_splits`` and persisted to
    data/splits.json. Every other phase reads that file; nothing re-splits.

Run ``python src/data.py`` to (re)build the split and print the summary.

Coding discipline follows the p10-coding-rules skill: guard clauses up front,
bounded loops, short typed functions, explicit error handling.
"""

from __future__ import annotations

import json
import os
import random
from typing import Callable, Iterator, Optional

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# --- Paths and constants ---------------------------------------------------

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ARCHIVE_DIR = os.path.join(PROJECT_ROOT, "archive")
SPLITS_PATH = os.path.join(PROJECT_ROOT, "data", "splits.json")

DUPLICATE_DIRNAME = "IDC_regular_ps50_idx5"  # nested full mirror — must be ignored
PATCH_SIZE = (50, 50)
CLASSES = (0, 1)  # 0 = non-IDC/benign, 1 = IDC/malignant
SPLIT_NAMES = ("train", "val", "test")
DEFAULT_RATIOS = (0.70, 0.15, 0.15)
N_QUINTILES = 5
DEFAULT_SEED = 42
DEFAULT_SEED_SEARCH = 2000  # candidate shuffle seeds swept to balance prevalence

# ImageNet stats — sensible default for the transfer-learning baseline; the
# custom CNN can override via build_transforms(normalize=...).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# --- Filesystem scanning (numeric top-level folders only) ------------------

def iter_patient_ids(archive_dir: str = ARCHIVE_DIR) -> Iterator[str]:
    """Yield top-level numeric patient-folder names under ``archive_dir``.

    Skips the nested duplicate mirror and any non-numeric entry, so the same
    patient is never visited twice.
    """
    assert os.path.isdir(archive_dir), f"archive dir not found: {archive_dir}"
    for name in sorted(os.listdir(archive_dir)):
        if name == DUPLICATE_DIRNAME or not name.isdigit():
            continue
        if os.path.isdir(os.path.join(archive_dir, name)):
            yield name


def _patch_is_full_size(path: str) -> bool:
    """True iff the PNG at ``path`` is exactly 50x50. Raises on unreadable file."""
    try:
        with Image.open(path) as im:
            return im.size == PATCH_SIZE
    except Exception as e:  # corrupt/unreadable is a hard error, not a skip
        raise RuntimeError(f"unreadable patch: {path}") from e


def list_patient_patches(
    patient_id: str, archive_dir: str = ARCHIVE_DIR
) -> list[tuple[str, int]]:
    """Return ``(path, label)`` for every valid 50x50 PNG patch of one patient.

    Non-50x50 edge patches are excluded. The folder label is cross-checked
    against the filename to catch any mislabelled file.
    """
    assert patient_id.isdigit(), f"non-numeric patient id: {patient_id!r}"
    patches: list[tuple[str, int]] = []
    for label in CLASSES:
        class_dir = os.path.join(archive_dir, patient_id, str(label))
        if not os.path.isdir(class_dir):
            continue
        for fname in sorted(os.listdir(class_dir)):  # bounded by folder contents
            if not fname.endswith(".png"):
                continue
            # Explicit raise (not assert): data-integrity guard must survive -O.
            if f"class{label}" not in fname:
                raise ValueError(
                    f"folder/filename label mismatch: {fname} in /{label}/"
                )
            path = os.path.join(class_dir, fname)
            if _patch_is_full_size(path):
                patches.append((path, label))
    return patches


# --- Split construction (quintile-stratified, patient-level) ----------------

def compute_patient_stats(archive_dir: str = ARCHIVE_DIR) -> dict[str, dict]:
    """Per-patient valid-patch counts and malignant fraction.

    Counts reflect ONLY the 50x50 patches that will actually be used.
    """
    stats: dict[str, dict] = {}
    for pid in iter_patient_ids(archive_dir):  # bounded: one pass over patients
        patches = list_patient_patches(pid, archive_dir)
        n1 = sum(label for _, label in patches)
        n0 = len(patches) - n1
        total = n0 + n1
        assert total > 0, f"patient {pid} has no valid patches"
        stats[pid] = {"n0": n0, "n1": n1, "frac": n1 / total}
    return stats


def _quintile_of_rank(rank: int, n: int) -> int:
    """Map a 0-based rank among ``n`` items to a quintile index 0..4."""
    return min(N_QUINTILES - 1, rank * N_QUINTILES // n)


def assign_quintile_splits(
    stats: dict[str, dict],
    ratios: tuple[float, float, float] = DEFAULT_RATIOS,
    seed: int = DEFAULT_SEED,
) -> dict[str, str]:
    """Assign each patient to train/val/test, stratified by malignant-fraction
    quintile.

    Rank-based quintiles (equal patient counts per bin) are robust to duplicate
    fractions. Within each quintile, patients are shuffled with ``seed`` and
    sliced 70/15/15 so every split sees the full range of tumour burden.
    """
    assert abs(sum(ratios) - 1.0) < 1e-9, f"ratios must sum to 1: {ratios}"
    rng = random.Random(seed)

    pids_by_frac = sorted(stats, key=lambda p: stats[p]["frac"])
    n = len(pids_by_frac)
    buckets: list[list[str]] = [[] for _ in range(N_QUINTILES)]
    for rank, pid in enumerate(pids_by_frac):
        buckets[_quintile_of_rank(rank, n)].append(pid)

    assignment: dict[str, str] = {}
    for bucket in buckets:  # bounded: N_QUINTILES buckets
        rng.shuffle(bucket)
        n_train = round(ratios[0] * len(bucket))
        n_val = round(ratios[1] * len(bucket))
        for i, pid in enumerate(bucket):  # bounded by bucket size
            if i < n_train:
                assignment[pid] = "train"
            elif i < n_train + n_val:
                assignment[pid] = "val"
            else:
                assignment[pid] = "test"
    assert len(assignment) == n, "every patient must be assigned exactly once"
    return assignment


def _split_summary(
    assignment: dict[str, str], stats: dict[str, dict]
) -> dict[str, dict]:
    """Aggregate per-split patient count, image count, and malignant fraction."""
    summary = {
        name: {"patients": 0, "images": 0, "malignant": 0}
        for name in SPLIT_NAMES
    }
    for pid, split in assignment.items():
        s = summary[split]
        s["patients"] += 1
        s["images"] += stats[pid]["n0"] + stats[pid]["n1"]
        s["malignant"] += stats[pid]["n1"]
    for s in summary.values():
        s["malignant_frac"] = s["malignant"] / s["images"] if s["images"] else 0.0
    return summary


def _overall_malignant_frac(stats: dict[str, dict]) -> float:
    """Image-weighted malignant prevalence over all valid patches."""
    total = sum(s["n0"] + s["n1"] for s in stats.values())
    malignant = sum(s["n1"] for s in stats.values())
    assert total > 0, "no valid patches found"
    return malignant / total


def _prevalence_deviation(summary: dict[str, dict], overall_frac: float) -> float:
    """Max absolute gap between any split's image-weighted prevalence and the
    overall prevalence. Lower is a more balanced split."""
    return max(
        abs(summary[name]["malignant_frac"] - overall_frac) for name in SPLIT_NAMES
    )


def _search_best_seed(
    stats: dict[str, dict],
    ratios: tuple[float, float, float],
    n_candidates: int,
) -> tuple[int, dict[str, str], dict[str, dict], float]:
    """Sweep ``n_candidates`` shuffle seeds (same quintile method) and pick the
    one minimising image-weighted prevalence deviation. Ties break to the lower
    seed. Returns ``(seed, assignment, summary, deviation)``."""
    assert n_candidates >= 1, f"need >=1 candidate seed, got {n_candidates}"
    overall = _overall_malignant_frac(stats)
    best: Optional[tuple[float, int, dict, dict]] = None
    for seed in range(n_candidates):  # bounded: exactly n_candidates iterations
        assignment = assign_quintile_splits(stats, ratios, seed)
        summary = _split_summary(assignment, stats)
        dev = _prevalence_deviation(summary, overall)
        if best is None or dev < best[0]:
            best = (dev, seed, assignment, summary)
    dev, seed, assignment, summary = best  # type: ignore[misc]
    return seed, assignment, summary, dev


def build_splits(
    archive_dir: str = ARCHIVE_DIR,
    splits_path: str = SPLITS_PATH,
    ratios: tuple[float, float, float] = DEFAULT_RATIOS,
    search_seeds: int = DEFAULT_SEED_SEARCH,
) -> dict:
    """Compute the patient-level split and persist it to ``splits_path``.

    The quintile-stratification method is fixed; only the shuffle seed is
    swept (``search_seeds`` candidates) to minimise image-weighted prevalence
    deviation across train/val/test. Returns the in-memory split document
    (also written to disk as JSON).
    """
    stats = compute_patient_stats(archive_dir)
    overall_frac = _overall_malignant_frac(stats)
    seed, assignment, summary, deviation = _search_best_seed(
        stats, ratios, search_seeds
    )
    document = {
        "config": {
            "seed": seed,
            "ratios": {"train": ratios[0], "val": ratios[1], "test": ratios[2]},
            "stratify": "per-patient malignant-fraction quintiles",
            "n_quintiles": N_QUINTILES,
            "patch_size": list(PATCH_SIZE),
            "excluded": "non-50x50 edge patches; nested IDC_regular_ps50_idx5 mirror",
            "overall_malignant_frac": overall_frac,
            "seed_search": {
                "candidates_searched": search_seeds,
                "metric": "max abs deviation of per-split image-weighted "
                "malignant fraction from overall",
                "achieved_deviation": deviation,
            },
        },
        "summary": summary,
        "assignment": assignment,
        "patient_stats": stats,
    }
    os.makedirs(os.path.dirname(splits_path), exist_ok=True)
    with open(splits_path, "w", encoding="utf-8") as f:
        json.dump(document, f, indent=2, sort_keys=True)
    return document


def load_split_patient_ids(split: str, splits_path: str = SPLITS_PATH) -> list[str]:
    """Read patient IDs assigned to ``split`` from the persisted splits file."""
    assert split in SPLIT_NAMES, f"unknown split {split!r}"
    if not os.path.isfile(splits_path):
        raise FileNotFoundError(
            f"{splits_path} not found — run `python src/data.py` to build it first"
        )
    with open(splits_path, encoding="utf-8") as f:
        document = json.load(f)
    assignment = document["assignment"]
    return sorted(pid for pid, s in assignment.items() if s == split)


# --- Transforms and Dataset -------------------------------------------------

def build_transforms(
    train: bool,
    image_size: int = PATCH_SIZE[0],
    normalize: Optional[tuple] = (IMAGENET_MEAN, IMAGENET_STD),
) -> Callable:
    """Histology-appropriate transforms.

    Train-time augmentation is limited to operations that could plausibly occur
    on a real slide scan: flips (any orientation is valid for a tissue patch),
    small rotations, and mild colour jitter to mimic staining variation. No
    aggressive warping that would distort tissue morphology.
    """
    steps: list = []
    if image_size != PATCH_SIZE[0]:
        steps.append(transforms.Resize((image_size, image_size)))
    if train:
        steps += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05
            ),
        ]
    steps.append(transforms.ToTensor())
    if normalize is not None:
        steps.append(transforms.Normalize(mean=normalize[0], std=normalize[1]))
    return transforms.Compose(steps)


class BreastHistopathologyDataset(Dataset):
    """IDC patch dataset for one split.

    Patient membership comes from data/splits.json (never re-split here). Patch
    paths are globbed directly from the numeric top-level archive folders, with
    non-50x50 patches excluded.
    """

    def __init__(
        self,
        split: str,
        transform: Optional[Callable] = None,
        splits_path: str = SPLITS_PATH,
        archive_dir: str = ARCHIVE_DIR,
    ) -> None:
        assert split in SPLIT_NAMES, f"unknown split {split!r}"
        self.split = split
        self.transform = transform
        patient_ids = load_split_patient_ids(split, splits_path)
        self.samples: list[tuple[str, int]] = []
        for pid in patient_ids:  # bounded by patients in this split
            self.samples.extend(list_patient_patches(pid, archive_dir))
        assert self.samples, f"no samples found for split {split!r}"

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        with Image.open(path) as im:
            # Explicit raise (not assert): defence-in-depth that no non-50x50
            # edge patch ever reaches the model, even if the filter is bypassed.
            if im.size != PATCH_SIZE:
                raise ValueError(
                    f"expected {PATCH_SIZE} patch, got {im.size}: {path}"
                )
            image = im.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def class_counts(self) -> tuple[int, int]:
        """Return ``(n_benign, n_malignant)`` for this split's loaded patches."""
        n1 = sum(label for _, label in self.samples)
        return len(self.samples) - n1, n1


def build_dataloader(
    split: str,
    batch_size: int = 64,
    image_size: int = PATCH_SIZE[0],
    num_workers: int = 4,
    shuffle: Optional[bool] = None,
) -> DataLoader:
    """Construct a DataLoader for ``split`` with the right transforms.

    Train shuffles and augments; val/test do neither.
    """
    is_train = split == "train"
    dataset = BreastHistopathologyDataset(
        split, transform=build_transforms(train=is_train, image_size=image_size)
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train if shuffle is None else shuffle,
        num_workers=num_workers,
        pin_memory=False,  # MPS does not benefit from pinned host memory
        drop_last=is_train,
    )


def _print_summary(document: dict) -> None:
    """Print the per-split table required after building the split."""
    summary = document["summary"]
    print("Patient-level split (quintile-stratified by malignant fraction):\n")
    header = f"{'split':<6} {'patients':>9} {'images':>10} {'malignant_frac':>15}"
    print(header)
    print("-" * len(header))
    for name in SPLIT_NAMES:
        s = summary[name]
        print(
            f"{name:<6} {s['patients']:>9} {s['images']:>10,} "
            f"{s['malignant_frac']:>15.4f}"
        )
    totals_p = sum(summary[n]["patients"] for n in SPLIT_NAMES)
    totals_i = sum(summary[n]["images"] for n in SPLIT_NAMES)
    print("-" * len(header))
    print(f"{'total':<6} {totals_p:>9} {totals_i:>10,}")


def main() -> None:
    document = build_splits()
    _print_summary(document)
    print(f"\nWrote split assignment to {SPLITS_PATH}")


if __name__ == "__main__":
    main()
