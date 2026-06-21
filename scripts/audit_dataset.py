"""Phase 1 data audit. Read-only. Walks top-level numeric patient folders in
archive/, ignoring the nested IDC_regular_ps50_idx5 duplicate. Reports per-class
and per-patient image counts, dimension consistency, and corrupt files.
"""

import os
import random
import sys
from collections import defaultdict

from PIL import Image

ARCHIVE = os.path.join(os.path.dirname(__file__), "..", "archive")
DUP_DIR = "IDC_regular_ps50_idx5"
SAMPLE_SIZE = 4000
random.seed(0)


def patient_dirs(root):
    for name in sorted(os.listdir(root)):
        if name == DUP_DIR:
            continue
        full = os.path.join(root, name)
        if os.path.isdir(full) and name.isdigit():
            yield name, full


def main():
    root = os.path.abspath(ARCHIVE)
    per_patient = {}  # pid -> {0: n, 1: n}
    all_files = []  # (path, class) for sampling
    label_mismatch = 0
    n_patients = 0

    for pid, pdir in patient_dirs(root):
        n_patients += 1
        counts = {0: 0, 1: 0}
        for cls in (0, 1):
            cdir = os.path.join(pdir, str(cls))
            if not os.path.isdir(cdir):
                continue
            for f in os.listdir(cdir):
                if not f.endswith(".png"):
                    continue
                counts[cls] += 1
                if f"class{cls}" not in f:
                    label_mismatch += 1
                all_files.append((os.path.join(cdir, f), cls))
        per_patient[pid] = counts

    total0 = sum(c[0] for c in per_patient.values())
    total1 = sum(c[1] for c in per_patient.values())
    total = total0 + total1

    print(f"Patients: {n_patients}")
    print(f"Total images: {total:,}")
    print(f"  class 0 (non-IDC/benign): {total0:,} ({100*total0/total:.2f}%)")
    print(f"  class 1 (IDC/malignant):  {total1:,} ({100*total1/total:.2f}%)")
    print(f"  imbalance ratio (0:1): {total0/total1:.2f} : 1")
    print(f"  filename/folder label mismatches: {label_mismatch}")

    # per-patient stats
    img_counts = [c[0] + c[1] for c in per_patient.values()]
    frac1 = [(c[1] / (c[0] + c[1])) if (c[0] + c[1]) else 0 for c in per_patient.values()]
    no_pos = sum(1 for c in per_patient.values() if c[1] == 0)
    no_neg = sum(1 for c in per_patient.values() if c[0] == 0)
    print("\nPer-patient image counts:")
    print(f"  min={min(img_counts)}  max={max(img_counts)}  "
          f"mean={sum(img_counts)/len(img_counts):.0f}  "
          f"median={sorted(img_counts)[len(img_counts)//2]}")
    print(f"  patients with 0 malignant patches: {no_pos}")
    print(f"  patients with 0 benign patches:    {no_neg}")
    print(f"  malignant fraction per patient: min={min(frac1):.3f} "
          f"max={max(frac1):.3f} mean={sum(frac1)/len(frac1):.3f}")

    # dimension + corruption audit on a random sample
    sample = random.sample(all_files, min(SAMPLE_SIZE, len(all_files)))
    dims = defaultdict(int)
    modes = defaultdict(int)
    corrupt = []
    for path, _ in sample:
        try:
            with Image.open(path) as im:
                dims[im.size] += 1
                modes[im.mode] += 1
                im.load()
        except Exception as e:  # noqa: BLE001 - audit wants every failure
            corrupt.append((path, repr(e)))
    print(f"\nDimension/corruption audit on random sample of {len(sample):,}:")
    print(f"  sizes: {dict(dims)}")
    print(f"  modes: {dict(modes)}")
    print(f"  corrupt/unreadable: {len(corrupt)}")
    for p, e in corrupt[:10]:
        print(f"    {p} -> {e}")


if __name__ == "__main__":
    sys.exit(main())
