#!/usr/bin/env bash
#
# Download the Breast Histopathology Images (IDC) dataset from Kaggle and
# extract it into ./archive so the layout matches archive/<patient_id>/{0,1}/*.png.
#
# This dataset is ~3.3 GB zipped / ~4.2 GB extracted (555k files) and is NOT
# committed to git (see .gitignore). Run this once before training/evaluation.
# The interactive app does NOT need it.
#
# Requirements:
#   - Kaggle CLI:  pip install kaggle      (already in requirements.txt is fine too)
#   - Kaggle API token at ~/.kaggle/kaggle.json
#       1. Log in at https://www.kaggle.com  ->  Account  ->  "Create New API Token"
#       2. mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/
#       3. chmod 600 ~/.kaggle/kaggle.json
#
# Usage:
#   bash scripts/download_dataset.sh
#
# Idempotent: if archive/ already looks populated it does nothing.

set -euo pipefail

SLUG="paultimothymooney/breast-histopathology-images"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARCHIVE_DIR="$REPO_ROOT/archive"
ZIP_PATH="$REPO_ROOT/archive.zip"

# --- Already downloaded? -----------------------------------------------------
# Count top-level numeric patient folders; if there are plenty, assume we're done.
if [ -d "$ARCHIVE_DIR" ]; then
  existing=$(find "$ARCHIVE_DIR" -maxdepth 1 -mindepth 1 -type d -name '[0-9]*' 2>/dev/null | wc -l | tr -d ' ')
  if [ "${existing:-0}" -gt 100 ]; then
    echo "archive/ already contains $existing patient folders — nothing to do."
    echo "(Delete archive/ and re-run if you want a fresh download.)"
    exit 0
  fi
fi

# --- Kaggle CLI present? -----------------------------------------------------
if ! command -v kaggle >/dev/null 2>&1; then
  cat >&2 <<'EOF'
ERROR: the 'kaggle' CLI is not installed or not on PATH.

  Install it:        pip install kaggle
  Then set up a token (see the header of this script).

Alternatively, download the zip manually in a browser from
  https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images
save it as ./archive.zip, and re-run this script — it will skip the download
and just extract.
EOF
  [ -f "$ZIP_PATH" ] || exit 1
fi

# --- Download ----------------------------------------------------------------
if [ -f "$ZIP_PATH" ]; then
  echo "Found existing $ZIP_PATH — skipping download, will extract it."
else
  echo "Downloading $SLUG from Kaggle (~3.3 GB)..."
  kaggle datasets download -d "$SLUG" -p "$REPO_ROOT"
  # Kaggle names the file after the dataset slug; normalise to archive.zip.
  downloaded="$REPO_ROOT/breast-histopathology-images.zip"
  if [ -f "$downloaded" ]; then
    mv -f "$downloaded" "$ZIP_PATH"
  fi
fi

# --- Extract -----------------------------------------------------------------
echo "Extracting into $ARCHIVE_DIR ..."
mkdir -p "$ARCHIVE_DIR"
# -o overwrite, -q quiet. The zip's top level is the patient-id folders plus a
# duplicate IDC_regular_ps50_idx5/ tree that the audit script already ignores.
unzip -oq "$ZIP_PATH" -d "$ARCHIVE_DIR"

count=$(find "$ARCHIVE_DIR" -maxdepth 1 -mindepth 1 -type d -name '[0-9]*' | wc -l | tr -d ' ')
echo "Done. archive/ now has $count patient folders."
echo "You can delete archive.zip to reclaim ~3.3 GB:  rm \"$ZIP_PATH\""
