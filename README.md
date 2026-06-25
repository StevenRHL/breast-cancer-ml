# Breast IDC Patch Detector

A patch-level IDC (Invasive Ductal Carcinoma) detection model with Grad-CAM
visualization, built as a clinical decision-support **research** tool.

---

## ⚠️ Clinical Disclaimer

> ### Research decision-support tool only — not a diagnostic device
>
> - This tool is for **research and decision-support only**. It is **not** a
>   substitute for review by a qualified pathologist.
> - Grad-CAM heatmaps are **spatial approximations only**. At 50×50 input the
>   convolutional feature maps are roughly 3×4 pixels, so heatmap localization is
>   **coarse** and must **not** be interpreted as precise lesion boundaries.
> - **All outputs must be reviewed by a qualified clinician** before any clinical
>   action is taken. All results must be confirmed by a licensed pathologist.

---

## Dataset

- **Source:** [Breast Histopathology Images (IDC), Kaggle](https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images)
  — 277,524 patches of 50×50 RGB tissue, cropped from 162 whole-slide images
  scanned at 40×.
- **Labels are patch-level, not patient-level.** Each 50×50 patch is labelled
  `0` (no IDC in this region) or `1` (IDC present). **Every patient in the
  dataset is a breast cancer patient** — the labels describe *regions of tissue*,
  not a *patient diagnosis*. There is no "healthy patient" class. This shapes how
  patient-level performance must be measured (see [Results](#results)).
- **Patient-level split (no patch leakage).** Patients are partitioned into
  train/val/test so that **no patient's patches ever appear in more than one
  split** — the single most important guard against inflated accuracy on this
  dataset. The split is quintile-stratified by each patient's malignant fraction
  so prevalence is matched across splits.

  | Split | Patients | Patches (50×50) | Malignant fraction |
  |-------|---------:|----------------:|-------------------:|
  | train |      194 |         187,798 |             0.2862 |
  | val   |       40 |          40,526 |             0.2861 |
  | test  |       45 |          46,898 |             0.2864 |

  > The `archive/` raw dataset is **not** committed to this repo (~3.3 GB zipped
  > / ~4.2 GB extracted). Fetch it from Kaggle with `bash
  > scripts/download_dataset.sh` (or `make data`) — see
  > [Dataset download](#dataset-download) below.

## Model

- **Architecture:** ResNet18, ImageNet-pretrained, fine-tuned with a fresh
  2-class head. Native 50×50 patches are upsampled to 128×128 for the network.
- **Why ResNet18 over a custom CNN?** We pre-registered a head-to-head against a
  lightweight from-scratch `SmallCNN`. **ImageNet transfer learning won on every
  metric that mattered** — tuned-threshold test recall (0.868 vs 0.831), PR-AUC
  (0.889 vs 0.869), and ROC-AUC (0.952 vs 0.941). This contradicts the common
  claim that tiny from-scratch CNNs beat transfer learning on 50×50 IDC patches;
  on a patient-level split with an honestly tuned threshold, transfer learning
  was better.
- **Checkpoint selection:** the best epoch by **validation PR-AUC**. PR-AUC is
  threshold-independent and robust to the ~2.5:1 class imbalance, so it cannot be
  gamed by a trivially high-recall / low-precision epoch the way raw recall can.
  The decision *threshold* is then tuned separately (on validation) for recall.

## Results

All numbers are on the **held-out TEST set**. Decision thresholds were tuned on
**validation only** and applied once to test — the test set never influenced any
threshold or checkpoint choice.

**Patch-level** (per 50×50 patch, tuned threshold):

| Recall | Precision | F1 | ROC-AUC | PR-AUC | Threshold |
|-------:|----------:|---:|--------:|-------:|----------:|
|  0.868 |     0.787 | 0.825 | 0.952 | 0.889 | 0.3162 |

**Patient-level — per-patient *patch* recall** (the meaningful patient metric for
this dataset). For each cancer patient: *what fraction of their IDC-positive
patches does the model correctly flag?* Reported at the deployed patient-level
threshold (0.26), across all 45 test patients:

| Mean recall | Std | Median | Min | Max |
|------------:|----:|-------:|----:|----:|
|       0.865 | 0.118 | 0.902 | 0.426 | 0.989 |

> **Why not a single patient "accuracy"?** Because there is no negative class:
> **all 45 test patients are IDC-positive**. Binary patient classification
> ("malignant vs. benign patient") is therefore *vacuous* here — it scores 1.0 by
> default and means nothing. Per-patient patch recall is the honest substitute.
> Patient-level **precision** is reported by `evaluate_patient.py` but is harder
> to interpret without true-negative patients; treat per-patient recall as the
> headline.

The single-patch view and the per-patient (batch) view use **separate VAL-tuned
thresholds** (0.3162 and 0.26 respectively), because "is this one patch
malignant?" and "how much of this patient's IDC tissue did we catch?" are
different questions. See `app/config.py`.

## Grad-CAM notes

- Grad-CAM targets `layer4[-1]` of ResNet18. At 128×128 input those feature maps
  are **4×4 pixels**, bilinearly upsampled to 128×128 for display.
- Heatmaps are **coarse approximations**, not precise lesion boundaries.
- **Do not** use the heatmap for lesion measurement, margin assessment, or
  surgical planning. It indicates *roughly where* the model looked, nothing more.

## Running locally

Requires Python 3.11. On Apple Silicon the app uses the MPS backend with a CPU
fallback for unsupported ops.

```bash
# 1. Clone, then create a virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the dataset (only needed for (re)training / evaluation, NOT the UI)
#    Needs the Kaggle CLI + API token; see scripts/download_dataset.sh header.
bash scripts/download_dataset.sh
#    (or download manually from Kaggle — see "Dataset download" below)

# 4. Launch the interactive app
PYTORCH_ENABLE_MPS_FALLBACK=1 python -m app.ui
#    or:  make run-local
```

Expected output ends with:

```
Running on local URL:  http://127.0.0.1:7860
```

Open that URL. **Tab 1** scores a single patch and shows a Grad-CAM overlay;
**Tab 2** scores a folder of patches and aggregates them per patient. The
clinical disclaimer is shown on every tab.

To reproduce the evaluation numbers:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python evaluate_patient.py   # patient-level
PYTHONPATH=src python src/evaluate.py --arch resnet18      # patch-level
```

## Dataset download

The raw dataset (~3.3 GB zipped / ~4.2 GB extracted, 277,524 patches) is **not**
committed to this repo. Only training and evaluation need it — the interactive
app runs from the checkpoints in `checkpoints/`.

**Source:** [Breast Histopathology Images (IDC), Kaggle](https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images)

**Automated (recommended)** — needs the Kaggle CLI and an API token:

```bash
pip install kaggle
# Kaggle -> Account -> "Create New API Token" downloads kaggle.json:
mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
bash scripts/download_dataset.sh     # or: make data
```

**Manual** — download the zip in a browser, save it as `./archive.zip`, then run
`bash scripts/download_dataset.sh` (it skips the download and just extracts).

**Expected layout after extraction** — `class 0` = non-IDC (benign),
`class 1` = IDC (malignant):

```
archive/
└── <patient_id>/
    ├── 0/   # benign:    <id>_idx5_x<X>_y<Y>_class0.png
    └── 1/   # malignant: <id>_idx5_x<X>_y<Y>_class1.png
```

## Project structure

```
archive/              Raw dataset (NOT in git; see "Dataset download")
data/                 Patient-level split + valid-patch manifest (generated)
src/
  data.py             Dataset, dataloaders, patient-level split
  models.py           ResNet18 + SmallCNN definitions
  train.py            Training loop (checkpoint on best val PR-AUC)
  evaluate.py         Patch-level metrics + threshold tuning
evaluate_patient.py   Patient-level (per-patient patch recall) + threshold search
app/
  config.py           Single source of truth (model, thresholds, paths, regex)
  disclaimer.py       Clinical disclaimer text + Gradio renderer
  model_loader.py     Architecture-aware checkpoint loading + Grad-CAM target
  gradcam.py          Hook-based Grad-CAM
  inference.py        Single-patch and batch inference
  aggregator.py       Per-patient roll-up of patch predictions
  utils.py            Image I/O and tensor helpers
  ui.py               Gradio app assembly (local entry point)
app_hf.py             HuggingFace Spaces entry point (CPU-safe wrapper)
.huggingface/         HuggingFace Space card (YAML frontmatter + disclaimer)
checkpoints/          Trained weights (resnet18_best.pt, smallcnn_best.pt)
Makefile              run-local / hf-push helpers
requirements.txt      Pinned dependencies
BRAIN.md / BLUEPRINT.md   Project memory and build plan
```

## Limitations

- **Patch-level model only.** There is no whole-slide inference pipeline; the
  model scores 50×50 patches, not whole-slide images.
- **Single public dataset.** Trained and evaluated only on the Kaggle IDC
  dataset. No external-cohort or different-scanner validation.
- **No true-negative patients.** Every patient in this dataset is IDC-positive,
  so the model's behaviour on genuinely cancer-free patients is **untested** on
  this data. Patient-level specificity cannot be measured here.
- **Coarse Grad-CAM localization** (4×4 feature maps). Not for measurement.
- **Not validated for clinical use.** This is a research prototype, not a
  diagnostic device.

## License

- **Code:** MIT (see [`LICENSE`](LICENSE)).
- **Dataset:** the IDC Breast Histopathology Images dataset has its own terms;
  obtain it from and comply with
  [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images).
