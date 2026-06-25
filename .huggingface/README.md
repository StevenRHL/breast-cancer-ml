---
title: IDC Breast Cancer Patch Detector
emoji: 🔬
colorFrom: pink
colorTo: purple
sdk: gradio
sdk_version: 4.44.1
app_file: app_hf.py
pinned: false
license: mit
---

# IDC Breast Cancer Patch Detector

A patch-level Invasive Ductal Carcinoma (IDC) detector for 50×50 breast
histopathology patches, with Grad-CAM overlays and per-patient aggregation.
Built as a clinical decision-support **research** tool, fine-tuned from an
ImageNet ResNet18.

> ### ⚠️ Research decision-support tool only — not a diagnostic device
>
> - This tool is for **research and decision-support only**. It is **not** a
>   substitute for review by a qualified pathologist.
> - Grad-CAM heatmaps are **spatial approximations only** — coarse at 50×50
>   input and must **not** be read as precise lesion boundaries.
> - **All outputs must be reviewed by a qualified clinician** before any clinical
>   action. All results must be confirmed by a licensed pathologist.
