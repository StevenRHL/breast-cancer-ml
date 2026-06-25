# Project helpers. Recipes are tab-indented (Make requirement).

.PHONY: run-local hf-push data

# Download + extract the Kaggle IDC dataset into archive/ (needed for training
# and evaluation, not for the UI). Requires the Kaggle CLI and an API token —
# see the header of scripts/download_dataset.sh.
data:
	bash scripts/download_dataset.sh

# Launch the Gradio app locally on Apple Silicon (MPS with CPU fallback).
run-local:
	PYTORCH_ENABLE_MPS_FALLBACK=1 python -m app.ui

# Push this repo to the HuggingFace Space. Requires a git remote named
# 'huggingface' pointing at the Space repo (see deployment instructions in
# the session report / README). Pushes the whole tree to the Space's main branch.
hf-push:
	git subtree push --prefix . huggingface main
