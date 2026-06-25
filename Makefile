# Project helpers. Recipes are tab-indented (Make requirement).

.PHONY: run-local hf-push

# Launch the Gradio app locally on Apple Silicon (MPS with CPU fallback).
run-local:
	PYTORCH_ENABLE_MPS_FALLBACK=1 python -m app.ui

# Push this repo to the HuggingFace Space. Requires a git remote named
# 'huggingface' pointing at the Space repo (see deployment instructions in
# the session report / README). Pushes the whole tree to the Space's main branch.
hf-push:
	git subtree push --prefix . huggingface main
