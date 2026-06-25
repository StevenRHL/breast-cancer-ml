"""HuggingFace Spaces entry point.

A thin wrapper around :func:`app.ui.build_app` for the HF Spaces runtime, which
differs from local in three ways:

  * **No Apple Silicon / MPS.** ``app.ui._select_device`` already falls back to
    CPU whenever MPS is unavailable, so on a Spaces (Linux/CPU) host the model
    loads on CPU automatically — no change needed, made explicit here.
  * **Checkpoint location.** Set the ``HF_CHECKPOINT_PATH`` env var (in the Space
    settings) to point at the bundled/downloaded weights; ``app.config`` reads it
    and falls back to the local ``checkpoints/`` path otherwise.
  * **Launch.** Spaces manages host/port/sharing, so we call ``launch()`` with no
    arguments.

Coding discipline follows the p10-coding-rules skill.
"""

from __future__ import annotations

import logging

from app.ui import build_app

logging.basicConfig(level=logging.INFO)

# Built at import time so HuggingFace Spaces can pick up the `demo` object. The
# gradio_client bool-schema shim and (CPU) model load happen inside build_app().
demo = build_app()


if __name__ == "__main__":
    # No args: HF Spaces injects the server host/port and handles sharing.
    demo.launch()
