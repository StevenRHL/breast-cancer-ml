"""``src`` package marker.

Makes the training / evaluation / inference modules importable as
``src.<module>`` (e.g. ``src.wsi_inference``) from the project root, so the
whole-slide pipeline can be run with ``python -m src.wsi_inference`` and reused
by the Gradio app via ``from src.wsi_inference import run_pipeline``.

Existing flat intra-``src`` imports (``from data import ...`` after inserting the
``src`` directory onto ``sys.path``) are unaffected — that pattern resolves the
modules as top-level names and does not depend on this package marker.
"""
