"""Clinical disclaimer shown on every screen of the app.

``DISCLAIMER_TEXT`` is a plain string (no Gradio import at module load, so it can
be imported in any context). ``render_disclaimer`` lazily imports Gradio and
returns a ready-to-place ``gr.Markdown`` component. Every tab must call it — this
is non-negotiable per the project brief.

Coding discipline follows the p10-coding-rules skill.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # import only for type checkers, never at runtime import time
    import gradio as gr

DISCLAIMER_TEXT: str = (
    "### ⚠️ Research decision-support tool only — not a diagnostic device\n"
    "\n"
    "- This tool is for **research and decision-support only**. It is **not** a "
    "substitute for review by a qualified pathologist.\n"
    "- Grad-CAM heatmaps are **spatial approximations only**. At 50×50 input the "
    "convolutional feature maps are roughly 3×4 pixels, so heatmap localization "
    "is **coarse** and must **not** be interpreted as precise lesion boundaries.\n"
    "- **All outputs must be reviewed by a qualified clinician** before any "
    "clinical action is taken. All results must be confirmed by a licensed "
    "pathologist.\n"
)


def render_disclaimer() -> "gr.Markdown":
    """Return the persistent disclaimer as a Gradio Markdown component.

    Gradio is imported lazily so importing this module (e.g. for
    ``DISCLAIMER_TEXT``) never requires Gradio to be installed.
    """
    import gradio as gr  # lazy: keep the app's optional UI dep out of imports

    return gr.Markdown(DISCLAIMER_TEXT)
