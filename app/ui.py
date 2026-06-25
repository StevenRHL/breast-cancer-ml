"""Gradio app assembly — wiring only, no business logic.

Tab 1 scores a single uploaded patch and shows probability, label, and a
Grad-CAM overlay. Tab 2 scores a folder/batch of patches and shows a per-patch
table plus a per-patient aggregate summary. The clinical disclaimer is rendered
at the top of every tab.

All model/inference work is delegated to the other ``app`` modules; this file
only adapts them to Gradio components.

Coding discipline follows the p10-coding-rules skill.
"""

from __future__ import annotations

import logging
import tempfile
from collections.abc import Callable
from pathlib import Path

from src.wsi_inference import Analysis, analyse, build_patient_heatmap, write_report

from .aggregator import aggregate_by_patient
from .config import CONFIG
from .disclaimer import render_disclaimer
from .gradcam import GradCAM
from .inference import predict_batch, predict_single
from .model_loader import get_target_layer, load_model, select_device

logger = logging.getLogger(__name__)

# Image extensions accepted in batch mode.
_IMAGE_SUFFIXES: frozenset[str] = frozenset({".png", ".jpg", ".jpeg"})

# Rendered UI copy (kept as constants so it is testable and easy to audit).
_HEATMAP_RESOLUTION_NOTE: str = (
    "_Heatmap is upsampled from a 4×4 feature map. Localization is approximate._"
)
_FILENAME_NOTE: str = (
    "**Keep original filenames** (e.g. `10253_idx5_x1001_y1001_class0.png`). "
    "Patient IDs are parsed from the filename to build the per-patient summary."
)
_WSI_NOTE: str = (
    "Enter a **folder path** on this machine: a single patient folder "
    "(`archive/10253`), a multi-patient root (`archive/`), or a flat folder of "
    "`.png` patches. The whole-slide pipeline scores every patch, rolls results "
    "up per patient, and writes a downloadable CSV. When ground-truth `class0/"
    "class1` labels are present, per-patient **patch recall** is reported."
)
_WSI_HEATMAP_NOTE: str = (
    "_Heatmap places each 50×50 patch's P(malignant) at its slide (x, y) tile. "
    "It is a coarse 50px-resolution coverage map — not a precise lesion boundary._"
)


def _patch_gradio_client_bool_schema() -> None:
    """Work around a gradio_client 1.3.0 / gradio 4.44 api-info crash.

    ``_json_schema_to_python_type`` recurses into boolean JSON-schema values
    (e.g. ``additionalProperties: true``) and then does ``"const" in schema`` on
    a ``bool``, raising ``TypeError: argument of type 'bool' is not iterable``.
    That 500s the startup healthcheck, which in turn makes ``launch`` believe
    localhost is unreachable. We short-circuit boolean schemas to ``Any``.
    Idempotent: only patches once.
    """
    from gradio_client import utils as gc_utils

    if getattr(gc_utils, "_bool_schema_patched", False):
        return
    original = gc_utils._json_schema_to_python_type

    def safe(schema, defs=None):  # noqa: ANN001 (mirrors patched signature)
        if isinstance(schema, bool):
            return "Any"
        return original(schema, defs)

    gc_utils._json_schema_to_python_type = safe
    gc_utils._bool_schema_patched = True


class AppRuntime:
    """Holds the loaded model and Grad-CAM so they are built once, not per call."""

    def __init__(self) -> None:
        self.device = select_device()
        self.model = load_model(CONFIG.active_model, CONFIG.active_checkpoint,
                                self.device)
        target_layer = get_target_layer(self.model, CONFIG.active_model)
        self.gradcam = GradCAM(self.model, target_layer)
        # Two operating points: single-patch view vs per-patient (batch) view.
        self.patch_threshold = CONFIG.patch_threshold
        self.patient_threshold = CONFIG.patient_threshold
        logger.info("runtime ready: model=%s device=%s patch_t=%.4f patient_t=%.4f",
                    CONFIG.active_model, self.device,
                    self.patch_threshold, self.patient_threshold)

    def classify_single(self, image_path: str | None):
        """Adapter for Tab 1: path -> (label+confidence markdown, overlay image)."""
        if not image_path:
            return "Upload a patch image to classify.", None
        result = predict_single(Path(image_path), self.model, self.gradcam,
                                self.patch_threshold)  # single-patch operating point
        summary = (
            f"**Prediction: {result['label']}**  \n"
            f"P(malignant) = {result['probability']:.3f} "
            f"(threshold {result['threshold_used']:.3f})"
        )
        return summary, result["overlay"]

    def classify_folder(self, files: list[str] | None):
        """Adapter for Tab 2: files -> (patch rows, patient-summary rows)."""
        paths = _image_paths(files)
        if not paths:
            return [], []
        # Per-patient view: use the patient-level operating point.
        predictions = predict_batch(paths, self.model, self.patient_threshold)
        patch_rows = [
            [p["path"].name, p["label"], f"{p['probability']:.3f}"]
            for p in predictions  # bounded by number of uploaded files
        ]
        patients = aggregate_by_patient(predictions, CONFIG.patient_id_pattern)
        patient_rows = [
            [s["patient_id"], s["total_patches"], s["malignant_count"],
             f"{s['malignant_pct']:.1f}%"]
            for s in patients.values()  # bounded by number of patients
        ]
        return patch_rows, patient_rows

    def analyse_folder(
        self, folder_path: str | None, progress: Callable | None = None,
    ) -> tuple[str, list[list], str | None, object | None]:
        """Adapter for the WSI tab: a folder path -> (summary, rows, CSV, heatmap).

        Reuses the already-loaded model (no reload). Errors are returned as a
        user-facing message rather than raised, so a bad path never 500s the UI.
        """
        if not folder_path or not folder_path.strip():
            return "Enter a patient folder path to analyse.", [], None, None
        input_dir = Path(folder_path.strip()).expanduser()
        if not input_dir.is_dir():
            return f"Not a directory: `{input_dir}`", [], None, None
        if progress is not None:
            progress(0.1, desc="Discovering & scoring patches…")
        try:
            result = analyse(input_dir, threshold=self.patient_threshold,
                             model_name=CONFIG.active_model, model=self.model)
        except (ValueError, NotADirectoryError) as exc:
            logger.warning("WSI analyse failed for %s: %s", input_dir, exc)
            return f"Could not analyse folder: {exc}", [], None, None
        if progress is not None:
            progress(0.8, desc="Writing report…")
        out_dir = Path(tempfile.mkdtemp(prefix="wsi_"))
        meta = write_report(result, input_dir, out_dir)
        if progress is not None:
            progress(1.0, desc="Done")
        return (
            _format_wsi_summary(meta),
            _wsi_patient_rows(result),
            meta["patient_summary_csv"],
            _dominant_heatmap(result),
        )


def _image_paths(files: list[str] | None) -> list[Path]:
    """Filter uploaded file paths down to supported image files."""
    if not files:
        return []
    return [
        Path(f) for f in files  # bounded by number of uploaded files
        if Path(f).suffix.lower() in _IMAGE_SUFFIXES
    ]


def _format_wsi_summary(meta: dict) -> str:
    """Render the WSI run metadata (counts + recall distribution) as Markdown."""
    dist = meta["per_patient_patch_recall"]
    header = (f"**{meta['total_patients']} patient(s), {meta['total_patches']} "
              f"patches** · threshold {meta['threshold']} · {meta['model']}")
    if dist is None:
        return (header + "  \n_Per-patient patch recall: N/A — no ground-truth "
                "`class0/class1` labels in this folder._")
    return (header + "  \n**Per-patient patch recall** (ground truth present): "
            f"mean **{dist['mean']:.3f}** (std {dist['std']:.3f}) · "
            f"median {dist['median']:.3f} · min {dist['min']:.3f} · "
            f"max {dist['max']:.3f} · [n={dist['n']}]")


def _wsi_patient_rows(analysis: Analysis) -> list[list]:
    """Per-patient rows for the WSI table (recall blank when undefined)."""
    return [
        [s.patient_id, s.total_patches, s.malignant_count,
         f"{s.malignant_pct:.1f}%",
         "" if s.patch_recall is None else f"{s.patch_recall:.3f}"]
        for s in analysis.summaries  # bounded by number of patients
    ]


def _dominant_heatmap(analysis: Analysis) -> object | None:
    """Stitch the heatmap for the patient with the most patches (or None)."""
    if not analysis.summaries:
        return None
    dominant = max(analysis.summaries, key=lambda s: s.total_patches)
    return build_patient_heatmap(analysis.refs, analysis.probs, dominant.patient_id)


def build_app():
    """Assemble and return the Gradio ``Blocks`` app."""
    import gradio as gr  # lazy: keep Gradio out of non-UI import paths

    _patch_gradio_client_bool_schema()  # gradio 4.44 api-info compatibility
    runtime = AppRuntime()
    title = f"Breast IDC Classifier — {CONFIG.active_model}"
    with gr.Blocks(title=title) as app:
        gr.Markdown(f"# {title}")

        with gr.Tab("Single patch"):
            render_disclaimer()  # non-negotiable: disclaimer on every tab
            with gr.Row():
                single_in = gr.Image(type="filepath", label="Patch (50×50 PNG)")
                with gr.Column():
                    single_out = gr.Markdown()
                    overlay_out = gr.Image(label="Grad-CAM overlay")
                    # Rendered resolution caveat directly under the overlay (not
                    # only in the bottom disclaimer): ResNet18 layer4 maps are 4×4
                    # at 128×128 input, bilinearly upsampled for display.
                    gr.Markdown(_HEATMAP_RESOLUTION_NOTE)
            gr.Button("Classify").click(
                runtime.classify_single, inputs=single_in,
                outputs=[single_out, overlay_out],
            )

        with gr.Tab("Batch / folder"):
            render_disclaimer()  # non-negotiable: disclaimer on every tab
            gr.Markdown(_FILENAME_NOTE)
            # type="filepath": Gradio writes uploads to a temp dir but preserves
            # the original basename, which is what the patient-ID regex parses.
            batch_in = gr.File(file_count="multiple", type="filepath",
                               label="Patch images")
            patch_table = gr.Dataframe(
                headers=["file", "prediction", "P(malignant)"],
                label="Per-patch results",
            )
            patient_table = gr.Dataframe(
                headers=["patient_id", "patches", "malignant", "% suspicious"],
                label="Per-patient aggregate",
            )
            gr.Button("Score folder").click(
                runtime.classify_folder, inputs=batch_in,
                outputs=[patch_table, patient_table],
            )

        with gr.Tab("Whole-slide / folder"):
            render_disclaimer()  # non-negotiable: disclaimer on every tab
            gr.Markdown(_WSI_NOTE)
            wsi_in = gr.Textbox(label="Patient folder path",
                                placeholder="e.g. archive/10253")
            wsi_summary = gr.Markdown()
            wsi_table = gr.Dataframe(
                headers=["patient_id", "patches", "malignant", "% suspicious",
                         "patch recall"],
                label="Per-patient summary",
            )
            with gr.Row():
                wsi_heatmap = gr.Image(label="Slide malignancy heatmap "
                                             "(largest patient)")
                wsi_csv = gr.File(label="Download per-patient CSV")
            gr.Markdown(_WSI_HEATMAP_NOTE)

            def _run_wsi(folder_path, progress=gr.Progress()):  # noqa: B008,ANN001
                # gr.Progress is injected by Gradio at call time; pass it through
                # to the gradio-free runtime handler as a plain callable.
                return runtime.analyse_folder(folder_path, progress)

            gr.Button("Analyse folder").click(
                _run_wsi, inputs=wsi_in,
                outputs=[wsi_summary, wsi_table, wsi_csv, wsi_heatmap],
            )

    return app


def main() -> None:
    """Launch the app with the configured Gradio server settings."""
    logging.basicConfig(level=logging.INFO)
    build_app().launch(
        server_name=CONFIG.gradio_server_name,
        server_port=CONFIG.gradio_server_port,
        share=CONFIG.gradio_share,
        show_api=False,  # we expose no programmatic API; avoids api-info schema walk
    )


if __name__ == "__main__":
    main()
