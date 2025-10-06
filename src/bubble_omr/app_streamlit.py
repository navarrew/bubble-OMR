#!/usr/bin/env python3
import streamlit as st
from pathlib import Path
import tempfile
import shutil

from bubble_omr.tools.bubble_score import load_config, Config

# Import cores (works whether run as module or installed package)
try:
    from .align_core import align_single_pdf
    from .visualize_core import overlay_config
    from .grade_core import grade_pdf
    from .stats_core import compute_stats
except Exception:
    from bubble_omr.align_core import align_single_pdf
    from bubble_omr.visualize_core import overlay_config
    from bubble_omr.grade_core import grade_pdf
    from bubble_omr.stats_core import compute_stats

st.set_page_config(page_title="bubble-OMR", layout="wide")
st.title("bubble-OMR")

# ---------- helpers ----------
def _save_upload_to_tmp(upload) -> str:
    """
    Write a Streamlit UploadedFile to a temp file, preserving its extension.
    Return the absolute path.
    """
    if upload is None:
        return ""
    suffix = Path(upload.name).suffix or ""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(upload.getbuffer())
        return str(Path(tmp.name).resolve())

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

# ---------- UI ----------
tab1, tab2, tab3, tab4 = st.tabs(["Config Visualizer", "Align", "Grade", "Stats"])

# ============ Visualize ============
with tab1:
    st.header("Visualize how your config file overlays on your bubble sheet")
    input_path = st.file_uploader("PDF/image", type=["pdf", "png", "jpg", "jpeg"], key="viz_input")

    # Accept YAML or JSON
    cfg_upload = st.file_uploader("Config (YAML or JSON)", type=["yaml", "yml", "json"], key="viz_cfg")

    out_name = st.text_input("Output image name", "zone_preview.png")
    dpi = st.number_input("Render DPI for PDFs", value=300, min_value=72, max_value=600)
    label_blocks = st.checkbox("Label answer blocks", value=True)

    if st.button("Draw overlay"):
        if not input_path or not cfg_upload:
            st.error("Provide both input and config.")
        else:
            try:
                in_path = Path(_save_upload_to_tmp(input_path))
                cfg_path = Path(_save_upload_to_tmp(cfg_upload))
                out_path = Path.cwd() / out_name

                # overlay_config(input_pdf: str, config_path: str, out_image: str = ..., dpi: int = ..., label_blocks: bool = ...)
                overlay_config(str(in_path), str(cfg_path), str(out_path), dpi=int(dpi), label_blocks=bool(label_blocks))

                st.success("Overlay generated.")
                st.download_button(out_name, out_path.read_bytes(), file_name=out_name)
            except Exception as e:
                st.error(f"Visualization failed: {e}")

# ============ Align ============
with tab2:
    st.header("Align scanned PDF")
    input_pdf = st.file_uploader("Scanned PDF", type=["pdf"], key="align_pdf")
    template = st.file_uploader("Template PDF", type=["pdf"], key="align_template")
    out_name = st.text_input("Output PDF name", "aligned_scans.pdf")
    dpi = st.number_input("DPI", value=300, min_value=72, max_value=600)
    method = st.selectbox("Method", ["auto", "aruco", "feature"], index=0)
    compress_pdf = st.checkbox("Compress output PDF with Ghostscript (if available)", value=False)
    compress_quality = st.selectbox("Compression quality", ["screen", "ebook", "printer", "prepress", "default"], index=1)

    if st.button("Run alignment"):
        if not input_pdf or not template:
            st.error("Please provide both scanned and template PDFs.")
        else:
            work = _ensure_dir(Path.cwd())
            in_path = Path(_save_upload_to_tmp(input_pdf))
            tpl_path = Path(_save_upload_to_tmp(template))
            out_path = work / out_name

            try:
                align_single_pdf(
                    str(in_path),
                    str(tpl_path),
                    str(out_path),
                    dpi=dpi,
                    method=method,
                    compress_pdf=compress_pdf,
                    compress_quality=compress_quality
                )
                st.success("Alignment complete.")
                st.download_button(out_name, out_path.read_bytes(), file_name=out_name)
            except Exception as e:
                st.error(f"Alignment failed: {e}")


# ============ Grade ============
with tab3:
    st.header("Grade aligned scans with a key you provide")
    input_pdf = st.file_uploader("Aligned scans (PDF)", type=["pdf"], key="grade_pdf")

    # Accept YAML or JSON
    cfg_upload = st.file_uploader("Config (YAML or JSON)", type=["yaml", "yml", "json"], key="grade_cfg")

    key_txt = st.file_uploader("Key (txt)", type=["txt"], key="grade_key")
    total_q = st.number_input("Total questions", value=40, min_value=1)
    out_csv = st.text_input("Results CSV", "results.csv")
    annotate_dir = st.text_input("Annotated images dir (optional)", "annotated")
    annotate_all = st.checkbox("Annotate all cells", value=False)
    mark_blanks = st.checkbox("Mark blanks as incorrect", value=True)
    label_density = st.checkbox("Label density", value=False)
    min_fill = st.number_input("min_fill", value=0.40, min_value=0.0, max_value=1.0, step=0.01)
    name_min_fill = st.number_input("name_min_fill", value=0.15, min_value=0.0, max_value=1.0, step=0.01)
    min_score = st.number_input("min_score", value=2.0, step=0.1)
    top2_ratio = st.number_input("top2_ratio", value=0.75, step=0.01)

    if st.button("Grade"):
        if not (input_pdf and cfg_upload and key_txt):
            st.error("Provide aligned PDF, config, and key.")
        else:
            try:
                in_path = Path(_save_upload_to_tmp(input_pdf))
                cfg_path = Path(_save_upload_to_tmp(cfg_upload))  # preserves .yaml/.json
                key_path = Path(_save_upload_to_tmp(key_txt))
                out_path = Path.cwd() / out_csv

                ann_dir = (Path.cwd() / annotate_dir) if annotate_dir else None
                if ann_dir:
                    _ensure_dir(ann_dir)

                grade_pdf(
                    str(in_path),
                    str(cfg_path),
                    str(key_path),
                    int(total_q),
                    str(out_path),
                    str(ann_dir) if ann_dir else None,
                    bool(annotate_all),
                    bool(mark_blanks),
                    bool(label_density),
                    float(min_fill),
                    float(name_min_fill),
                    float(min_score),
                    float(top2_ratio),
                )
                st.success("Grading complete.")
                st.download_button(out_csv, out_path.read_bytes(), file_name=out_csv)
            except Exception as e:
                st.error(f"Grading failed: {e}")

# ============ Stats ============
with tab4:
    st.header("Stats / Item analysis")
    results_csv = st.file_uploader("Results CSV", type=["csv"])
    output_csv = st.text_input("Output CSV", "results_with_item_stats.csv")
    item_report = st.text_input("Item report CSV", "item_analysis.csv")
    exam_stats = st.text_input("Exam stats CSV", "exam_stats.csv")
    plots_dir = st.text_input("Plots dir (optional)", "")
    # default decimals=3 (your compute_stats/run now rounds to 3dp by default)
    if st.button("Compute stats"):
        if not results_csv:
            st.error("Provide results CSV")
        else:
            try:
                in_csv = Path(_save_upload_to_tmp(results_csv))
                out_csv_path = Path.cwd() / output_csv
                item_csv_path = Path.cwd() / item_report
                exam_csv_path = Path.cwd() / exam_stats
                pdir = (Path.cwd() / plots_dir) if plots_dir else None
                if pdir:
                    _ensure_dir(pdir)

                compute_stats(
                    str(in_csv),
                    output_csv=str(out_csv_path),
                    item_report_csv=str(item_csv_path),
                    exam_stats_csv=str(exam_csv_path),
                    plots_dir=(str(pdir) if pdir else None),
                )

                st.success("Stats generated.")
                st.download_button(output_csv, out_csv_path.read_bytes(), file_name=output_csv)
                st.download_button(item_report, item_csv_path.read_bytes(), file_name=item_report)
                st.download_button(exam_stats, exam_csv_path.read_bytes(), file_name=exam_stats)
            except Exception as e:
                st.error(f"Stats failed: {e}")