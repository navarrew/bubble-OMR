from __future__ import annotations

import sys
import subprocess
from pathlib import Path
from typing import Optional

import typer
from rich import print as rprint

# Config loader that now supports YAML (.yaml/.yml) and JSON
from .config_io import load_config

# Core modules
from .align_core import align_single_pdf
from .visualize_core import overlay_config
from .grade_core import grade_pdf
from .tools import bubble_stats as stats_mod  # has run(...)

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="bubble-OMR: align, visualize, grade, and analyze bubble-sheet exams.",
)


# ----------------------------- ALIGN ---------------------------------
@app.command()
def align(
    input_pdf: str = typer.Argument(..., help="Raw scans PDF"),
    template: str = typer.Option(..., "--template", "-t", help="Template PDF to align to"),
    out_pdf: str = typer.Option("aligned_scans.pdf", "--out-pdf", "-o", help="Output aligned PDF"),
    dpi: int = typer.Option(300, "--dpi", help="Render DPI for alignment & output"),
    template_page: int = typer.Option(0, "--template-page", help="Template page index to use"),
    method: str = typer.Option("auto", "--method", help="Alignment method: auto|aruco|feature"),
    min_markers: int = typer.Option(4, "--min-markers", help="Min ArUco markers to accept"),
    ransac: float = typer.Option(3.0, "--ransac", help="RANSAC reprojection threshold"),
    orb_nfeatures: int = typer.Option(3000, "--orb-nfeatures", help="ORB features for feature-based align"),
    match_ratio: float = typer.Option(0.75, "--match-ratio", help="Lowe ratio for feature matching"),
    dict_name: str = typer.Option("DICT_4X4_50", "--dict-name", help="ArUco dictionary"),
    first_page: Optional[int] = typer.Option(None, "--first-page", help="First page index (1-based)"),
    last_page: Optional[int] = typer.Option(None, "--last-page", help="Last page index (inclusive, 1-based)"),
    save_debug: Optional[str] = typer.Option(None, "--save-debug", help="Directory to dump debug overlays"),
    compress_pdf: bool = typer.Option(True, "--compress-pdf/--no-compress-pdf", help="Compress output PDF via Ghostscript if available"),
    compress_quality: str = typer.Option(
        "ebook",
        "--compress-quality",
        help="Ghostscript -dPDFSETTINGS: screen|ebook|printer|prepress",
    ),
    compress_inplace: bool = typer.Option(
        True,
        "--compress-inplace/--compress-to-newfile",
        help="If false, writes aligned_scans_compressed.pdf alongside out-pdf",
    ),
    fallback_original: bool = typer.Option(
        True,
        "--compress-fallback-original/--no-compress-fallback-original",
        help="If compression fails, keep original out_pdf",
    ),
):
    """
    Align raw scans to a template PDF. Optionally compress the resulting PDF.
    """
    out = align_single_pdf(
        input_pdf=input_pdf,
        template=template,
        out_pdf=out_pdf,
        dpi=dpi,
        template_page=template_page,
        method=method,
        min_markers=min_markers,
        ransac=ransac,
        orb_nfeatures=orb_nfeatures,
        match_ratio=match_ratio,
        dict_name=dict_name,
        first_page=first_page,
        last_page=last_page,
        save_debug=save_debug,
        # compression options (implemented in align_core)
        compress_pdf=compress_pdf,
        compress_quality=compress_quality,
        compress_inplace=compress_inplace,
        fallback_original=fallback_original,
    )
    rprint(f"[green]Wrote:[/green] {out}")


# --------------------------- VISUALIZE -------------------------------
@app.command()
def visualize(
    input_pdf: str = typer.Argument(..., help="An aligned page PDF or template PDF"),
    config: str = typer.Option(..., "--config", "-c", help="Config file (.yaml/.yml or .json) — YAML recommended"),
    out_image: str = typer.Option("config_overlay.png", "--out-image", "-o", help="Output overlay PNG"),
    dpi: int = typer.Option(300, "--dpi", help="Render DPI"),
):
    """
    Overlay the config bubble zones on top of a PDF page to verify placement.
    """
    try:
        overlay_config(
            config_path=config,
            input_path=input_pdf,
            out_image=out_image,
            dpi=dpi,
        )
    except Exception as e:
        rprint(f"[red]Visualization failed for {config}:[/red] {e}")
        raise typer.Exit(code=2)

    rprint(f"[green]Wrote:[/green] {out_image}")
    
    
#----------------------------- GRADE ---------------------------------
@app.command()
def grade(
    input_pdf: str = typer.Argument(..., help="Aligned scans PDF"),
    config: str = typer.Option(..., "--config", "-c", help="Config file (.yaml/.yml or .json) — YAML recommended"),
    key_txt: Optional[str] = typer.Option(None, "--key-txt", "-k",
        help="Answer key file (A/B/C/... one per line). If provided, only first len(key) questions are graded/output."),
    out_csv: str = typer.Option("results.csv", "--out-csv", "-o", help="Output CSV of per-student results"),
    out_annotated_dir: Optional[str] = typer.Option(None, "--out-annotated-dir", help="Directory to write annotated sheets"),
    # These two are active and supported by grade_pdf:
    annotate_all_cells: bool = typer.Option(False, "--annotate-all-cells",
        help="Draw every bubble in each row (not just the chosen one)."),
    label_density: bool = typer.Option(False, "--label-density",
        help="Overlay % fill text at bubble centers in annotated images."),
    # Thresholds and misc:
    min_fill: float = typer.Option(0.40, "--min-fill", help="Min fill threshold for answers (0–1)"),
    top2_ratio: float = typer.Option(0.75, "--top2-ratio", help="Top1/Top2 ratio threshold for 'multi' detection"),
    dpi: int = typer.Option(300, "--dpi", help="Scan/PDF render DPI"),
):
    """
    Grade aligned scans using axis-based config.
    If a key is provided, ONLY the first len(key) questions are graded and written to CSV.
    """
    # Quick config sanity so we fail early with a useful error
    try:
        _ = load_config(config)
    except Exception as e:
        rprint(f"[red]Failed to load config {config}:[/red] {e}")
        raise typer.Exit(code=2)

    try:
        grade_pdf(
            input_path=input_pdf,
            config_path=config,
            out_csv=out_csv,
            key_txt=key_txt,
            out_annotated_dir=out_annotated_dir,
            dpi=dpi,
            min_fill=min_fill,
            top2_ratio=top2_ratio,
            annotate_all_cells=annotate_all_cells,  # <-- now wired through
            label_density=label_density,            # <-- now wired through
        )
    except Exception as e:
        rprint(f"[red]Grading failed:[/red] {e}")
        raise typer.Exit(code=2)

    rprint(f"[green]Wrote results:[/green] {out_csv}")

# ------------------------------ STATS --------------------------------
@app.command()
def stats(
    input_csv: str = typer.Argument(..., help="Results CSV (from 'grade')"),
    output_csv: str = typer.Option("results_with_stats.csv", "--output-csv", "-o", help="Augmented CSV with summary rows"),
    item_pattern: str = typer.Option(r"^Q\d+$", "--item-pattern", help="Regex for item columns (default: ^Q\\d+$)"),
    percent: bool = typer.Option(True, "--percent/--proportion", help="Report difficulty as percent (0-100) or proportion (0-1)"),
    label_col: Optional[str] = typer.Option("name", "--label-col", help="Column containing student label (name/id)"),
    exam_stats_csv: Optional[str] = typer.Option(None, "--exam-stats-csv", help="Optional CSV with KR-20/KR-21, mean, SD"),
    plots_dir: Optional[str] = typer.Option(None, "--plots-dir", help="Optional directory for IRT-ish item plots"),
    key_row_index: Optional[int] = typer.Option(None, "--key-row-index", help="Row index of answer key (0-based). Auto-detect if omitted."),
    answers_mode: str = typer.Option("letters", "--answers-mode", help="letters|index depending on how answers are stored"),
    item_report_csv: Optional[str] = typer.Option(None, "--item-report-csv", help="Optional per-item distractor report CSV"),
    key_label: str = typer.Option("KEY", "--key-label", help="Label string for the key row used in auto-detection"),
    decimals: int = typer.Option(3, "--decimals", help="Number of decimals for output rounding (default: 3)"),
):
    """
    Compute item difficulty, point-biserial, and exam reliability (KR-20/KR-21).
    """
    try:
        # Newer bubble_stats.run signatures accept 'decimals'; older ones don't.
        stats_mod.run(
            input_csv=input_csv,
            output_csv=output_csv,
            item_pattern=item_pattern,
            percent=percent,
            label_col=label_col,
            exam_stats_csv=exam_stats_csv,
            plots_dir=plots_dir,
            key_row_index=key_row_index,
            answers_mode=answers_mode,
            item_report_csv=item_report_csv,
            key_label=key_label,
            decimals=decimals,  # default 3
        )
    except TypeError:
        # Backward-compat: call without 'decimals'
        stats_mod.run(
            input_csv=input_csv,
            output_csv=output_csv,
            item_pattern=item_pattern,
            percent=percent,
            label_col=label_col,
            exam_stats_csv=exam_stats_csv,
            plots_dir=plots_dir,
            key_row_index=key_row_index,
            answers_mode=answers_mode,
            item_report_csv=item_report_csv,
            key_label=key_label,
        )
        rprint("[yellow]Note:[/yellow] your bubble_stats.run() doesn’t support a 'decimals' parameter; "
               "update it to get consistent 3-decimal rounding in all outputs.")
    rprint(f"[green]Wrote stats:[/green] {output_csv}")
    if exam_stats_csv:
        rprint(f"[green]Exam summary:[/green] {exam_stats_csv}")
    if item_report_csv:
        rprint(f"[green]Item report:[/green] {item_report_csv}")


# --------------------------- COMPRESS-PDF ----------------------------
@app.command("compress-pdf")
def compress_pdf_cmd(
    input_pdf: str = typer.Argument(..., help="Input PDF to compress"),
    output_pdf: Optional[str] = typer.Option(None, "--output-pdf", "-o", help="Output path; if omitted, overwrite input"),
    quality: str = typer.Option("ebook", "--quality", help="Ghostscript -dPDFSETTINGS: screen|ebook|printer|prepress"),
    quiet: bool = typer.Option(True, "--quiet/--no-quiet", help="Suppress Ghostscript output"),
):
    """
    Compress a PDF using Ghostscript if installed (macOS: `brew install ghostscript`).
    """
    gs = "gs"
    args = [
        gs, "-sDEVICE=pdfwrite",
        "-dCompatibilityLevel=1.4",
        f"-dPDFSETTINGS=/{quality}",
        "-dNOPAUSE",
        "-dBATCH",
    ]
    if quiet:
        args.append("-dQUIET")

    in_path = Path(input_pdf).expanduser().resolve()
    if output_pdf:
        out_path = Path(output_pdf).expanduser().resolve()
    else:
        out_path = in_path  # overwrite

    if out_path == in_path:
        tmp_out = in_path.with_suffix(".compressed.pdf")
        args.extend(["-sOutputFile=" + str(tmp_out), str(in_path)])
        try:
            rprint(f"[cyan]Running:[/cyan] {' '.join(args)}")
            subprocess.run(args, check=True)
            tmp_out.replace(in_path)
            rprint(f"[green]Compressed (inplace):[/green] {in_path}")
        except FileNotFoundError:
            rprint("[red]Ghostscript ('gs') not found. Install it (e.g., `brew install ghostscript`).[/red]")
            raise typer.Exit(code=3)
        except subprocess.CalledProcessError as e:
            rprint(f"[red]Ghostscript failed:[/red] {e}")
            raise typer.Exit(code=4)
    else:
        args.extend(["-sOutputFile=" + str(out_path), str(in_path)])
        try:
            rprint(f"[cyan]Running:[/cyan] {' '.join(args)}")
            subprocess.run(args, check=True)
            rprint(f"[green]Compressed ->[/green] {out_path}")
        except FileNotFoundError:
            rprint("[red]Ghostscript ('gs') not found. Install it (e.g., `brew install ghostscript`).[/red]")
            raise typer.Exit(code=3)
        except subprocess.CalledProcessError as e:
            rprint(f"[red]Ghostscript failed:[/red] {e}")
            raise typer.Exit(code=4)


# ------------------------------- GUI ---------------------------------
@app.command()
def gui(
    port: int = typer.Option(8501, "--port", help="Port to serve Streamlit GUI"),
    browser: bool = typer.Option(True, "--open-browser/--no-open-browser", help="Open browser automatically"),
):
    """
    Launch the Streamlit GUI.
    """
    # Resolve the path to the app module
    app_py = (Path(__file__).resolve().parent / "app_streamlit.py")
    if not app_py.exists():
        rprint(f"[red]Cannot locate app_streamlit.py at {app_py}[/red]")
        raise typer.Exit(code=2)

    cmd = ["streamlit", "run", str(app_py), "--server.port", str(port)]
    if not browser:
        cmd.extend(["--server.headless", "true"])

    rprint(f"[cyan]Launching:[/cyan] {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        rprint("[red]Streamlit not found. Install it in your environment (`pip install streamlit`).[/red]")
        raise typer.Exit(code=3)
    except subprocess.CalledProcessError as e:
        rprint(f"[red]Streamlit exited with error:[/red] {e}")
        raise typer.Exit(code=4)


# ------------------------------- MAIN --------------------------------
def app_main() -> None:
    """Entry point for console_scripts."""
    try:
        app()
    except KeyboardInterrupt:
        rprint("\n[red]Interrupted[/red]")
        sys.exit(130)


if __name__ == "__main__":
    app_main()