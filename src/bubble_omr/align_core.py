import os
import shutil
import subprocess
from types import SimpleNamespace
from typing import Optional
from .tools.scan_aligner import load_template_image, process_one_pdf, save_images_as_pdf

def _has_ghostscript() -> bool:
    """Return True if 'gs' is found on PATH."""
    return shutil.which("gs") is not None

def compress_pdf_with_gs(in_pdf: str, out_pdf: Optional[str] = None, quality: str = "ebook") -> str:
    """
    Compress a PDF using Ghostscript.

    quality: one of 'screen', 'ebook', 'printer', 'prepress'
    Returns the path to the compressed PDF.
    """
    if not _has_ghostscript():
        raise FileNotFoundError("Ghostscript (gs) not found on PATH")

    if out_pdf is None:
        if in_pdf.lower().endswith(".pdf"):
            out_pdf = in_pdf[:-4] + "_small.pdf"
        else:
            out_pdf = in_pdf + "_small.pdf"

    cmd = [
        "gs",
        "-sDEVICE=pdfwrite",
        "-dCompatibilityLevel=1.4",
        f"-dPDFSETTINGS=/{quality}",
        "-dNOPAUSE",
        "-dQUIET",
        "-dBATCH",
        f"-sOutputFile={out_pdf}",
        in_pdf,
    ]
    subprocess.run(cmd, check=True)
    return out_pdf

def align_single_pdf(
    input_pdf: str,
    template: str,
    out_pdf: str = "aligned_scans.pdf",
    dpi: int = 300,
    template_page: int = 0,
    method: str = "auto",
    dict_name: str = "DICT_4X4_50",
    min_markers: int = 4,
    ransac: float = 3.0,
    orb_nfeatures: int = 3000,
    match_ratio: float = 0.75,
    first_page: Optional[int] = None,
    last_page: Optional[int] = None,
    save_debug: Optional[str] = None,
    fallback_original: bool = True,
    # Compression options
    compress_pdf: bool = False,
    compress_quality: str = "ebook",
    compress_inplace: bool = True,
) -> str:
    # Load template page as BGR image used for alignment
    template_bgr = load_template_image(template, dpi=dpi, template_page=template_page)

    # Arguments expected by process_one_pdf (match your scan_aligner.py)
    args = SimpleNamespace(
        dpi=dpi,
        first_page=first_page,
        last_page=last_page,
        dict_name=dict_name,
        min_markers=min_markers,
        ransac=ransac,
        orb_nfeatures=orb_nfeatures,
        match_ratio=match_ratio,
        method=method,
        save_debug=save_debug,
        fallback_original=fallback_original,
    )

    # *** THIS is the line that was missing: produce 'pages' ***
    pages, _metrics = process_one_pdf(input_pdf, template_bgr, args)

    # Write aligned pages to a single PDF
    save_images_as_pdf(pages, out_pdf, dpi=dpi)

    # Optional Ghostscript recompression
    if compress_pdf:
        try:
            if compress_inplace:
                tmp_out = out_pdf + ".tmp.pdf"
                compress_pdf_with_gs(out_pdf, tmp_out, quality=compress_quality)
                os.replace(tmp_out, out_pdf)
                return out_pdf
            else:
                small = compress_pdf_with_gs(out_pdf, None, quality=compress_quality)
                return small
        except FileNotFoundError:
            print("[align] Ghostscript not found on PATH; skipping compression.")
        except subprocess.CalledProcessError as e:
            print(f"[align] Ghostscript failed ({e}); keeping uncompressed PDF.")

    return out_pdf