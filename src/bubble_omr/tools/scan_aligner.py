#!/usr/bin/env python3
"""
Unified Scan Alignment Tool (Batch-capable)
==========================================

Align scanned PDF pages to a template using either:
  1) ArUco markers (robust when markers are present), or
  2) Feature matching (ORB keypoints + RANSAC homography) when no markers exist.

This version supports both single- and multi-PDF workflows, with options to:
  • Process one PDF with many pages (default behavior)
  • Process multiple PDFs and write ONE combined output PDF
  • Process multiple PDFs and write ONE OUTPUT PER INPUT (into --out-dir)
  • Restrict pages via --first-page / --last-page
  • Emit a per-page metrics CSV across all processed pages

Dependencies
------------
  pip install opencv-contrib-python numpy Pillow PyMuPDF

Usage (Quick Examples)
----------------------
# 1) Single PDF → single aligned PDF
python scan_aligner.py \
  --template TEMPLATE.pdf \
  --input-pdf scans.pdf \
  --out aligned.pdf \
  --method auto \
  --dpi 300 \
  --save-debug debug_single

# 2) Multiple PDFs → ONE combined output
python scan_aligner.py \
  --template TEMPLATE.pdf \
  --inputs scans_day1.pdf scans_day2.pdf \
  --out aligned_all.pdf \
  --method auto \
  --dpi 300 \
  --metrics-csv metrics_all.csv

# 3) Multiple PDFs → one output PER input
python scan_aligner.py \
  --template TEMPLATE.pdf \
  --inputs scans_day1.pdf scans_day2.pdf \
  --out-dir aligned_outputs \
  --method auto \
  --dpi 300

# 4) Page range (0-based, inclusive bounds)
python scan_aligner.py \
  --template TEMPLATE.pdf \
  --input-pdf scans.pdf \
  --out aligned_subset.pdf \
  --first-page 2 --last-page 9

Notes
-----
* The template can be a PDF (default: first page) or an image file.
* For ArUco, include ≥4 shared markers between template and scan for robust homography.
* For feature matching, adequate texture/contrast improves results.
* When alignment fails and --fallback-original is set, the page is resized to the
  template canvas so the final PDF has uniform page size.

"""
from __future__ import annotations
import argparse
import csv
import os
import sys
from typing import Dict, Tuple, List, Optional, Iterable

import numpy as np
import cv2 as cv
from PIL import Image

# Prefer PyMuPDF for fast, dependency-light rasterization
try:
    import fitz  # PyMuPDF
    _HAVE_PYMUPDF = True
except Exception:
    _HAVE_PYMUPDF = False

# -------------------------
# Image / PDF I/O utilities
# -------------------------

def imread_any(path: str) -> Optional[np.ndarray]:
    """Read image from path into BGR np.ndarray or return None on failure."""
    img = cv.imdecode(np.fromfile(path, dtype=np.uint8), cv.IMREAD_COLOR)
    return img

def pil_to_bgr(img: Image.Image) -> np.ndarray:
    return cv.cvtColor(np.array(img.convert("RGB")), cv.COLOR_RGB2BGR)

def bgr_to_pil(img_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB))

# PDF rendering via PyMuPDF

def render_pdf_page(pdf_path: str, page_index: int = 0, dpi: int = 300) -> Image.Image:
    if not _HAVE_PYMUPDF:
        raise RuntimeError("PyMuPDF not available; please install 'pymupdf'.")
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    with fitz.open(pdf_path) as doc:
        page = doc[page_index]
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return img

# -------------------------
# ArUco-based alignment
# -------------------------


def detect_aruco_centers(img_bgr: np.ndarray, dict_name: str = "DICT_4X4_50") -> Dict[int, Tuple[float, float]]:
    """Return dict id -> (cx, cy) center for detected ArUco markers.
    Accepts a bgr image in numpy.ndarray format (bgr)"""
    gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)
    aruco = cv.aruco
    name = dict_name.upper()
    if not name.startswith("DICT_"):
        name = "DICT_4X4_50"
    DICT_CONST = getattr(aruco, name, aruco.DICT_4X4_50)
    adict = aruco.getPredefinedDictionary(DICT_CONST)
    params = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(adict, params)
    corners, ids, _ = detector.detectMarkers(gray)
    centers: Dict[int, Tuple[float, float]] = {}
    if ids is not None:
        ids = ids.flatten()
        for i, cid in enumerate(ids):
            c = corners[i][0]
            cx = float(np.mean(c[:, 0]))
            cy = float(np.mean(c[:, 1]))
            centers[int(cid)] = (cx, cy)
    return centers

def homography_from_points(src_pts: np.ndarray, dst_pts: np.ndarray, ransac: float = 3.0) -> Optional[np.ndarray]:
    if len(src_pts) >= 4 and len(dst_pts) == len(src_pts):
        H, _ = cv.findHomography(src_pts, dst_pts, cv.RANSAC, ransac)
        return H
    return None

def align_with_aruco(page_bgr: np.ndarray,
                     template_bgr: np.ndarray,
                     dict_name: str = "DICT_4X4_50",
                     min_markers: int = 4,
                     ransac: float = 3.0) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[int]]:
    t_centers = detect_aruco_centers(template_bgr, dict_name)
    p_centers = detect_aruco_centers(page_bgr, dict_name)
    common = sorted(set(t_centers.keys()) & set(p_centers.keys()))
    if len(common) < max(3, min_markers):
        return None, None, common
    src = np.float32([p_centers[i] for i in common]).reshape(-1, 1, 2)
    dst = np.float32([t_centers[i] for i in common]).reshape(-1, 1, 2)
    H = homography_from_points(src, dst, ransac=ransac)
    if H is None:
        return None, None, common
    Ht, Wt = template_bgr.shape[:2]
    aligned = cv.warpPerspective(page_bgr, H, (Wt, Ht))
    return aligned, H, common

# -------------------------
# Feature-based alignment (ORB)
# -------------------------

def align_with_features(page_bgr: np.ndarray,
                        template_bgr: np.ndarray,
                        nfeatures: int = 3000,
                        match_ratio: float = 0.75,
                        ransac: float = 3.0) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int]:
    """Return (aligned_bgr, H, n_inliers) or (None, None, 0) on failure."""
    page_gray = cv.cvtColor(page_bgr, cv.COLOR_BGR2GRAY)
    templ_gray = cv.cvtColor(template_bgr, cv.COLOR_BGR2GRAY)
    orb = cv.ORB_create(nfeatures=nfeatures)
    k1, d1 = orb.detectAndCompute(page_gray, None)
    k2, d2 = orb.detectAndCompute(templ_gray, None)
    if d1 is None or d2 is None:
        return None, None, 0
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(d1, d2, k=2)
    good = [m for m, n in matches if m.distance < match_ratio * n.distance]
    if len(good) < 8:
        return None, None, 0
    src = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv.findHomography(src, dst, cv.RANSAC, ransac)
    if H is None:
        return None, None, 0
    inliers = int(mask.sum()) if mask is not None else 0
    Ht, Wt = template_bgr.shape[:2]
    aligned = cv.warpPerspective(page_bgr, H, (Wt, Ht))
    return aligned, H, inliers

# -------------------------
# PDF writing
# -------------------------

def save_images_as_pdf(pil_images: List[Image.Image], out_path: str, dpi: int = 300) -> None:
    if not pil_images:
        raise ValueError("No pages to save.")
    imgs = [im.convert("RGB") for im in pil_images]
    imgs[0].save(out_path, save_all=True, append_images=imgs[1:], format="PDF", resolution=dpi)

# -------------------------
# Main pipeline helpers
# -------------------------

def load_template_image(template_path: str, dpi: int, template_page: int = 0) -> np.ndarray:
    if template_path.lower().endswith((".pdf")):
        pil_img = render_pdf_page(template_path, template_page, dpi)
        return pil_to_bgr(pil_img)
    else:
        img = imread_any(template_path)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {template_path}")
        return img


def iter_input_pages(input_pdf: str, dpi: int, first: Optional[int], last: Optional[int]) -> Iterable[Tuple[int, Image.Image]]:
    if not _HAVE_PYMUPDF:
        raise RuntimeError("PyMuPDF not available; please install 'pymupdf'.")
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    doc = fitz.open(input_pdf)
    try:
        n = len(doc)
        start = 0 if first is None else max(0, first)
        end = n - 1 if last is None else min(n - 1, last)
        if end < start:
            return
        for i in range(start, end + 1):
            page = doc[i]
            pix = page.get_pixmap(matrix=mat, alpha=False)
            pil_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            yield i, pil_img
    finally:
        doc.close()


def process_one_pdf(pdf_path: str,
                    template_bgr: np.ndarray,
                    args) -> Tuple[List[Image.Image], List[dict]]:
    """Process a single PDF; return (aligned PIL pages, per-page metrics dicts)."""
    Ht, Wt = template_bgr.shape[:2]
    out_pages: List[Image.Image] = []
    metrics: List[dict] = []

    for i, page_pil in iter_input_pages(pdf_path, dpi=args.dpi, first=args.first_page, last=args.last_page):
        page_idx_1 = i + 1
        page_bgr = pil_to_bgr(page_pil)
        status = ""
        aligned_bgr: Optional[np.ndarray] = None
        method_used = None
        detail_val = None

        try_methods = []
        if args.method == "aruco":
            try_methods = ["aruco"]
        elif args.method == "feature":
            try_methods = ["feature"]
        else:
            try_methods = ["aruco", "feature"]

        H_used = None
        aruco_common = None
        feature_inliers = None

        for m in try_methods:
            if m == "aruco":
                aligned_bgr, H_used, common = align_with_aruco(
                    page_bgr, template_bgr,
                    dict_name=args.dict_name,
                    min_markers=args.min_markers,
                    ransac=args.ransac,
                )
                status = f"aruco: {len(common)} markers"
                aruco_common = len(common)
                if aligned_bgr is not None:
                    method_used = "aruco"
                    detail_val = aruco_common
                    break
            else:
                aligned_bgr, H_used, n_inliers = align_with_features(
                    page_bgr, template_bgr,
                    nfeatures=args.orb_nfeatures,
                    match_ratio=args.match_ratio,
                    ransac=args.ransac,
                )
                status = f"feature: {n_inliers} inliers" if H_used is not None else "feature: failed"
                feature_inliers = int(n_inliers)
                if aligned_bgr is not None:
                    method_used = "feature"
                    detail_val = feature_inliers
                    break

        if aligned_bgr is None:
            if args.fallback_original:
                # resize to template canvas so output has consistent page size
                aligned_bgr = cv.resize(page_bgr, (Wt, Ht), interpolation=cv.INTER_LINEAR)
                status = (status + "; fallback original").strip()
                method_used = method_used or "fallback"
            else:
                raise RuntimeError(f"{os.path.basename(pdf_path)} page {page_idx_1}: alignment failed.")

        # Save debug overlays if requested
        if args.save_debug:
            os.makedirs(args.save_debug, exist_ok=True)
            dbg = aligned_bgr.copy()
            cv.rectangle(dbg, (2, 2), (Wt - 3, Ht - 3), (0, 255, 0), 2)
            cv.putText(dbg, f"{os.path.basename(pdf_path)} p{page_idx_1}: {status}", (20, 40), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2, cv.LINE_AA)
            cv.imwrite(os.path.join(args.save_debug, f"{os.path.splitext(os.path.basename(pdf_path))[0]}_p{page_idx_1:03d}.png"), dbg)

        out_pages.append(bgr_to_pil(aligned_bgr))

        metrics.append({
            "pdf": os.path.basename(pdf_path),
            "page_index": i,
            "page_number": page_idx_1,
            "method": method_used,
            "aruco_common_markers": aruco_common if aruco_common is not None else "",
            "feature_inliers": feature_inliers if feature_inliers is not None else "",
            "status": status,
        })

    return out_pages, metrics

# -------------------------
# CLI
# -------------------------

def main():
    ap = argparse.ArgumentParser(description="Align scanned PDF pages to a template using ArUco markers or feature matching.")

    # Inputs
    g_in = ap.add_mutually_exclusive_group(required=True)
    g_in.add_argument("--input-pdf", help="Single input scanned PDF to align.")
    g_in.add_argument("--inputs", nargs="+", help="Multiple input PDFs to align.")

    ap.add_argument("--template", required=True, help="Template file (PDF or image). If PDF, use --template-page to select page.")

    # Output routing
    ap.add_argument("--out", help="Output PDF path. Use for single input or for combined output with --inputs.")
    ap.add_argument("--out-dir", help="Directory for per-input outputs when using --inputs. If set, writes one aligned PDF per input.")

    # Common options
    ap.add_argument("--dpi", type=int, default=300, help="DPI for rasterization. Default: 300")
    ap.add_argument("--template-page", type=int, default=0, help="Template page index if template is a PDF. Default: 0")
    ap.add_argument("--method", choices=["aruco", "feature", "auto"], default="auto", help="Alignment method. 'auto' tries ArUco then feature.")

    # Page range
    ap.add_argument("--first-page", type=int, default=None, help="0-based first page to process (inclusive).")
    ap.add_argument("--last-page", type=int, default=None, help="0-based last page to process (inclusive).")

    # ArUco options
    ap.add_argument("--dict", dest="dict_name", default="DICT_4X4_50", help="ArUco dictionary, e.g. DICT_4X4_50, DICT_5X5_100, etc.")
    ap.add_argument("--min-markers", type=int, default=4, help="Minimum common markers for ArUco alignment. 4+ recommended.")
    ap.add_argument("--ransac", type=float, default=3.0, help="RANSAC reprojection threshold (pixels) for homography.")

    # Feature options
    ap.add_argument("--orb-nfeatures", type=int, default=3000, help="ORB nfeatures for feature-based alignment.")
    ap.add_argument("--match-ratio", type=float, default=0.75, help="Lowe ratio for KNN matches (0–1). Lower is stricter.")

    # Misc
    ap.add_argument("--fallback-original", action="store_true", help="If alignment fails, include the unaligned (resized) page.")
    ap.add_argument("--save-debug", metavar="DIR", help="If set, write per-page debug PNGs into this directory.")
    ap.add_argument("--metrics-csv", help="Write a CSV with per-page alignment metrics across all processed pages.")

    args = ap.parse_args()

    # Validate outputs
    multiple_inputs = args.inputs is not None
    if multiple_inputs and not (args.out or args.out_dir):
        ap.error("With --inputs, specify either --out (combined single PDF) or --out-dir (one PDF per input).")
    if not multiple_inputs and not args.out:
        ap.error("With --input-pdf, you must specify --out.")
    if args.out_dir and not multiple_inputs:
        ap.error("--out-dir is only for use with --inputs.")

    # Validate paths
    if multiple_inputs:
        for p in args.inputs:
            if not os.path.exists(p):
                sys.exit(f"Input PDF not found: {p}")
    else:
        if not os.path.exists(args.input_pdf):
            sys.exit(f"Input PDF not found: {args.input_pdf}")

    if not os.path.exists(args.template):
        sys.exit(f"Template not found: {args.template}")

    # Load template once
    template_bgr = load_template_image(args.template, dpi=args.dpi, template_page=args.template_page)

    all_metrics: List[dict] = []

    if not multiple_inputs:
        pages, metrics = process_one_pdf(args.input_pdf, template_bgr, args)
        save_images_as_pdf(pages, args.out, dpi=args.dpi)
        print(f"Wrote aligned PDF -> {args.out}")
        all_metrics.extend(metrics)
    else:
        if args.out_dir:
            os.makedirs(args.out_dir, exist_ok=True)
            for p in args.inputs:
                pages, metrics = process_one_pdf(p, template_bgr, args)
                out_path = os.path.join(args.out_dir, f"{os.path.splitext(os.path.basename(p))[0]}_aligned.pdf")
                save_images_as_pdf(pages, out_path, dpi=args.dpi)
                print(f"Wrote aligned PDF -> {out_path}")
                all_metrics.extend(metrics)
        else:
            # Combined output
            combined_pages: List[Image.Image] = []
            for p in args.inputs:
                pages, metrics = process_one_pdf(p, template_bgr, args)
                combined_pages.extend(pages)
                all_metrics.extend(metrics)
            save_images_as_pdf(combined_pages, args.out, dpi=args.dpi)
            print(f"Wrote combined aligned PDF -> {args.out}")

    # Write metrics CSV if requested
    if args.metrics_csv:
        fieldnames = ["pdf", "page_index", "page_number", "method", "aruco_common_markers", "feature_inliers", "status"]
        with open(args.metrics_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for row in all_metrics:
                w.writerow(row)
        print(f"Wrote metrics CSV -> {args.metrics_csv}")


if __name__ == "__main__":
    main()
