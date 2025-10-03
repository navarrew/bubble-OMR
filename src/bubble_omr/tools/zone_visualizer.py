# src/bubble_omr/tools/zone_visualizer.py
from __future__ import annotations
from typing import Iterable, List, Tuple
from pathlib import Path

import cv2 as cv
import numpy as np
import fitz  # PyMuPDF

# ---------- I/O ----------
def load_input_image(path: str, dpi: int = 300) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Load first page of a PDF at the given DPI, or a raster image.
    Returns (BGR image, (width_px, height_px)).
    """
    p = str(path)
    if p.lower().endswith(".pdf"):
        doc = fitz.open(p)
        if len(doc) < 1:
            raise RuntimeError(f"No pages in PDF: {p}")
        page = doc.load_page(0)
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
        doc.close()
    else:
        img = cv.imread(p, cv.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {p}")
    Ht, Wt = img.shape[:2]
    return img, (Wt, Ht)

# ---------- geometry helpers (normalized zone -> pixel rects) ----------
def zone_rect(gs, Wt: int, Ht: int) -> Tuple[int, int, int, int]:
    """
    Convert gs.zone = (x0,y0,x1,y1) in 0..1 normalized coords to integer pixel coords.
    Returns (x0, y0, x1, y1).
    """
    x0n, y0n, x1n, y1n = gs.zone
    x0 = int(round(x0n * Wt))
    y0 = int(round(y0n * Ht))
    x1 = int(round(x1n * Wt))
    y1 = int(round(y1n * Ht))
    # clamp & ensure at least 1px size
    x0 = max(0, min(x0, Wt - 1)); y0 = max(0, min(y0, Ht - 1))
    x1 = max(x0 + 1, min(x1, Wt));  y1 = max(y0 + 1, min(y1, Ht))
    return x0, y0, x1, y1

def cell_rects(gs, Wt: int, Ht: int) -> List[Tuple[int, int, int, int]]:
    """
    Compute per-cell rectangles within the zone for a grid of (rows x cols).
    Returns list of (x0, y0, x1, y1).
    """
    x0, y0, x1, y1 = zone_rect(gs, Wt, Ht)
    rows, cols = int(gs.rows), int(gs.cols)
    pad = float(getattr(gs, "pad", 0.12) or 0.0)
    W = x1 - x0; H = y1 - y0
    cw = W / cols; ch = H / rows
    px = pad * cw; py = pad * ch

    rects: List[Tuple[int, int, int, int]] = []
    for r in range(rows):
        for c in range(cols):
            rx0 = int(round(x0 + c * cw + px))
            ry0 = int(round(y0 + r * ch + py))
            rx1 = int(round(x0 + (c + 1) * cw - px))
            ry1 = int(round(y0 + (r + 1) * ch - py))
            # ensure valid
            if rx1 <= rx0: rx1 = rx0 + 1
            if ry1 <= ry0: ry1 = ry0 + 1
            rects.append((rx0, ry0, rx1, ry1))
    return rects

# ---------- drawing ----------
def draw_zone_rect(img_bgr: np.ndarray, rect: Tuple[int, int, int, int],
                   color=(0, 255, 0), thickness: int = 2) -> None:
    x0, y0, x1, y1 = rect
    cv.rectangle(img_bgr, (x0, y0), (x1, y1), color, thickness)

def draw_cells(img_bgr: np.ndarray, rects: Iterable[Tuple[int, int, int, int]],
               color=(0, 255, 0), thickness: int = 1, shape: str | None = None) -> None:
    """
    Draw each cell. If `shape == "circle"` or gs.shape == "circle" at the callsite,
    draw inscribed circles, else rectangles.
    """
    for (x0, y0, x1, y1) in rects:
        if shape == "circle":
            cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
            r = int(round(0.45 * min(x1 - x0, y1 - y0)))
            cv.circle(img_bgr, (cx, cy), max(1, r), color, thickness)
        else:
            cv.rectangle(img_bgr, (x0, y0), (x1, y1), color, thickness)
