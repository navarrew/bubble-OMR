#!/usr/bin/env python3
"""
bubble_score.py
---------------
Axis-based grading for bubble OMR sheets (answers + names + ID + version).

Uses config_io.Config with per-layout axis definitions:
  x_topleft, y_topleft, x_bottomright, y_bottomright, radius_pct,
  questions (rows), choices (cols), selection_axis ("row" or "col"), labels (for rows).

Decoding behavior:
- selection_axis == "row":    pick ONE column (choice) per row  (answers, version-as-row).
- selection_axis == "col":    pick ONE row (label) per column  (last/first name, ID).

Outputs CSV with columns:
  page_index, LastName, FirstName, StudentID, Version, Q1..Qn, [score, total if key provided]
"""

from __future__ import annotations
import argparse
import csv
import os
from typing import List, Tuple, Iterable, Optional

import numpy as np
import cv2

from ..config_io import load_config, GridLayout, Config

# ------------------------------------------------------------------------------
# Geometry
# ------------------------------------------------------------------------------

def grid_centers_axis_mode(
    x_tl: float, y_tl: float, x_br: float, y_br: float,
    rows: int, cols: int
) -> List[Tuple[float, float]]:
    """
    Return normalized (x%, y%) centers for a rows x cols grid
    by interpolating between top-left and bottom-right bubble centers.
    """
    centers: List[Tuple[float, float]] = []
    r_den = max(1, rows - 1)
    c_den = max(1, cols - 1)
    for r in range(rows):
        v = r / r_den
        y = y_tl + (y_br - y_tl) * v
        for c in range(cols):
            u = c / c_den
            x = x_tl + (x_br - x_tl) * u
            centers.append((x, y))
    return centers


def centers_to_circle_rois(
    centers_pct: Iterable[Tuple[float, float]],
    img_w: int, img_h: int,
    radius_pct: float
) -> List[Tuple[int, int, int, int]]:
    r_px = max(1.0, radius_pct * img_w)
    rois: List[Tuple[int, int, int, int]] = []
    for (cxp, cyp) in centers_pct:
        cx = float(cxp) * img_w
        cy = float(cyp) * img_h
        x = int(round(cx - r_px))
        y = int(round(cy - r_px))
        w = h = int(round(2 * r_px))
        x = max(0, min(x, img_w - 1))
        y = max(0, min(y, img_h - 1))
        if x + w > img_w:
            w = img_w - x
        if y + h > img_h:
            h = img_h - y
        rois.append((x, y, w, h))
    return rois


def circle_mask(w: int, h: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    cx, cy = w / 2.0, h / 2.0
    r = min(w, h) / 2.0
    cv2.circle(mask, (int(round(cx)), int(round(cy))), int(round(r)), 255, thickness=-1, lineType=cv2.LINE_AA)
    return mask

# ------------------------------------------------------------------------------
# Scoring primitives
# ------------------------------------------------------------------------------

def roi_darkness_scores(gray: np.ndarray, rois: List[Tuple[int,int,int,int]]) -> List[float]:
    """Return 0..1 darkness score (mean inverted intensity inside circle) for each ROI."""
    scores: List[float] = [0.0] * len(rois)
    mask_cache: dict[Tuple[int, int], np.ndarray] = {}
    for idx, (x, y, w, h) in enumerate(rois):
        crop = gray[y:y+h, x:x+w]
        key = (w, h)
        if key not in mask_cache:
            mask_cache[key] = circle_mask(w, h)
        m = mask_cache[key]
        inside = crop[m == 255]
        scores[idx] = float(np.mean(255 - inside) / 255.0) if inside.size else 0.0
    return scores


def select_per_row(
    scores: List[float], rows: int, cols: int, min_fill: float, top2_ratio: float
) -> List[Optional[int]]:
    """For each row, pick the best column if confident; else None."""
    out: List[Optional[int]] = []
    for r in range(rows):
        row_scores = scores[r*cols:(r+1)*cols]
        order = np.argsort(row_scores)[::-1]
        best = order[0]
        best_val = row_scores[best]
        second_val = row_scores[order[1]] if cols > 1 else 0.0
        if best_val >= min_fill and (cols == 1 or (second_val <= top2_ratio * best_val)):
            out.append(int(best))
        else:
            out.append(None)
    return out


def select_per_col(
    scores: List[float], rows: int, cols: int, min_fill: float, top2_ratio: float
) -> List[Optional[int]]:
    """For each column, pick the best row if confident; else None."""
    out: List[Optional[int]] = []
    for c in range(cols):
        col_scores = [scores[r*cols + c] for r in range(rows)]
        order = np.argsort(col_scores)[::-1]
        best = order[0]
        best_val = col_scores[best]
        second_val = col_scores[order[1]] if rows > 1 else 0.0
        if best_val >= min_fill and (rows == 1 or (second_val <= top2_ratio * best_val)):
            out.append(int(best))
        else:
            out.append(None)
    return out

# ------------------------------------------------------------------------------
# Zone decoders
# ------------------------------------------------------------------------------

def decode_layout(
    gray: np.ndarray, layout: GridLayout,
    min_fill: float, top2_ratio: float
):
    """
    Decode one layout according to its selection_axis.
    Returns (selected_indices, rois, scores).
    - If selection_axis == "row": indices are column indices per row (len = rows)
    - If selection_axis == "col": indices are row indices per column (len = cols)
    """
    h, w = gray.shape[:2]
    centers = grid_centers_axis_mode(
        layout.x_topleft, layout.y_topleft,
        layout.x_bottomright, layout.y_bottomright,
        layout.questions, layout.choices
    )
    rois = centers_to_circle_rois(centers, w, h, layout.radius_pct)
    scores = roi_darkness_scores(gray, rois)

    if layout.selection_axis == "row":
        picked = select_per_row(scores, layout.questions, layout.choices, min_fill, top2_ratio)
    else:
        picked = select_per_col(scores, layout.questions, layout.choices, min_fill, top2_ratio)

    return picked, rois, scores


def indices_to_labels_row(picked: List[Optional[int]], choices: int, choice_labels: List[str]) -> List[Optional[str]]:
    """Map per-row selected column index → label (A.. or from provided)."""
    out: List[Optional[str]] = []
    for idx in picked:
        out.append(choice_labels[idx] if idx is not None and 0 <= idx < choices else None)
    return out


def indices_to_text_col(picked: List[Optional[int]], row_labels: str) -> str:
    """Map per-column selected row index → character; None → blank."""
    chars: List[str] = []
    for idx in picked:
        if idx is None or idx < 0 or idx >= len(row_labels):
            chars.append("")
        else:
            chars.append(row_labels[idx])
    return "".join(chars)

# ------------------------------------------------------------------------------
# Key handling & scoring
# ------------------------------------------------------------------------------

def load_key_txt(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    chars = [c for c in raw if c.isalpha()]
    return [c.upper() for c in chars]


def score_against_key(selections: List[Optional[str]], key: List[str]) -> Tuple[int, int]:
    correct = 0
    total = min(len(selections), len(key))
    for i in range(total):
        if selections[i] is not None and selections[i] == key[i]:
            correct += 1
    return correct, total

# ------------------------------------------------------------------------------
# Page processing
# ------------------------------------------------------------------------------

def process_page_all(
    img_bgr: np.ndarray,
    cfg: Config,
    min_fill: float = 0.20,
    top2_ratio: float = 0.80
):
    """
    Decode ID, names, version (if present) and all answer_layouts.
    Returns:
      info = dict(last_name, first_name, student_id, version)
      answers = list[Optional[str]]  # Q1..Qn
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Names / ID / Version
    info = {"last_name": "", "first_name": "", "student_id": "", "version": ""}

    if cfg.last_name_layout:
        picked, _, _ = decode_layout(gray, cfg.last_name_layout, min_fill, top2_ratio)
        # per-column picks (len = choices); map rows->letters
        info["last_name"] = indices_to_text_col(picked, cfg.last_name_layout.labels or " ABCDEFGHIJKLMNOPQRSTUVWXYZ").strip()

    if cfg.first_name_layout:
        picked, _, _ = decode_layout(gray, cfg.first_name_layout, min_fill, top2_ratio)
        info["first_name"] = indices_to_text_col(picked, cfg.first_name_layout.labels or " ABCDEFGHIJKLMNOPQRSTUVWXYZ").strip()

    if cfg.id_layout:
        picked, _, _ = decode_layout(gray, cfg.id_layout, min_fill, top2_ratio)
        info["student_id"] = indices_to_text_col(picked, cfg.id_layout.labels or "0123456789")

    if cfg.version_layout:
        picked, _, _ = decode_layout(gray, cfg.version_layout, min_fill, top2_ratio)
        # selection_axis likely "row" with one row; picked is per-row list of column indices
        if cfg.version_layout.selection_axis == "row":
            # pick first (only) row
            idx = picked[0] if picked else None
            labels = list(cfg.version_layout.labels or "ABCD")
            info["version"] = labels[idx] if idx is not None and 0 <= idx < len(labels) else ""
        else:
            # column-wise selection (rare); treat as characters-in-columns
            info["version"] = indices_to_text_col(picked, cfg.version_layout.labels or "ABCD")

    # Answers
    answers: List[Optional[str]] = []
    for i, layout in enumerate(cfg.answer_layouts):
        picked, _, _ = decode_layout(gray, layout, min_fill, top2_ratio)  # per-row indices
        choice_labels = list(layout.labels) if layout.labels else [chr(ord('A') + k) for k in range(layout.choices)]
        row_labels = indices_to_labels_row(picked, layout.choices, choice_labels)
        answers.extend(row_labels)

    return info, answers

# ------------------------------------------------------------------------------
# I/O helpers
# ------------------------------------------------------------------------------

def load_pages(paths: List[str], dpi: int = 300) -> List[np.ndarray]:
    images: List[np.ndarray] = []
    for p in paths:
        ext = os.path.splitext(p)[1].lower()
        if ext in (".pdf",):
            try:
                from pdf2image import convert_from_path
            except Exception as e:
                raise RuntimeError("pdf2image is required to read PDFs. Install with `pip install pdf2image`") from e
            pil_pages = convert_from_path(p, dpi=dpi)
            for pg in pil_pages:
                rgb = np.array(pg)  # HxWx3 RGB
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                images.append(bgr)
        else:
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"Could not read image: {p}")
            images.append(img)
    return images

# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Grade OMR sheets (answers + names + ID + version) using axis-based centers.")
    ap.add_argument("--config", required=True, help="YAML config path (axis-mode fields).")
    ap.add_argument("--key-txt", required=False, help="Answer key text file (A/B/C/...).")
    ap.add_argument("--out-csv", default="results.csv", help="Output CSV path.")
    ap.add_argument("--dpi", type=int, default=300, help="DPI when rasterizing PDFs.")
    ap.add_argument("--min-fill", type=float, default=0.20, help="Min fill darkness (0..1) to accept a bubble.")
    ap.add_argument("--top2-ratio", type=float, default=0.80, help="Second-best <= top2_ratio * best to accept.")
    ap.add_argument("inputs", nargs="+", help="PDF(s) or image(s) to grade.")
    args = ap.parse_args()

    cfg = load_config(args.config)

    # Optional key
    key: Optional[List[str]] = load_key_txt(args.key_txt) if args.key_txt else None

    pages = load_pages(args.inputs, dpi=args.dpi)

    # CSV header
    total_q = sum(a.questions for a in cfg.answer_layouts)
    header = ["page_index", "LastName", "FirstName", "StudentID", "Version"] + [f"Q{i+1}" for i in range(total_q)]
    if key:
        header += ["score", "total"]

    rows_out: List[List[str]] = []

    for pi, img_bgr in enumerate(pages):
        info, answers = process_page_all(img_bgr, cfg, min_fill=args.min_fill, top2_ratio=args.top2_ratio)

        # Score if key provided
        row = [str(pi), info["last_name"], info["first_name"], info["student_id"], info["version"]]
        ans_for_csv = [a if a is not None else "" for a in answers]
        row.extend(ans_for_csv)

        if key:
            got, tot = score_against_key([a or "" for a in answers], key)
            row += [str(got), str(tot)]

        rows_out.append(row)

    # Write CSV
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows_out:
            w.writerow(r)

    print(f"[OK] Wrote {args.out_csv} with {len(rows_out)} row(s).")


if __name__ == "__main__":
    main()