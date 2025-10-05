# src/bubble_omr/grade_core.py
from __future__ import annotations
from typing import Optional, List, Tuple
import os
import csv
import cv2
import numpy as np

from .tools.bubble_score import (
    load_pages,            # load PDF pages or images into BGR arrays
    process_page_all,      # decode last/first name, ID, version, answers
    load_key_txt,          # load key as list[str] like ["A","B",...]
    score_against_key,     # compare predictions to key
)
from .config_io import load_config, Config, GridLayout
from .tools.zone_visualizer import (
    grid_centers_axis_mode,
    centers_to_radius_px,
)

def _ensure_dir(path: str) -> None:
    if path and not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)

def _annotate_page(img_bgr: np.ndarray, cfg: Config,
                   color=(0, 255, 0), thickness=2) -> np.ndarray:
    """Draw the exact circles used for decoding (for debugging/visual QA)."""
    h, w = img_bgr.shape[:2]
    out = img_bgr.copy()
    def _draw_layout(layout: GridLayout):
        centers = grid_centers_axis_mode(
            layout.x_topleft, layout.y_topleft,
            layout.x_bottomright, layout.y_bottomright,
            layout.questions, layout.choices
        )
        pts, r_px = centers_to_radius_px(centers, w, h, layout.radius_pct)
        for (cx, cy) in pts:
            cv2.circle(out, (cx, cy), r_px, color, thickness, lineType=cv2.LINE_AA)

    for lay in (cfg.answer_layouts or []):
        _draw_layout(lay)
    for name in ("last_name_layout", "first_name_layout", "id_layout", "version_layout"):
        lay = getattr(cfg, name, None)
        if lay is not None:
            _draw_layout(lay)
    return out

def grade_pdf(
    input_path: str,
    config_path: str,
    out_csv: str,
    key_txt: Optional[str] = None,
    out_annotated_dir: Optional[str] = None,
    dpi: int = 300,
    min_fill: float = 0.20,
    top2_ratio: float = 0.80,
) -> str:
    """
    Grade a PDF or image stack using axis-based geometry.

    Behavior:
      - If key is provided: limit output columns and scoring to first len(key) questions.
      - If no key: output all decoded questions.
    """
    cfg: Config = load_config(config_path)

    # Load pages and optional key
    pages = load_pages([input_path], dpi=dpi)
    key: Optional[List[str]] = load_key_txt(key_txt) if key_txt else None

    # Determine how many Qs to output
    total_q = sum(a.questions for a in cfg.answer_layouts)
    q_out = len(key) if key else total_q
    q_out = max(0, min(q_out, total_q))  # clamp

    # CSV header
    header = ["page_index", "LastName", "FirstName", "StudentID", "Version"] \
             + [f"Q{i+1}" for i in range(q_out)]
    if key:
        header += ["score", "total"]

    _ensure_dir(os.path.dirname(out_csv) or ".")
    if out_annotated_dir:
        _ensure_dir(out_annotated_dir)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for page_idx, img_bgr in enumerate(pages, start=1):
            info, answers = process_page_all(
                img_bgr, cfg, min_fill=min_fill, top2_ratio=top2_ratio
            )

            # Slice answers to the number of columns we want to emit
            answers_out = answers[:q_out]
            answers_csv = [a if a is not None else "" for a in answers_out]

            row = [
                str(page_idx),
                info.get("last_name", ""),
                info.get("first_name", ""),
                info.get("student_id", ""),
                info.get("version", ""),
            ] + answers_csv

            if key:
                # Also slice key in case it’s longer than decoded answers
                key_out = key[:q_out]
                got, tot = score_against_key([a or "" for a in answers_out], key_out)
                row += [str(got), str(tot)]

            writer.writerow(row)

            # Optional per-page overlay for QA
            if out_annotated_dir:
                vis = _annotate_page(img_bgr, cfg, color=(0, 255, 0), thickness=2)
                out_png = os.path.join(out_annotated_dir, f"page_{page_idx:03d}_overlay.png")
                cv2.imwrite(out_png, vis)

    return out_csv