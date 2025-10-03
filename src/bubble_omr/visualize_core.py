# src/bubble_omr/visualize_core.py

from __future__ import annotations
from pathlib import Path
from typing import Iterable, Optional, Tuple, Union

from .tools.bubble_score import load_config, Config, GridSpec
from .tools.zone_visualizer import (
    load_input_image,        # (pdf_path, dpi) -> (bgr_img, (Wt, Ht))
    zone_rect,               # (gs: GridSpec, Wt, Ht) -> (x0,y0,x1,y1)
    cell_rects,              # (gs: GridSpec, Wt, Ht) -> Iterable[(x0,y0,x1,y1)]
    draw_zone_rect,          # (img_bgr, (x0,y0,x1,y1), color=(...), thickness=...)
    draw_cells,              # (img_bgr, cells: Iterable[tuple], color=(...), thickness=...)
)

def _overlay_one_grid(
    img_bgr,
    gs: GridSpec,
    Wt: int,
    Ht: int,
    zone_color: Tuple[int, int, int] = (0, 255, 0),
    cell_color: Tuple[int, int, int] = (255, 0, 0),
    thickness: int = 2,
    label: Optional[str] = None,
):
    """Draw a single GridSpec: zone rectangle + all cell rects."""
    x0, y0, x1, y1 = zone_rect(gs, Wt, Ht)
    draw_zone_rect(img_bgr, (x0, y0, x1, y1), color=zone_color, thickness=thickness)

    cells = cell_rects(gs, Wt, Ht)
    draw_cells(img_bgr, cells, color=cell_color, thickness=1)

    if label:
        import cv2 as cv
        cv.putText(
            img_bgr, label, (int(x0) + 6, int(y0) + 22),
            cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv.LINE_AA
        )
        cv.putText(
            img_bgr, label, (int(x0) + 6, int(y0) + 22),
            cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv.LINE_AA
        )

def overlay_config(
    input_pdf: str,
    config_or_path: Union[str, Path, Config],
    out_image: str = "config_overlay.png",
    dpi: int = 300,
    label_blocks: bool = True,
):
    """
    Render the first page of `input_pdf` and overlay all configured grids
    from a Config object or path to YAML/JSON. Writes `out_image` (PNG).
    """
    # 1) Load config if a path/string was provided
    if isinstance(config_or_path, (str, Path)):
        cfg: Config = load_config(str(config_or_path))
    else:
        cfg: Config = config_or_path  # already a Config

    # 2) Render page to BGR
    img_bgr, (Wt, Ht) = load_input_image(input_pdf, dpi=dpi)

    # 3) Answer layout(s)
    answer_blocks: Iterable[GridSpec] = cfg.answer_layouts or ([cfg.answer_layout] if cfg.answer_layout else [])
    for i, gs in enumerate(answer_blocks, start=1):
        lbl = f"answers[{i}]" if label_blocks and (cfg.answer_layouts and len(cfg.answer_layouts) > 1) else "answers"
        _overlay_one_grid(img_bgr, gs, Wt, Ht, zone_color=(0, 255, 0), cell_color=(0, 200, 0), label=lbl)

    # 4) Optional name blocks (last/first/name), ID, version
    if cfg.last_name_layout:
        _overlay_one_grid(img_bgr, cfg.last_name_layout, Wt, Ht, zone_color=(255, 165, 0), cell_color=(255, 140, 0),
                          label="last_name")
    if cfg.first_name_layout:
        _overlay_one_grid(img_bgr, cfg.first_name_layout, Wt, Ht, zone_color=(255, 165, 0), cell_color=(255, 140, 0),
                          label="first_name")
    if cfg.name_layout:
        _overlay_one_grid(img_bgr, cfg.name_layout, Wt, Ht, zone_color=(255, 165, 0), cell_color=(255, 140, 0),
                          label="name")
    if cfg.id_layout:
        _overlay_one_grid(img_bgr, cfg.id_layout, Wt, Ht, zone_color=(0, 165, 255), cell_color=(0, 140, 255),
                          label="id")
    if cfg.version_layout:
        _overlay_one_grid(img_bgr, cfg.version_layout, Wt, Ht, zone_color=(147, 112, 219), cell_color=(138, 43, 226),
                          label="version")

    # 5) Save PNG
    import cv2 as cv
    out_path = Path(out_image).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv.imwrite(str(out_path), img_bgr)
    return str(out_path)