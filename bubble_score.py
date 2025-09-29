#!/usr/bin/env python3
r"""
bubble_scan.py

Merged Scantron/REMARK bubble sheet reader:
- Keeps v5-style image pipeline, CLI knobs, and annotated outputs.
- Restores "good CSV": KEY row; last_name, first_name, id, version; Q1..Qn capped by --total-questions.
- Draws green/red circles for correct/incorrect; optional label of fill% on each bubble.

Dependencies:
  pip install opencv-python numpy pillow
  # For PDF input (choose one):
  pip install pdf2image poppler-utils   # preferred
  # or:
  pip install pymupdf

Example:
  python bubble_reader_merged.py \
    --config config.json \
    --key-txt key.txt \
    --total-questions 40 \
    --out-csv results.csv \ <- the csv file with the raw scanned data
    --out-annotated-dir annotated_pages \ <- visual representations of each scan in a folder
    --annotate-all-cells \ <-draws circles around each bubble to show you how well it aligns
    --label-density \  <-writes out the pct filled for each bubble
    --min-fill 0.35 --min-score 1.6 --top2-ratio 0.70 \ <-adjustment thresholds
    aligned_scans.pdf
"""

import os, sys, json, csv, math, argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import cv2

# ---------- Utils ----------

def imread_any(path: str) -> List[np.ndarray]:
    """
    Read image(s). If PDF, convert pages to images; else return single image list.
    Tries pdf2image first, then PyMuPDF (fitz). Returns BGR images for OpenCV.
    """
    lower = path.lower()
    if lower.endswith(".pdf"):
        pages = []
        try:
            # pdf2image route (requires poppler)
            from pdf2image import convert_from_path
            pil_pages = convert_from_path(path, dpi=300)
            for p in pil_pages:
                rgb = np.array(p)  # RGB
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                pages.append(bgr)
            return pages
        except Exception:
            try:
                import fitz
                doc = fitz.open(path)
                for i in range(len(doc)):
                    page = doc[i]
                    pix = page.get_pixmap(dpi=300, alpha=False)
                    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
                    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    pages.append(bgr)
                doc.close()
                return pages
            except Exception as e:
                raise RuntimeError(f"Failed to rasterize PDF. Install pdf2image+poppler or PyMuPDF. Error: {e}")
    else:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        return [img]

def ensure_dir(path: str):
    if path and not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)

def clamp_int(v, lo, hi): return max(lo, min(hi, int(v)))

def subimage(img: np.ndarray, rect: Tuple[int,int,int,int]) -> np.ndarray:
    x0, y0, x1, y1 = rect
    x0, y0 = max(0,x0), max(0,y0)
    x1, y1 = min(img.shape[1], x1), min(img.shape[0], y1)
    if x1 <= x0 or y1 <= y0:
        return img[0:0, 0:0]
    return img[y0:y1, x0:x1]

# ---------- Geometry ----------

def cell_rects(zone_rect_abs: Tuple[int,int,int,int], rows: int, cols: int, pad: float=0.12) -> List[Tuple[int,int,int,int]]:
    """
    Robust, balanced padding per cell. pad is total fraction (e.g., 0.12 => 6% per side).
    """
    x0, y0, x1, y1 = zone_rect_abs
    W = max(1, x1 - x0); H = max(1, y1 - y0)
    cw = W / cols; ch = H / rows
    px = int(cw * pad * 0.5); py = int(ch * pad * 0.5)
    rects = []
    for r in range(rows):
        for c in range(cols):
            cx0 = int(x0 + c*cw + px); cy0 = int(y0 + r*ch + py)
            cx1 = int(x0 + (c+1)*cw - px); cy1 = int(y0 + (r+1)*ch - py)
            rects.append((cx0, cy0, cx1, cy1))
    return rects

def draw_circle_annotation(vis: np.ndarray, rect: Tuple[int,int,int,int], color: Tuple[int,int,int], thickness: int=2):
    x0,y0,x1,y1 = rect
    cx = (x0+x1)//2
    cy = (y0+y1)//2
    r  = max(2, int(0.48*min(x1-x0, y1-y0)))
    cv2.circle(vis, (cx,cy), r, color, thickness, lineType=cv2.LINE_AA)

def put_center_text(vis: np.ndarray, rect: Tuple[int,int,int,int], text: str, scale: float=0.5, color=(255,255,255)):
    x0,y0,x1,y1 = rect
    cx = (x0+x1)//2
    cy = (y0+y1)//2
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    org = (int(cx - tw/2), int(cy + th/2))
    cv2.putText(vis, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)

# ---------- Threshold & fill ----------

def load_and_warp(img: np.ndarray, target_height: int=1600, do_warp: bool=False, warp_debug_path: Optional[str]=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Rescales to target_height; optional page-quad detection + perspective warp; Otsu INV threshold on warped gray.
    Returns: (warped_color, warped_gray, thresh)
    """
    scale = target_height / img.shape[0]
    image_resized = cv2.resize(img, (int(img.shape[1]*scale), target_height), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5,5), 0)
    edged = cv2.Canny(gray_blur, 50, 150)

    page_contour = None
    if do_warp:
        cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            img_area = image_resized.shape[0] * image_resized.shape[1]
            best = None; best_area = 0
            for c in cnts:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02*peri, True)
                if len(approx) != 4:
                    continue
                area = cv2.contourArea(approx)
                if area < 0.5*img_area:
                    continue
                (w,h) = cv2.minAreaRect(approx)[1]
                if w == 0 or h == 0:
                    continue
                ar = max(w,h)/max(1.0, min(w,h))
                if ar > 1.8:
                    continue
                if area > best_area:
                    best = approx.reshape(4,2); best_area = area
            if best is not None:
                page_contour = best

    if page_contour is not None:
        warped = _four_point_transform(image_resized, page_contour)
        warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    else:
        warped = image_resized; warped_gray = gray

    if warp_debug_path is not None:
        dbg = image_resized.copy()
        if page_contour is not None:
            cv2.polylines(dbg, [page_contour.astype(int)], True, (0,255,0), 3)
        else:
            cv2.putText(dbg, "NO QUAD—SKIPPING WARP", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
        cv2.imwrite(warp_debug_path, dbg)

    blur = cv2.GaussianBlur(warped_gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return warped, warped_gray, thresh

def _order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1); rect[0] = pts[np.argmin(s)]; rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1); rect[1] = pts[np.argmin(diff)]; rect[3] = pts[np.argmax(diff)]
    return rect

def zone_to_pixels(zone, img_shape):
    """
    Accept either pixel rect [x0,y0,x1,y1] or normalized floats in [0,1],
    and return integer pixel rect (x0,y0,x1,y1) for the given image shape.
    """
    x0, y0, x1, y1 = zone
    H, W = img_shape[:2]
    # If all look like normalized floats, scale to pixels
    if 0.0 <= x0 <= 1.0 and 0.0 <= y0 <= 1.0 and 0.0 <= x1 <= 1.0 and 0.0 <= y1 <= 1.0:
        x0 = int(round(x0 * W)); x1 = int(round(x1 * W))
        y0 = int(round(y0 * H)); y1 = int(round(y1 * H))
    return (int(x0), int(y0), int(x1), int(y1))

def _four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect = _order_points(pts.astype("float32"))
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl); widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))
    heightA = np.linalg.norm(tr - br); heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))
    dst = np.array([[0,0],[maxWidth-1,0],[maxWidth-1,maxHeight-1],[0,maxHeight-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def measure_fill_ratio(thresh_img: np.ndarray, rect: Tuple[int,int,int,int], shape: str="circle") -> float:
    """
    Fraction of white (255) pixels within rect/circle in a binary-inverted image (white=ink).
    """
    roi = subimage(thresh_img, rect)
    if roi.size == 0:
        return 0.0
    if shape == "circle":
        H, W = roi.shape[:2]
        cx = W//2; cy = H//2; r = int(min(W,H)*0.48)
        yy, xx = np.ogrid[:H, :W]
        mask = (xx-cx)**2 + (yy-cy)**2 <= r*r
        filled = np.count_nonzero((roi > 0) & mask)
        total  = np.count_nonzero(mask)
    else:
        filled = cv2.countNonZero(roi)
        total  = roi.size
    return float(filled) / max(1.0, float(total))

# ---------- Config data ----------

@dataclass
class GridSpec:
    zone: Tuple[int,int,int,int]
    rows: int
    cols: int
    pad: float = 0.12
    shape: str = "circle"
    symbols: Optional[List[str]] = None
    positions: Optional[int] = None         # NEW: for old schema convenience
    symbols_axis: str = "rows"              # NEW: "rows" (default) or "cols"

@dataclass
class Config:
    answer_layout: GridSpec
    answer_layouts: Optional[List[GridSpec]]
    last_name_layout: Optional[GridSpec] = None
    first_name_layout: Optional[GridSpec] = None
    name_layout: Optional[GridSpec] = None
    id_layout: Optional[GridSpec] = None
    version_layout: Optional[GridSpec] = None

def load_config(path: str) -> Config:
    import math
    with open(path, "r") as f:
        J = json.load(f)

    default_shape = str(J.get("bubble_shape", "circle"))
    default_pad   = float(J.get("default_pad", 0.12))

    def gs_from_block(k: dict, fallback_shape: str, default_pad: float) -> GridSpec:
        zone  = tuple(k["zone"])
        pad   = float(k.get("pad", default_pad))
        shape = str(k.get("shape", fallback_shape))

        # v6-style explicit
        rows    = k.get("rows")
        cols    = k.get("cols")
        symbols = k.get("symbols")

        # old-style fields
        positions   = k.get("positions")
        alphabet    = k.get("alphabet")
        digits      = k.get("digits")
        sym_src     = alphabet if alphabet is not None else digits
        orientation = str(k.get("orientation", "vertical")).lower()

        # Old schema: positions + alphabet/digits
        if (rows is None or cols is None) and (positions is not None) and (sym_src is not None):
            if isinstance(sym_src, list):
                sym_list = [str(s) for s in sym_src]
            else:
                sym_list = [ch for ch in str(sym_src)]

            if orientation in ("horizontal", "h", "row"):
                # symbols laid out across columns
                rows = int(positions)
                cols = len(sym_list)
                symbols_axis = "cols"
            else:
                # symbols stacked vertically (default)
                rows = len(sym_list)
                cols = int(positions)
                symbols_axis = "rows"

            return GridSpec(
                zone=zone, rows=int(rows), cols=int(cols),
                pad=pad, shape=shape, symbols=sym_list,
                positions=int(positions), symbols_axis=symbols_axis
            )

        # v6 explicit: require rows/cols
        if rows is None or cols is None:
            raise ValueError("Grid must define rows/cols or positions+alphabet/digits")

        return GridSpec(
            zone=zone, rows=int(rows), cols=int(cols),
            pad=pad, shape=shape, symbols=symbols,
            positions=positions, symbols_axis="rows"
        )

    # answers: support single "answer_layout" or list "answer_layouts"
    answer_layouts: List[GridSpec] = []
    answer_layout: Optional[GridSpec] = None

    if "answer_layouts" in J and J["answer_layouts"]:
        for blk in J["answer_layouts"]:
            # old schema: questions/choices/questions_per_row → derive rows/cols
            if "rows" not in blk or "cols" not in blk:
                Q   = int(blk.get("questions", 0))
                C   = int(blk.get("choices", 0))
                qpr = int(blk.get("questions_per_row", 1))
                if not (Q and C):
                    raise ValueError("answer_layouts block missing questions/choices")
                rows = Q if qpr == 1 else math.ceil(Q / qpr)
                cols = C if qpr == 1 else C * qpr
                blk  = {**blk, "rows": rows, "cols": cols}
            answer_layouts.append(gs_from_block(blk, default_shape, default_pad))
    elif "answer_layout" in J:
        answer_layout = gs_from_block(J["answer_layout"], default_shape, default_pad)
    else:
        raise ValueError("Config must include answer_layout or answer_layouts[]")

    def gs(name: str) -> Optional[GridSpec]:
        if name not in J or J[name] is None: return None
        return gs_from_block(J[name], default_shape, default_pad)

    return Config(
        answer_layout=answer_layout,
        answer_layouts=answer_layouts if answer_layouts else None,
        last_name_layout=gs("last_name_layout"),
        first_name_layout=gs("first_name_layout"),
        name_layout=gs("name_layout"),
        id_layout=gs("id_layout"),
        version_layout=gs("version_layout"),
    )

# ---------- Decoders ----------

def decode_grid_symbols(thresh: np.ndarray,
                        layout: GridSpec,
                        min_fill: float,
                        annotate: bool = False,
                        label_density: bool = False,
                        vis: Optional[np.ndarray] = None,
                        # NEW: apply same strictness as answers
                        min_score: float = 2.0,
                        top2_ratio: float = 0.75,
                        min_abs: float = 0.10) -> str:
    """
    Decode a name/ID/version grid with stricter rules:
      pick only if (top >= min_fill) AND (top >= min_abs) AND
                   ( (top - second)*100 >= min_score  OR  second <= top*top2_ratio )

    Honors layout.symbols_axis == "rows" (vertical symbols) or "cols" (horizontal symbols).
    Annotates green at the winning bubble (if any), gray for others; optional density labels.
    """
    zone_px = zone_to_pixels(layout.zone, thresh.shape)
    rects = cell_rects(zone_px, layout.rows, layout.cols, layout.pad)

    # Default symbols
    if not layout.symbols:
        if layout.symbols_axis == "cols":
            layout.symbols = [chr(ord('A') + i) for i in range(layout.cols)]
        else:
            layout.symbols = [chr(ord('A') + i) for i in range(layout.rows)]

    out_chars = []

    def passes_strict(top: float, second: float) -> bool:
        if top < max(min_fill, min_abs):
            return False
        sep_score = (top - second) * 100.0
        sep_ratio_ok = (second <= top * max(0.0, min(1.0, top2_ratio)))
        return (sep_score >= min_score) or sep_ratio_ok

    if layout.symbols_axis == "rows":
        # columns are positions; rows are choices (vertical stack)
        for c in range(layout.cols):
            fills = []
            for r in range(layout.rows):
                rect = rects[r * layout.cols + c]
                fills.append(measure_fill_ratio(thresh, rect, layout.shape))
            order = np.argsort(fills)[::-1] if fills else []
            best_r = int(order[0]) if len(order) > 0 else -1
            top = fills[best_r] if best_r >= 0 else 0.0
            second = fills[order[1]] if len(order) > 1 else 0.0

            chosen = layout.symbols[best_r] if (best_r >= 0 and passes_strict(top, second)) else ""
            out_chars.append(chosen)

            if annotate and vis is not None:
                for r in range(layout.rows):
                    rect = rects[r * layout.cols + c]
                    color = (0,200,0) if (r == best_r and chosen) else (64,64,64)
                    draw_circle_annotation(vis, rect, color, 2)
                    if label_density:
                        put_center_text(vis, rect, f"{int(round(fills[r]*100))}", 0.45, (255,0,0))

    else:
        # rows are positions; columns are choices (horizontal symbols)
        for r in range(layout.rows):
            fills = []
            for c in range(layout.cols):
                rect = rects[r * layout.cols + c]
                fills.append(measure_fill_ratio(thresh, rect, layout.shape))
            order = np.argsort(fills)[::-1] if fills else []
            best_c = int(order[0]) if len(order) > 0 else -1
            top = fills[best_c] if best_c >= 0 else 0.0
            second = fills[order[1]] if len(order) > 1 else 0.0

            chosen = layout.symbols[best_c] if (best_c >= 0 and passes_strict(top, second)) else ""
            out_chars.append(chosen)

            if annotate and vis is not None:
                for c in range(layout.cols):
                    rect = rects[r * layout.cols + c]
                    color = (0,200,0) if (c == best_c and chosen) else (64,64,64)
                    draw_circle_annotation(vis, rect, color, 2)
                    if label_density:
                        put_center_text(vis, rect, f"{int(round(fills[c]*100))}", 0.45, (255,0,0))

    return "".join(out_chars).strip()

def collect_answer_cells(thresh: np.ndarray, layouts: List[GridSpec]) -> Tuple[List[Tuple[int,int,int,int]], int]:
    """
    Returns (all_rects, choices_per_question).
    Concatenates cells from each answer layout in order, assuming each layout
    has the same number of choices (cols). Zones may be disjoint.
    """
    all_rects: List[Tuple[int,int,int,int]] = []
    cols_ref = None
    for lay in layouts:
        zone_px = zone_to_pixels(lay.zone, thresh.shape)
        rects = cell_rects(zone_px, lay.rows, lay.cols, lay.pad)
        if cols_ref is None:
            cols_ref = lay.cols
        else:
            if lay.cols != cols_ref:
                raise ValueError(f"answer_layouts cols mismatch: {cols_ref} vs {lay.cols}")
        all_rects.extend(rects)
    if cols_ref is None:
        raise ValueError("No answer layouts provided")
    return all_rects, cols_ref

def decode_answers(thresh: np.ndarray,
                   cfg: Config,
                   total_questions: int,
                   min_fill: float, min_score: float, top2_ratio: float,
                   annotate: bool, label_density: bool, vis: Optional[np.ndarray],
                   key_map: Optional[Dict[int, str]]=None
                   ) -> Tuple[List[Optional[str]], List[float], List[List[float]]]:
    """
    Supports single answer_layout or multiple answer_layouts concatenated.
    """
    if cfg.answer_layouts:
        rects_all, cols = collect_answer_cells(thresh, cfg.answer_layouts)
        shape = cfg.answer_layouts[0].shape
    else:
        lay = cfg.answer_layout
        zone_px = zone_to_pixels(lay.zone, thresh.shape)
        rects_all = cell_rects(zone_px, lay.rows, lay.cols, lay.pad)
        cols = lay.cols
        shape = lay.shape  # <-- fixed

    max_questions_available = len(rects_all) // cols
    Q = min(total_questions, max_questions_available)  # cap to avoid IndexError

    choice_labels = [chr(ord('A')+i) for i in range(cols)]
    chosen: List[Optional[str]] = []
    topfills: List[float] = []
    rawfills: List[List[float]] = []

    for q in range(Q):
        fills = []
        row_base = q * cols
        for c in range(cols):
            rect = rects_all[row_base + c]
            fr = measure_fill_ratio(thresh, rect, shape)
            fills.append(fr)

        rawfills.append(fills)
        order = np.argsort(fills)[::-1]
        top = fills[order[0]]
        second = fills[order[1]] if cols >= 2 else 0.0

        pick = None
        if top >= min_fill:
            sep_score = (top - second) * 100.0
            sep_ratio_ok = (second <= top * max(0.0, min(1.0, top2_ratio)))
            if (sep_score >= min_score) or sep_ratio_ok:
                pick = choice_labels[order[0]]

        chosen.append(pick)
        topfills.append(top)

        if annotate and vis is not None:
            for c in range(cols):
                rect = rects_all[row_base + c]
                color = (64, 64, 64)
                if pick is not None and c == order[0]:
                    if key_map and (q+1) in key_map:
                        correct = key_map[q+1]
                        color = (0,200,0) if choice_labels[c] == correct else (0,0,255)
                    else:
                        color = (0,200,0)
                draw_circle_annotation(vis, rect, color, 2)
                if label_density:
                    pct = int(round(fills[c]*100))
                    put_center_text(vis, rect, f"{pct}", 0.45, (255,255,255))

    return chosen, topfills, rawfills
    
# ---------- CSV ----------

def make_csv_header(total_questions: int) -> List[str]:
    base = [
        "sheet","name","last_name","first_name","id","version",
        "total","correct","wrong","blank","multi","percent"
    ]
    for i in range(1, total_questions+1):
        base.append(f"Q{i}")
    return base

def write_key_row(writer, key_answers: Dict[int,str], total_questions: int):
    row = {
        "sheet":"KEY","name":"","last_name":"","first_name":"","id":"","version":"",
        "total": total_questions, "correct":"","wrong":"","blank":"","multi":"","percent":""
    }
    for i in range(1, total_questions+1):
        row[f"Q{i}"] = key_answers.get(i, "")
    writer.writerow(row)

# ---------- Main ----------

def parse_args():
    ap = argparse.ArgumentParser(description="Merged bubble sheet reader with v5 pipeline + enhanced CSV.")
    ap.add_argument("input", help="Path to aligned_scans.(pdf|png|jpg)")
    ap.add_argument("--config", required=True, help="JSON config defining zones/layouts")
    ap.add_argument("--key-txt", required=True, help="Text file with the answer key (e.g., A B C ... or one per line)")
    ap.add_argument("--total-questions", type=int, required=True, help="Number of questions to score (caps the grid)")
    ap.add_argument("--out-csv", default="results.csv", help="Output CSV path")
    ap.add_argument("--out-annotated-dir", default=None, help="Folder to write annotated page PNGs")
    ap.add_argument("--annotate-all-cells", action="store_true", help="Annotate bubbles even when not selected")
    ap.add_argument("--label-density", action="store_true", help="Overlay fill%% labels on each bubble")
    # Sensitivity knobs:
    ap.add_argument("--min-fill", type=float, default=0.40, help="Min fill (0..1) to consider an ANSWER bubble filled")
    ap.add_argument("--name-min-fill", type=float, default=0.15, help="Min fill (0..1) for NAME/ID bubbles, to reject faint marks or printed letters. Try tuning between 0.12-0.25.")
    ap.add_argument("--min-score", type=float, default=2.0, help="Min (top-second)*100 separation")
    ap.add_argument("--top2-ratio", type=float, default=0.75, help="Require second <= top*ratio (0..1)")
    # Warp/debug:
    ap.add_argument("--fix-warp", action="store_true", help="Attempt page quad detection and perspective warp")
    ap.add_argument("--warp-debug", action="store_true", help="Write a warp debug PNG per page")
    return ap.parse_args()

def load_key_map(path: str, total_questions: int) -> Dict[int, str]:
    """
    Accepts either space-delimited string on one line, or one char per line.
    Supports A-D (or beyond) letters. Returns 1-based dict: {1:'A', 2:'C', ...}
    """
    with open(path, "r") as f:
        content = [tok.strip() for tok in f.read().replace("\r\n","\n").split()]
    if len(content) == 1 and len(content[0]) > 1:
        # Single token like "ABCD..." -> expand
        arr = list(content[0].strip())
    else:
        arr = content
    key = {}
    for i, ch in enumerate(arr[:total_questions], start=1):
        key[i] = ch.upper()
    return key

def main():
    args = parse_args()
    cfg = load_config(args.config)
    key_map = load_key_map(args.key_txt, args.total_questions)

    images = imread_any(args.input)
    ensure_dir(args.out_annotated_dir or "")
    do_warp = bool(args.fix_warp)

    # Prepare CSV
    header = make_csv_header(args.total_questions)
    out_csv_fp = open(args.out_csv, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(out_csv_fp, fieldnames=header)
    writer.writeheader()
    write_key_row(writer, key_map, args.total_questions)

    for page_idx, img in enumerate(images, start=1):
        warp_dbg_path = None
        if args.warp_debug and args.out_annotated_dir:
            warp_dbg_path = os.path.join(args.out_annotated_dir, f"page{page_idx:03d}_warpdebug.png")
        warped, gray, thresh = load_and_warp(img, target_height=1600, do_warp=do_warp, warp_debug_path=warp_dbg_path)

        # For annotations, operate on a color copy of warped
        vis = warped.copy() if args.out_annotated_dir else None

        # Decode name/id/version blocks if present
        def get_text(layout: Optional[GridSpec]) -> str:
            if layout is None:
                return ""
            return decode_grid_symbols(
                thresh, layout, args.name_min_fill,
                annotate=bool(args.annotate_all_cells),
                label_density=bool(args.label_density),
                vis=vis,
                # reuse the same knobs as answers
                min_score=args.min_score,
                top2_ratio=args.top2_ratio,
                min_abs=0.0  # Use name_min_fill as the primary threshold now
            )
            
        last_name = get_text(cfg.last_name_layout)
        first_name = get_text(cfg.first_name_layout)
        name_full = get_text(cfg.name_layout)
        student_id = get_text(cfg.id_layout)
        version = get_text(cfg.version_layout)

        # Answers:
        chosen, topfills, rawfills = decode_answers(
            thresh, cfg, args.total_questions,
            args.min_fill, args.min_score, args.top2_ratio,
            annotate=args.annotate_all_cells,
            label_density=bool(args.label_density),
            vis=vis if args.out_annotated_dir else None,
            key_map=key_map
        )

        # Score vs key:
        correct = 0; wrong = 0; blank = 0; multi = 0
        qvals = {}
        for i, pick in enumerate(chosen, start=1):
            k = key_map.get(i, "")
            if pick is None or pick == "":
                blank += 1
                qvals[f"Q{i}"] = ""
            else:
                if k and pick == k:
                    correct += 1
                    qvals[f"Q{i}"] = pick
                else:
                    wrong += 1
                    qvals[f"Q{i}"] = pick
        total = args.total_questions
        multi = 0  # reserved for future multi-mark detection

        percent = round((100.0 * correct / total), 2) if total > 0 else 0.0

        row = {
            "sheet": f"p{page_idx}",
            "name": name_full,
            "last_name": last_name,
            "first_name": first_name,
            "id": student_id,
            "version": version,
            "total": total,
            "correct": correct,
            "wrong": wrong,
            "blank": blank,
            "multi": multi,
            "percent": percent
        }
        for i in range(1, args.total_questions+1):
            row[f"Q{i}"] = qvals.get(f"Q{i}", "")

        writer.writerow(row)

        # Save annotations if requested
        if args.out_annotated_dir:
            out_png = os.path.join(args.out_annotated_dir, f"aligned_scans_p{page_idx}_annotated.png")
            cv2.imwrite(out_png, vis)

    out_csv_fp.close()
    print(f"[DONE] Wrote CSV: {args.out_csv}")

# ---------- Entry ----------

if __name__ == "__main__":
    main()

