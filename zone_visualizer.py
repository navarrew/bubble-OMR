#!/usr/bin/env python3
"""
zone_visualizer_warp.py
- Visualize configured zones and per-bubble cells (rects or circles) on a page.
- Supports images and PDFs; optional perspective warp detection.
- Respects "bubble_shape" from config or CLI --bubble-shape.
"""

import argparse, json, os
import cv2, numpy as np, fitz

# ---------- geometry / warp ----------
def _order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def _four_point_transform(image, pts):
    rect = _order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))
    dst = np.array([[0,0],[maxWidth-1,0],[maxWidth-1,maxHeight-1],[0,maxHeight-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight))

def load_and_warp(path, target_height=1600, do_warp=True, warp_debug_path=None):
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    scale = target_height / image.shape[0]
    image_resized = cv2.resize(image, (int(image.shape[1]*scale), target_height))

    gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5,5), 0)
    edged = cv2.Canny(gray_blur, 50, 150)

    page_contour = None
    if do_warp:
        cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            img_area = image_resized.shape[0]*image_resized.shape[1]
            best = None; best_area = 0
            for c in cnts:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02*peri, True)
                if len(approx) != 4: continue
                area = cv2.contourArea(approx)
                if area < 0.5*img_area: continue
                (w,h) = cv2.minAreaRect(approx)[1]
                if w==0 or h==0: continue
                ar = max(w,h)/max(1.0, min(w,h))
                if ar > 1.8: continue
                if area > best_area:
                    best = approx.reshape(4,2); best_area = area
            if best is not None:
                page_contour = best

    if page_contour is not None:
        warped = _four_point_transform(image_resized, page_contour)
    else:
        warped = image_resized

    if warp_debug_path is not None:
        dbg = image_resized.copy()
        if page_contour is not None:
            cv2.polylines(dbg, [page_contour.astype(int)], True, (0,255,0), 3)
        else:
            cv2.putText(dbg, "NO QUAD—SKIPPING WARP", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
        cv2.imwrite(warp_debug_path, dbg)

    return warped

def _is_pdf(path: str) -> bool: return path.lower().endswith(".pdf")

def _render_pdf_page_to_image(pdf_path: str, page_index: int, dpi: int = 300):
    doc = fitz.open(pdf_path)
    if page_index < 0 or page_index >= len(doc):
        raise IndexError(f"PDF page index out of range: {page_index}")
    page = doc[page_index]
    zoom = dpi / 72.0; mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR); doc.close(); return img

def zone_rect(img, norm_rect):
    h, w = img.shape[:2]
    x0 = int(norm_rect[0]*w); y0 = int(norm_rect[1]*h)
    x1 = int(norm_rect[2]*w); y1 = int(norm_rect[3]*h)
    return x0, y0, x1, y1

def cell_rects(zone_rect_abs, rows, cols, pad=0.1):
    x0,y0,x1,y1 = zone_rect_abs
    W = x1 - x0; H = y1 - y0
    cw = W / cols; ch = H / rows
    rects = []
    for r in range(rows):
        for c in range(cols):
            rx0 = int(x0 + c*cw + pad*cw)
            ry0 = int(y0 + r*ch + pad*ch)
            rx1 = int(x0 + (c+1)*cw - pad*cw)
            ry1 = int(y0 + (r+1)*ch - pad*ch)
            rects.append((rx0, ry0, rx1, ry1))
    return rects

def draw_zone_rect(img, rect, color=(0,255,0), thickness=2, label=None):
    x0,y0,x1,y1 = rect
    cv2.rectangle(img, (x0,y0), (x1,y1), color, thickness)
    if label:
        cv2.putText(img, label, (x0, max(0,y0-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def draw_cells(img, rects, shape="rect", color=(0,255,0)):
    for (x0,y0,x1,y1) in rects:
        if shape == "circle":
            cx, cy = (x0+x1)//2, (y0+y1)//2
            r = int(min((x1-x0), (y1-y0)) * 0.45)  # ~90% of half-min dimension
            cv2.circle(img, (cx,cy), r, color, 2)
        else:
            cv2.rectangle(img, (x0,y0), (x1,y1), color, 1)

def main():
    ap = argparse.ArgumentParser(description="Zone visualizer with circular bubble support")
    ap.add_argument("--config", required=True, help="Path to JSON config")
    ap.add_argument("--input", required=True, help="Input image/PDF for visualization")
    ap.add_argument("--page", type=int, default=1, help="PDF page (1-based)")
    ap.add_argument("--no-warp", action="store_true", help="Disable perspective warp detection")
    ap.add_argument("--bubble-shape", choices=["rect","circle"], default=None, help="Override bubble shape (default from config)")
    ap.add_argument("--out", default="zones_overlay.jpg", help="Output overlay image")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = json.load(f)

    shape = args.bubble_shape if args.bubble_shape is not None else cfg.get("bubble_shape","rect")

    # Load input
    if _is_pdf(args.input):
        img = _render_pdf_page_to_image(args.input, max(1,args.page)-1, dpi=300)
    else:
        img = cv2.imread(args.input)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {args.input}")

    # Warp
    if args.no_warp or _is_pdf(args.input):
        warped = img
    else:
        warped = load_and_warp(args.input, do_warp=True)

    vis = warped.copy()

    # Draw name zones
    for key, color in [("last_name_layout",(0,210,255)), ("first_name_layout",(0,170,255))]:
        z = cfg.get(key)
        if z:
            rect = zone_rect(warped, z["zone"])
            draw_zone_rect(vis, rect, color, 2, key.replace("_"," ").title())
            rows = len(z.get("alphabet","ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
            cols = z["positions"]
            cells = cell_rects(rect, rows, cols, pad=0.15)
            draw_cells(vis, cells, shape=shape, color=color)

    # Student ID
    z = cfg.get("id_layout")
    if z:
        rect = zone_rect(warped, z["zone"])
        draw_zone_rect(vis, rect, (255,180,0), 2, "ID")
        rows = len(z.get("digits","0123456789"))
        cols = z["positions"]
        cells = cell_rects(rect, rows, cols, pad=0.15)
        draw_cells(vis, cells, shape=shape, color=(255,180,0))

    # Version
    z = cfg.get("version_layout")
    if z:
        rect = zone_rect(warped, z["zone"])
        draw_zone_rect(vis, rect, (180,255,0), 2, "VERSION")
        symbols = z.get("symbols", z.get("digits","ABCD"))
        if z.get("orientation") == "horizontal":
            rows = 1; cols = len(symbols)
        else:
            rows = len(symbols); cols = z.get("positions",1)
        cells = cell_rects(rect, rows, cols, pad=0.15)
        draw_cells(vis, cells, shape=shape, color=(180,255,0))

    # Answer zones
    def _get_answer_zones(cfg):
        if "answer_layouts" in cfg and isinstance(cfg["answer_layouts"], list):
            return cfg["answer_layouts"]
        elif "answer_layout" in cfg and isinstance(cfg["answer_layout"], dict):
            return [cfg["answer_layout"]]
        else:
            return []

    for i, ans_layout in enumerate(_get_answer_zones(cfg), start=1):
        rect = zone_rect(warped, ans_layout["zone"])
        draw_zone_rect(vis, rect, (0,255,0), 2, f"ANS {i}")
        q = ans_layout["questions"]
        qpr = max(1, ans_layout["questions_per_row"])
        choices = ans_layout["choices"]
        rows = int(np.ceil(q / qpr))
        cols = qpr * choices
        cells = cell_rects(rect, rows, cols, pad=0.15)
        draw_cells(vis, cells, shape=shape, color=(0,255,0))

    cv2.imwrite(args.out, vis)
    print(f"[INFO] Wrote overlay to {args.out}")

if __name__ == "__main__":
    main()
