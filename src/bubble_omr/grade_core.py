from __future__ import annotations

from typing import Optional, Dict, Union
import os, csv, cv2
from .tools.bubble_score import (
    load_config,            # <-- use this; returns a Config
    Config,
    load_key_map, imread_any, ensure_dir, make_csv_header,
    write_key_row, load_and_warp, decode_grid_symbols, decode_answers
)


# Config + tools
from .tools.bubble_score import (
    load_config,      # ← use this; handles .yaml/.yml/.json
    Config,
    load_key_map,
    imread_any,
    ensure_dir,
    make_csv_header,
    write_key_row,
    load_and_warp,
    decode_grid_symbols,
    decode_answers,
)

def grade_pdf(
    input_path: str,
    config_or_path: Union[str, os.PathLike, Config],   # <— renamed & widened
    key_txt: str,
    total_questions: int,
    out_csv: str = "results.csv",
    out_annotated_dir: Optional[str] = None,
    annotate_all_cells: bool = False,
    mark_blanks_incorrect: bool = False,
    label_density: bool = False,
    min_fill: float = 0.40,
    name_min_fill: float = 0.15,
    min_score: float = 2.0,
    top2_ratio: float = 0.75,
    fix_warp: bool = False,
    warp_debug: bool = False,
) -> str:
    # --- normalize config to a Config object ---
    if isinstance(config_or_path, Config):
        cfg = config_or_path
    elif isinstance(config_or_path, (str, os.PathLike)):
        cfg = load_config(str(config_or_path))  # YAML/JSON file -> Config
    else:
        raise TypeError(f"config_or_path must be a path or Config, not {type(config_or_path)}")

    key_map: Dict[int, str] = load_key_map(key_txt, total_questions)

    images = imread_any(input_path)
    if out_annotated_dir:
        ensure_dir(out_annotated_dir)

    do_warp = bool(fix_warp)
    header = make_csv_header(total_questions)

    with open(out_csv, "w", newline="", encoding="utf-8") as out_csv_fp:
        writer = csv.DictWriter(out_csv_fp, fieldnames=header)
        writer.writeheader()
        write_key_row(writer, key_map, total_questions)

        for page_idx, img in enumerate(images, start=1):
            warp_dbg_path = (
                os.path.join(out_annotated_dir, f"page{page_idx:03d}_warpdebug.png")
                if (warp_debug and out_annotated_dir)
                else None
            )

            warped, gray, thresh = load_and_warp(
                img, target_height=1600, do_warp=do_warp, warp_debug_path=warp_dbg_path
            )
            vis = warped.copy() if out_annotated_dir else None

            def get_text(layout):
                if layout is None:
                    return ""
                return decode_grid_symbols(
                    thresh,
                    layout,
                    name_min_fill,
                    annotate=bool(annotate_all_cells),
                    label_density=bool(label_density),
                    vis=vis,
                )

            last_name  = get_text(getattr(cfg, "last_name_layout", None))
            first_name = get_text(getattr(cfg, "first_name_layout", None))
            name_full_layout = getattr(cfg, "name_layout", None)
            if name_full_layout is not None:
                name_full = get_text(name_full_layout)
            else:
                name_full = (last_name + " " + first_name).strip()

            student_id = get_text(getattr(cfg, "id_layout", None))
            version    = get_text(getattr(cfg, "version_layout", None))

            correct = wrong = blank = 0
            qvals: Dict[str, str] = {}

            picks = decode_answers(
                thresh,
                cfg,  # Config object
                total_questions,
                min_fill=min_fill,
                min_score=min_score,
                top2_ratio=top2_ratio,
                annotate=bool(annotate_all_cells),
                label_density=bool(label_density),
                vis=vis,
                key_map=key_map,
            )

            # normalize picks into (q_index, answer_str)
            def _iter_q_answer(p, n_total: int):
                if isinstance(p, dict):
                    for k, v in p.items():
                        yield int(k), ("" if v is None else str(v))
                    return
                if isinstance(p, tuple) and len(p) >= 1 and isinstance(p[0], (list, tuple)):
                    answers = p[0]
                    for i, v in enumerate(answers, start=1):
                        yield i, ("" if v is None else str(v))
                    return
                if isinstance(p, (list, tuple)):
                    for i, v in enumerate(p, start=1):
                        yield i, ("" if v is None else str(v))
                    return
                try:
                    seq = list(p)
                    for i, v in enumerate(seq, start=1):
                        yield i, ("" if v is None else str(v))
                except TypeError:
                    raise TypeError(f"Unexpected return shape from decode_answers: {type(p)}")

            for i, pick in _iter_q_answer(picks, total_questions):
                k = key_map.get(int(i), "")
                if pick == "":
                    if mark_blanks_incorrect:
                        blank += 1
                    qvals[f"Q{int(i)}"] = ""
                else:
                    if k and pick == str(k):
                        correct += 1
                    else:
                        wrong += 1
                    qvals[f"Q{int(i)}"] = pick

            total = total_questions
            multi = 0
            percent = round((100.0 * correct / total), 2) if total > 0 else 0.0

            row = {
                "sheet":   f"p{page_idx}",
                "name":    name_full,
                "last_name":  last_name,
                "first_name": first_name,
                "id":      student_id,
                "version": version,
                "total":   total,
                "correct": correct,
                "wrong":   wrong,
                "blank":   blank,
                "multi":   multi,
                "percent": percent,
            }
            for i in range(1, total_questions + 1):
                row[f"Q{i}"] = qvals.get(f"Q{i}", "")
            writer.writerow(row)

            if out_annotated_dir and vis is not None:
                out_png = os.path.join(out_annotated_dir, f"aligned_scans_p{page_idx}_annotated.png")
                cv2.imwrite(out_png, vis)

    return out_csv