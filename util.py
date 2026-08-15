"""
util.py
=======
Tien ich: load YOLOv8, ve label len anh, ghi CSV, resolve path.
"""
from __future__ import annotations
import csv
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
from paddle_ocr_engine import format_plate_display


# ────────────────────────────────────────────────────────────
# PATH HELPERS
# ────────────────────────────────────────────────────────────
def resolve_path(base_dir: Path, raw_path: str | Path) -> Path:
    """Tra ve duong dan tuyet doi, neu raw_path la relative thi noi voi base."""
    p = Path(raw_path)
    return p if p.is_absolute() else (base_dir / p).resolve()


# ────────────────────────────────────────────────────────────
# YOLO PLATE DETECTOR
# ────────────────────────────────────────────────────────────
def load_yolo_model(model_path: Path, label: str = "YOLO model") -> YOLO:
    """Load YOLOv8 weights (.pt)."""
    if not model_path.exists():
        raise FileNotFoundError(f"{label} not found: {model_path}")
    print(f"[OK] {label}: {model_path.name}")
    return YOLO(str(model_path))


def clamp_xyxy(frame: np.ndarray, bbox) -> tuple[int, int, int, int]:
    """Gioi han toa do trong frame, ep ve int."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = map(int, bbox[:4])
    return (max(0, min(x1, w - 1)),
            max(0, min(y1, h - 1)),
            max(0, min(x2, w)),
            max(0, min(y2, h)))


def detect_plates(model: YOLO, frame: np.ndarray, min_conf: float = 0.25) -> list[list[float]]:
    """
    Chay YOLO detect bien so.
    Tra ve danh sach [x1, y1, x2, y2, conf].
    """
    if model is None:
        return []
    r = model(frame, verbose=False)[0]
    if r.boxes is None:
        return []
    out = []
    for d in r.boxes.data.tolist():
        if len(d) >= 5 and d[4] >= min_conf:
            out.append([float(d[0]), float(d[1]), float(d[2]), float(d[3]), float(d[4])])
    return out


# ────────────────────────────────────────────────────────────
# VE LABEL LEN ANH
# ────────────────────────────────────────────────────────────
def _fit_text(line: str, font, scale_init: float, max_w: int, thickness: int = 2) -> tuple[float, tuple[int, int]]:
    """Tu dong giam font neu chu rong qua bbox."""
    sc = scale_init
    while sc > 0.35:
        (tw, th), _ = cv2.getTextSize(line, font, sc, thickness)
        if tw + 10 <= max_w:
            break
        sc -= 0.05
    return sc, cv2.getTextSize(line, font, sc, thickness)[0]


def draw_plate_label(frame: np.ndarray,
                     bbox: tuple[int, int, int, int],
                     text: str,
                     text_score: float,
                     color_label: str,
                     color_score: float,
                     purpose: str,
                     box_color: tuple[int, int, int]):
    """Ve bbox + 2 dong label (text + color/purpose) len frame."""
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)

    # 2 dong label — dung format_plate_display de chen dau '-' dung vi tri
    formatted = format_plate_display(text)
    plate_disp = f"{formatted} ({text_score:.0%})"
    lines = [
        (plate_disp,                                                 0.65),
        (f"{color_label} ({color_score:.0%}) - {purpose}",           0.55),
    ]

    max_label_w = W - max(0, x1) - 4
    fitted = [_fit_text(l, font, sc, max_label_w) for l, sc in lines]
    sizes  = [sz for _, sz in fitted]
    scales = [sc for sc, _ in fitted]
    pad    = 6
    total_h = sum(th + pad * 2 for _, th in sizes)

    # Ve duoi bbox neu con cho, nguoc lai ve tren
    if y2 + total_h + 4 <= H:
        cur_y = y2 + 4
        for (line, _), sc, (tw, th) in zip(lines, scales, sizes):
            bx1 = max(0, x1)
            bx2 = min(W - 2, bx1 + tw + 10)
            cv2.rectangle(frame, (bx1, cur_y), (bx2, cur_y + th + pad * 2), box_color, -1)
            cv2.putText(frame, line, (bx1 + 5, cur_y + th + pad - 2),
                        font, sc, (0, 0, 0), 2, cv2.LINE_AA)
            cur_y += th + pad * 2 + 2
    else:
        cur_y = y1 - 4
        for (line, _), sc, (tw, th) in reversed(list(zip(lines, scales, sizes))):
            block_h = th + pad * 2
            by_top  = max(0, cur_y - block_h)
            bx1     = max(0, x1)
            bx2     = min(W - 2, bx1 + tw + 10)
            cv2.rectangle(frame, (bx1, by_top), (bx2, by_top + block_h), box_color, -1)
            cv2.putText(frame, line, (bx1 + 5, by_top + th + pad - 2),
                        font, sc, (0, 0, 0), 2, cv2.LINE_AA)
            cur_y = by_top - 2


# ────────────────────────────────────────────────────────────
# OUTPUT
# ────────────────────────────────────────────────────────────
def write_csv(results: dict, output_path: str | Path):
    """Ghi ket qua ra CSV."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame", "plate_text", "text_score", "color", "purpose", "bbox_score"])
        for fnr in sorted(results):
            for cid in sorted(results[fnr]):
                lp = results[fnr][cid].get("license_plate", {})
                w.writerow([
                    fnr,
                    lp.get("text", ""),
                    f"{lp.get('text_score', 0):.4f}",
                    lp.get("color", ""),
                    lp.get("purpose", ""),
                    f"{lp.get('bbox_score', 0):.4f}",
                ])
