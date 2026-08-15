"""
paddle_ocr_engine.py
====================
Wrapper cho PaddleOCR — thay thế CharCNN cũ.

Pipeline:
  1. Tiền xử lý ảnh biển (deskew + CLAHE + scale up)
  2. PaddleOCR (PP-OCRv5 mobile, lang='en')
  3. Hậu xử lý: ghép dòng, lọc ký tự hợp lệ, sửa định dạng VN
  4. Fallback multi-scale nếu confidence thấp

API công khai:
  - PaddlePlateOCR().recognize(plate_bgr, plate_color=None) -> (text, score)
"""
from __future__ import annotations
import os
import re
import warnings

import cv2
import numpy as np

os.environ.setdefault("FLAGS_use_mkldnn", "false")
os.environ.setdefault("FLAGS_enable_pir_in_executor", "0")
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
# Ẩn log C++ (glog) của Paddle: 0=INFO ... 3=chỉ FATAL. Đặt 3 để bớt rườm rà.
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("GLOG_v", "0")
warnings.filterwarnings("ignore", category=UserWarning)

import logging  # noqa: E402

from paddleocr import PaddleOCR  # noqa: E402

# ── Tắt log "Creating model: ..." và "Model files already exist..." ──────────
# Các dòng này do logger của paddlex/ppocr in ra ở mức INFO. Nâng ngưỡng các
# logger liên quan lên ERROR (sau khi import xong) để chỉ còn hiện lỗi thật sự.
for _logger_name in ("ppocr", "paddle", "paddlex", "paddlex.inference", "paddleocr"):
    logging.getLogger(_logger_name).setLevel(logging.ERROR)


# ────────────────────────────────────────────────────────────
# CONSTANTS — định dạng biển số Việt Nam
# ────────────────────────────────────────────────────────────
_VALID_SERIES = set("ABCDEFGHKLMNPSTUVXYZ")

# Map nhầm chữ ↔ số (tại VỊ TRÍ SỐ)
_FIX_TO_DIGIT = {
    "O": "0", "Q": "0", "D": "0",
    "I": "1", "L": "1", "J": "1",
    "Z": "2", "A": "4",
    "S": "5", "G": "6", "C": "6",
    "T": "7", "Y": "7",
    "B": "8",
    "P": "9",
}
# Map nhầm số ↔ chữ (tại VỊ TRÍ CHỮ SERIES)
_FIX_TO_LETTER = {
    "0": "D", "1": "T", "2": "Z",
    "3": "B", "4": "A", "5": "S",
    "6": "G", "7": "T", "8": "B", "9": "P",
}

# Biển nền tối (Xanh) thường nhầm B → E/H/F
_DARK_BG_FIX = {"E": "B", "H": "B", "F": "B"}
# CHỈ áp dụng dark bg fix cho Xanh, KHÔNG áp cho Đỏ (quân đội)
_DARK_BG_COLORS = {"Xanh", "xanh", "XANH"}
# Đỏ vẫn cần invert màu để OCR nhưng KHÔNG fix format như dân sự
_INVERT_COLORS = {"Xanh", "Do", "xanh", "do", "XANH", "DO"}

# Regex biển dân sự/kinh doanh: 2 số tỉnh + 1-2 chữ series + 4-5 số
# Biển dân sự: 2 số tỉnh + 1-2 chữ series + 4-6 số (6 số cho biển đặc biệt vd 29G1-333.33)
_PLATE_RE = re.compile(r"^\d{2}[A-Z]{1,2}\d{4,6}$")

# Regex biển quân đội: 2 chữ + 2 số + 2 số (vd: PK5346, TM2366, KV6938, BQ...)
# Format thực: XX-NN-NN (6 ký tự alphanumeric, 2 chữ đầu + 4 số)
_MILITARY_RE = re.compile(r"^[A-Z]{2}\d{4}$")

# Prefix chữ cái đặc trưng biển quân đội VN
_MILITARY_PREFIXES = {
    "BQ", "PK", "TM", "KV", "QS", "QP", "HC", "HQ", "KQ",
    "BB", "BD", "BH", "BT", "CM", "CT", "CZ", "DN", "DT",
    "LC", "LD", "LH", "LT", "LX", "MB", "MC", "MH", "ML",
    "MT", "NB", "ND", "NL", "NT", "PB", "PC", "PD", "PH",
    "PM", "PS", "PT", "PV", "QA", "QB", "QC", "QD", "QG",
    "QH", "QK", "QL", "QN", "QT", "QV", "SB", "SC", "SD",
    "SH", "SK", "SL", "SM", "SN", "SP", "ST", "SV", "TA",
    "TB", "TC", "TD", "TG", "TH", "TK", "TL", "TN", "TP",
    "TQ", "TR", "TS", "TT", "TV", "TX", "VH", "VK", "VN",
    "VT", "VX", "XD",
}


def _is_military_plate(text: str) -> bool:
    """Kiểm tra có phải biển quân đội không (2 chữ + 4 số)."""
    if len(text) != 6:
        return False
    return bool(_MILITARY_RE.match(text)) and text[:2] in _MILITARY_PREFIXES


# ────────────────────────────────────────────────────────────
# TIỀN XỬ LÝ ẢNH BIỂN
# ────────────────────────────────────────────────────────────
def _deskew(img_bgr: np.ndarray) -> np.ndarray:
    """Sửa biển bị nghiêng nhẹ (<25°) bằng horizontal projection."""
    h, w = img_bgr.shape[:2]
    if h < 15 or w < 15:
        return img_bgr
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    best_var, best_angle = 0.0, 0.0
    for a in np.arange(-18, 18.5, 1.5):
        M = cv2.getRotationMatrix2D((w / 2, h / 2), a, 1.0)
        rot = cv2.warpAffine(bw, M, (w, h), borderValue=0)
        var = float(np.var(np.sum(rot > 0, axis=1).astype(float)))
        if var > best_var:
            best_var = var
            best_angle = float(a)

    if abs(best_angle) < 2.0:
        return img_bgr
    M = cv2.getRotationMatrix2D((w / 2, h / 2), -best_angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nW = int(h * sin + w * cos)
    nH = int(h * cos + w * sin)
    M[0, 2] += (nW / 2) - w / 2
    M[1, 2] += (nH / 2) - h / 2
    return cv2.warpAffine(img_bgr, M, (nW, nH),
                          flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=(255, 255, 255))


def _enhance_plate(img_bgr: np.ndarray, dark_bg: bool = False) -> np.ndarray:
    """
    Tăng contrast bằng CLAHE trên kênh L (LAB), upscale nếu nhỏ.
    dark_bg=True (Xanh/Đỏ): đảo màu để chữ trắng → chữ đen.
    """
    h, w = img_bgr.shape[:2]
    target_h = 96
    if h < target_h:
        s = target_h / h
        img_bgr = cv2.resize(img_bgr, (int(w * s), target_h),
                             interpolation=cv2.INTER_CUBIC)

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    L = clahe.apply(L)
    img_bgr = cv2.cvtColor(cv2.merge([L, A, B]), cv2.COLOR_LAB2BGR)

    if dark_bg:
        img_bgr = cv2.bitwise_not(img_bgr)
    return img_bgr


def _is_two_row_plate(img_bgr: np.ndarray) -> bool:
    """Biển 2 dòng khi tỉ lệ ngang/cao < 2.5 (xe máy, biển vuông)."""
    h, w = img_bgr.shape[:2]
    return w / max(h, 1) < 2.5


# ────────────────────────────────────────────────────────────
# HẬU XỬ LÝ TEXT
# ────────────────────────────────────────────────────────────
def _clean_text(raw: str) -> str:
    """Bỏ dấu câu, khoảng trắng, lower → upper."""
    return re.sub(r"[^0-9A-Za-z]", "", raw).upper()


def _fix_military_plate(text: str) -> str:
    """
    Sửa biển quân đội format: 2 chữ + 4 số.
    Ví dụ: PK5346, TM2366, KV6938, BQ1234
    - Vị trí 0, 1: phải là CHỮ
    - Vị trí 2, 3, 4, 5: phải là SỐ
    """
    if len(text) < 6:
        return text
    r = list(text[:6])  # chỉ lấy 6 ký tự

    # Vị trí 0, 1: phải là CHỮ
    for i in (0, 1):
        if r[i].isdigit():
            r[i] = _FIX_TO_LETTER.get(r[i], r[i])

    # Vị trí 2, 3, 4, 5: phải là SỐ
    for i in range(2, 6):
        if not r[i].isdigit():
            r[i] = _FIX_TO_DIGIT.get(r[i], r[i])

    return "".join(r)


def _fix_plate_format(text: str, is_military: bool = False) -> str:
    """
    Chuẩn hóa về định dạng VN.
    - Biển quân đội: 2 chữ + 4 số
    - Biển dân sự: 2 số + 1-2 chữ + 4-5 số
    """
    if len(text) < 5:
        return text

    # Phát hiện biển quân đội dựa trên 2 ký tự đầu là chữ
    if is_military or (len(text) >= 6 and text[0].isalpha() and text[1].isalpha()):
        return _fix_military_plate(text)

    r = list(text)

    # Vị trí 0, 1: phải là SỐ
    for i in (0, 1):
        if not r[i].isdigit():
            r[i] = _FIX_TO_DIGIT.get(r[i], r[i])

    # Vị trí 2: phải là CHỮ
    if r[2].isdigit():
        r[2] = _FIX_TO_LETTER.get(r[2], r[2])

    # Vị trí 3: có thể là CHỮ (series 2 chữ) hoặc SỐ
    if len(r) > 3:
        if r[3].isalpha() and r[2].isalpha():
            pass  # biển có 2 chữ series, giữ nguyên
        elif not r[3].isdigit():
            r[3] = _FIX_TO_DIGIT.get(r[3], r[3])

    # Vị trí 4 trở đi: phải là SỐ
    for i in range(4, len(r)):
        if not r[i].isdigit():
            r[i] = _FIX_TO_DIGIT.get(r[i], r[i])

    fixed = "".join(r)

    if _PLATE_RE.match(fixed):
        return fixed

    for trimmed in (fixed[1:], fixed[:-1], fixed[1:-1]):
        if _PLATE_RE.match(trimmed):
            return trimmed

    return fixed


def _apply_dark_bg_fix(text: str) -> str:
    """Biển Xanh: E/H/F → B tại vị trí series. KHÔNG áp cho biển quân đội."""
    if len(text) < 3:
        return text
    # Chỉ sửa nếu là biển dân sự (2 số đầu)
    if text[0].isdigit() and text[1].isdigit():
        if text[2] in _DARK_BG_FIX:
            text = text[:2] + _DARK_BG_FIX[text[2]] + text[3:]
    return text


def _format_display(text: str) -> str:
    """
    Tạo chuỗi hiển thị có dấu gạch ngang + chấm theo đúng format biển VN.

    Các trường hợp:
    - Biển quân đội (2 chữ + 4 số): PK5346      → PK-53-46
    - Biển 2 chữ series + 5 số:     29AA69999   → 29AA-699.99
    - Biển 2 chữ series + 4 số:     29AA6999    → 29AA-69.99
    - Biển 1 chữ series + 6 số:     29G133333   → 29G1-333.33
      (biển 2 dòng xe máy: dòng 1 "29-G1", dòng 2 "333.33")
    - Biển 1 chữ series + 5 số:     29B12345    → 29B-123.45
    - Biển 1 chữ series + 4 số:     29B1234     → 29B-12.34
    """
    t = text

    # Biển quân đội: 2 chữ đầu
    if len(t) >= 6 and t[0].isalpha() and t[1].isalpha():
        return f"{t[:2]}-{t[2:4]}-{t[4:]}"

    if len(t) < 6:
        return t

    # Tìm vị trí kết thúc phần series (chữ cái liên tiếp từ vị trí 2)
    series_end = 2
    while series_end < len(t) and t[series_end].isalpha():
        series_end += 1

    prefix = t[:series_end]   # vd "29G" hoặc "29AA"
    nums   = t[series_end:]   # phần số

    # Trường hợp đặc biệt: 1 chữ series + 6 số
    # → biển 2 dòng xe máy, dòng 1 = "29G1", dòng 2 = "33333"
    # → số đầu tiên thuộc prefix hiển thị
    if len(prefix) == 3 and len(nums) == 6:
        prefix = prefix + nums[0]   # "29G" + "1" = "29G1"
        nums   = nums[1:]           # "33333" (5 số)

    # Format phần số: nếu >= 5 số → NNN..N.NN (dot cách cuối 2 ký tự)
    if len(nums) >= 5:
        num_fmt = nums[:-2] + "." + nums[-2:]
    elif len(nums) == 4:
        # 4 số: NN.NN
        num_fmt = nums[:2] + "." + nums[2:]
    else:
        num_fmt = nums

    return f"{prefix}-{num_fmt}"


# ────────────────────────────────────────────────────────────
# CLASS CHÍNH
# ────────────────────────────────────────────────────────────
class PaddlePlateOCR:
    """
    OCR biển số Việt Nam dùng PaddleOCR.

    Sử dụng:
        ocr = PaddlePlateOCR()
        text, score = ocr.recognize(plate_crop, plate_color="Trang")

    Tham số init:
        use_mobile: True (default) → PP-OCRv5_mobile (nhẹ, nhanh, ~50MB)
                    False → PP-OCRv5_server (chính xác hơn, ~200MB)
        score_thresh: text recognition threshold (mặc định 0.5)
    """

    def __init__(self,
                 use_mobile: bool = True,
                 score_thresh: float = 0.5,
                 verbose: bool = True):
        det_name = "PP-OCRv5_mobile_det" if use_mobile else "PP-OCRv5_server_det"
        rec_name = "en_PP-OCRv5_mobile_rec" if use_mobile else "en_PP-OCRv5_server_rec"

        # Đặt lại ngưỡng log ngay trước khi tạo model (chống paddlex bật lại INFO)
        for _ln in ("ppocr", "paddle", "paddlex", "paddlex.inference", "paddleocr"):
            logging.getLogger(_ln).setLevel(logging.ERROR)

        if verbose:
            print(f"[INFO] Loading PaddleOCR ({det_name} + {rec_name})...")

        self._ocr = PaddleOCR(
            text_detection_model_name=det_name,
            text_recognition_model_name=rec_name,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            text_rec_score_thresh=score_thresh,
            enable_mkldnn=False,
        )
        if verbose:
            print("[OK] PaddleOCR ready")

    # ────────────────────────────────────────────────────────
    def _raw_predict(self, img_bgr: np.ndarray) -> tuple[list[str], list[float], list[list]]:
        """Gọi PaddleOCR thuần, trả về (texts, scores, boxes)."""
        try:
            result = list(self._ocr.predict(img_bgr))
        except Exception as e:
            print(f"[WARN] PaddleOCR error: {e}")
            return [], [], []
        if not result:
            return [], [], []
        r = result[0]
        texts  = list(r.get("rec_texts",  []) or [])
        scores = list(r.get("rec_scores", []) or [])
        boxes  = list(r.get("rec_polys",  []) or [])
        return texts, scores, boxes

    # ────────────────────────────────────────────────────────
    def _merge_two_rows(self,
                        texts: list[str],
                        scores: list[float],
                        boxes: list) -> tuple[str, float]:
        """
        Biển 2 dòng: ghép theo thứ tự trên-xuống, trái-phải.
        Mỗi box có dạng [[x1,y1],[x2,y2],[x3,y3],[x4,y4]].
        """
        if not texts:
            return "", 0.0
        if len(texts) == 1:
            return _clean_text(texts[0]), float(scores[0]) if scores else 0.0

        items = []
        for t, s, b in zip(texts, scores, boxes):
            try:
                arr = np.array(b).reshape(-1, 2)
                y_center = float(arr[:, 1].mean())
                x_left   = float(arr[:, 0].min())
            except Exception:
                y_center, x_left = 0.0, 0.0
            items.append((y_center, x_left, t, float(s)))

        ys = sorted([it[0] for it in items])
        if len(ys) >= 2:
            gaps = [(ys[i+1] - ys[i], i) for i in range(len(ys) - 1)]
            gmax, gi = max(gaps)
            mid_y = (ys[gi] + ys[gi+1]) / 2 if gmax > 5 else float("inf")
        else:
            mid_y = float("inf")

        top    = sorted([it for it in items if it[0] <  mid_y], key=lambda x: x[1])
        bottom = sorted([it for it in items if it[0] >= mid_y], key=lambda x: x[1])

        merged_text = "".join(_clean_text(it[2]) for it in (top + bottom))
        all_scores  = [it[3] for it in (top + bottom)]
        avg_score   = float(np.mean(all_scores)) if all_scores else 0.0
        return merged_text, avg_score

    # ────────────────────────────────────────────────────────
    def _try_pass(self, img_bgr: np.ndarray, two_row: bool) -> tuple[str, float]:
        """Một lần gọi PaddleOCR + merge kết quả."""
        texts, scores, boxes = self._raw_predict(img_bgr)
        if not texts:
            return "", 0.0
        if two_row:
            return self._merge_two_rows(texts, scores, boxes)
        items = []
        for t, s, b in zip(texts, scores, boxes):
            try:
                arr = np.array(b).reshape(-1, 2)
                x_left = float(arr[:, 0].min())
            except Exception:
                x_left = 0.0
            items.append((x_left, t, float(s)))
        items.sort(key=lambda x: x[0])
        merged = "".join(_clean_text(it[1]) for it in items)
        avg    = float(np.mean([it[2] for it in items])) if items else 0.0
        return merged, avg

    # ────────────────────────────────────────────────────────
    def recognize(self,
                  plate_bgr: np.ndarray,
                  plate_color: str | None = None) -> tuple[str, float]:
        """
        Nhận diện text từ ảnh biển số.

        Args:
            plate_bgr: ảnh crop biển số (BGR, OpenCV format)
            plate_color: 'Trang' / 'Vang' / 'Xanh' / 'Do' / None

        Returns:
            (text, score) — text đã fix định dạng VN, score in [0, 1]
        """
        if plate_bgr is None or plate_bgr.size == 0:
            return "UNREAD", 0.0

        # Biển tối cần invert để OCR (Xanh + Đỏ)
        invert_bg = plate_color in _INVERT_COLORS
        two_row   = _is_two_row_plate(plate_bgr)

        best_text, best_score = "", -1.0

        # === Lần 1: ảnh gốc + enhance ===
        enhanced = _enhance_plate(plate_bgr, dark_bg=invert_bg)
        t1, s1 = self._try_pass(enhanced, two_row)
        if t1 and s1 > best_score:
            best_text, best_score = t1, s1

        # === Lần 2: deskew + enhance (nếu lần 1 chưa tốt) ===
        if best_score < 0.85:
            deskewed = _deskew(plate_bgr)
            if deskewed.shape != plate_bgr.shape:
                enh2 = _enhance_plate(deskewed, dark_bg=invert_bg)
                t2, s2 = self._try_pass(enh2, two_row)
                if t2 and s2 > best_score:
                    best_text, best_score = t2, s2

        # === Lần 3: thử lại với màu đảo ngược (nếu confidence thấp) ===
        if best_score < 0.6:
            enh3 = _enhance_plate(plate_bgr, dark_bg=not invert_bg)
            t3, s3 = self._try_pass(enh3, two_row)
            if t3 and s3 > best_score + 0.05:
                best_text, best_score = t3, s3

        # === Lần 4: multi-scale (khi vẫn quá kém) ===
        if best_score < 0.5:
            ph, pw = plate_bgr.shape[:2]
            for sc in (1.5, 2.0, 0.7):
                rs = cv2.resize(plate_bgr, (max(20, int(pw*sc)), max(20, int(ph*sc))),
                                interpolation=cv2.INTER_CUBIC if sc > 1 else cv2.INTER_AREA)
                enh4 = _enhance_plate(rs, dark_bg=invert_bg)
                t4, s4 = self._try_pass(enh4, two_row)
                if t4 and s4 > best_score + 0.05:
                    best_text, best_score = t4, s4
                if best_score >= 0.75:
                    break

        if not best_text:
            return "UNREAD", 0.0

        # Hậu xử lý format VN
        # Phát hiện biển quân đội: 2 chữ đầu là alpha
        is_military = (len(best_text) >= 2 and
                       best_text[0].isalpha() and best_text[1].isalpha())

        fixed = _fix_plate_format(best_text, is_military=is_military)

        # Chỉ áp dark bg fix cho biển Xanh dân sự, không áp cho biển Đỏ quân đội
        if plate_color in _DARK_BG_COLORS:
            fixed = _apply_dark_bg_fix(fixed)

        return fixed, max(best_score, 0.0) # Trả về text đã fix và score.


def format_plate_display(text: str) -> str:
    """
    Hàm public để format hiển thị biển số có dấu gạch ngang đúng format.
    Dùng trong util.py / gui.py khi hiển thị kết quả.
    """
    return _format_display(text) #Hàm tiện ích để format biển số hiển thị đúng chuẩn (thêm gạch ngang, chấm).
