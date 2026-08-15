"""
color_detector.py — Nhận màu biển số
======================================
Kết hợp 2 nguồn:
  - ColorCNN : phân loại màu qua mạng nơ-ron
  - HSV heuristic : tính tỉ lệ pixel theo không gian màu HSV

Luật kết hợp (fusion):
  - Đồng thuận                 -> cộng hưởng confidence (weighted average)
  - Bất đồng + HSV bắt ĐỎ rõ   -> ưu tiên HSV ở ngưỡng thấp (cứu biển quân đội)
  - Bất đồng + HSV mạnh (>30%) -> ưu tiên HSV
  - Còn lại                    -> tin CNN
"""
from __future__ import annotations
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms

from color_cnn import load_color_model


CLASSES = ["Do", "Trang", "Vang", "Xanh"]

PLATE_PURPOSE = {
    "Trang": "Xe ca nhan / Doanh nghiep",
    "Xanh":  "Xe co quan Nha nuoc",
    "Vang":  "Xe kinh doanh van tai",
    "Do":    "Xe Quan doi",
}

BOX_COLOR = {
    "Trang": (180, 180, 180),
    "Vang":  (0,   200, 255),
    "Do":    (0,   0,   220),
    "Xanh":  (200, 80,  0  ),
}

# Pipeline chuẩn hóa ảnh đầu vào cho CNN
_TFM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])
class ColorDetector:
    """Wrapper gọi ColorCNN + HSV fusion để nhận màu biển số."""

    def __init__(self, weights_path: str, device: torch.device):
        self.device = device
        self.model  = load_color_model(weights_path, device)
        print("[OK] ColorCNN ready")

    def _predict_cnn(self, crop_bgr: np.ndarray) -> tuple[str, float]:
        """
        Forward pass qua ColorCNN, trả (nhãn, confidence).

        LƯU Ý: ColorCNN ở eval mode (xem color_cnn.py) ĐÃ trả về xác suất
        Softmax sẵn. Vì vậy ở đây KHÔNG được softmax lại lần nữa — nếu softmax
        chồng softmax ("double softmax") thì phân phối bị làm phẳng về phía 0.25,
        khiến confidence của CNN bị bóp dẹt và luôn thua HSV trong bước fusion.
        """
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        t   = _TFM(rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            prob = self.model(t)             # đã là xác suất Softmax
            conf, idx = torch.max(prob, 1)
        return CLASSES[idx.item()], float(conf.item())

    @staticmethod
    def _predict_hsv(crop_bgr: np.ndarray) -> tuple[str, float]:
        """
        Tính tỉ lệ pixel từng màu trong không gian HSV.
        Cắt bỏ viền 10% để loại khung kim loại.
        Chỉ xét pixel sáng (V > 90) để loại bóng/nhiễu tối.
        """
        h, w   = crop_bgr.shape[:2]
        py, px = max(1, h // 10), max(1, w // 10)
        inner  = crop_bgr[py:h-py, px:w-px]
        if inner.size == 0:
            inner = crop_bgr
        hsv = cv2.cvtColor(inner, cv2.COLOR_BGR2HSV)

        mask_bright = cv2.inRange(hsv, (0, 0, 90), (180, 255, 255))
        total = max(cv2.countNonZero(mask_bright), 1)
        bright_pixels = hsv[mask_bright > 0]
        avg_sat = float(np.mean(bright_pixels[:, 1])) if bright_pixels.size else 255.0
        avg_val = float(np.mean(bright_pixels[:, 2])) if bright_pixels.size else 0.0

        def pct(mask):
            return cv2.countNonZero(cv2.bitwise_and(mask, mask_bright)) / total

        # Màu đỏ: Hue vắt qua hai đầu 0/180 nên phải OR hai dải.
        # Biển đỏ quân đội thường là ĐỎ THẪM (bordeaux): S và V thấp hơn đỏ tươi.
        # Vì vậy hạ ngưỡng xuống S>=70, V>=50 để không bỏ sót biển đỏ tối màu.
        m_do = cv2.bitwise_or(
            cv2.inRange(hsv, (0,   70, 50), (12,  255, 255)),
            cv2.inRange(hsv, (155, 70, 50), (180, 255, 255)),
        )
        scores = {
            "Do":    pct(m_do),
            "Vang":  pct(cv2.inRange(hsv, (18, 130, 90), (40,  255, 255))),
            "Xanh":  pct(cv2.inRange(hsv, (95, 90,  60), (135, 255, 255))),
            "Trang": pct(cv2.inRange(hsv, (0,  0,  125), (180, 75,  255))),
        }
        if scores["Trang"] >= 0.12 and avg_sat <= 95 and avg_val >= 140:
            return "Trang", scores["Trang"]
        best = max(scores, key=scores.get)
        if best == "Xanh" and scores["Trang"] >= 0.08 and avg_sat <= 105 and avg_val >= 135:
            return "Trang", scores["Trang"]
        return best, scores[best]

    def detect(self, crop_bgr: np.ndarray) -> tuple[str, float]:
        """
        Kết hợp CNN và HSV để đưa ra nhãn màu cuối cùng.
        Trả về (nhãn, confidence).
        """
        cnn_label, cnn_conf = self._predict_cnn(crop_bgr)
        hsv_label, hsv_val  = self._predict_hsv(crop_bgr)

        def to_display(strength: float) -> float:
            s = min(max(strength, 0.0), 1.0)
            return round(0.70 + 0.30 * s, 4)

        # 1) Hai luồng đồng thuận -> trộn bằng chứng của CNN và HSV.
        #    CNN và HSV là 2 phương pháp ĐỘC LẬP, cùng ra một màu thì rất đáng tin.
        #    hsv_val được chuẩn hóa theo 0.5 (tỉ lệ pixel 1 màu hiếm khi vượt mức này).
        if cnn_label == hsv_label:
            strength = 0.7 * cnn_conf + 0.3 * min(hsv_val / 0.5, 1.0)
            return cnn_label, to_display(strength)

        # 2) Bất đồng nhưng HSV bắt được màu ĐỎ rõ -> ưu tiên HSV.
        #    Đỏ là tín hiệu vật lý đặc trưng (biển quân đội) và CNN hay nhầm
        #    Đỏ <-> Xanh, nên chỉ cần ratio đỏ >= 0.12 là tin HSV. Đây chính là
        #    luật cứu các ca như biển "KC-88-88" bị CNN dự đoán nhầm thành Xanh.
        if hsv_label == "Do" and hsv_val >= 0.12:
            return hsv_label, to_display(min(hsv_val / 0.4, 1.0))

        # 3) Bất đồng, HSV mạnh ở các màu còn lại -> tin HSV.
        if hsv_val > 0.30:
            return hsv_label, to_display(min(hsv_val / 0.5, 1.0))

        # 4) Còn lại -> tin CNN. 
        return cnn_label, to_display(cnn_conf)
