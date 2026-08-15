"""
main.py — Nhận Diện Biển Số Xe Việt Nam
========================================
Pipeline 3 bước:
    1. YOLOv8            : Phát hiện vùng biển số
    2. PaddleOCR          : Đọc text trên biển
    3. ColorCNN + HSV     : Nhận màu biển (Trắng / Vàng / Xanh / Đỏ)

Chạy:
    python main.py --image anh.jpg --show
    python main.py --image anh.jpg --output-image result.jpg

Cài đặt:
    pip install -r requirements.txt
Run: python main.py --image anh.jpg --show
with video : python main.py --video ten_video.mp4 --show
"""
from __future__ import annotations
import argparse #đọc tham số từ terminal bên dưới ví dụ: python main.py --image abc.jpg
from pathlib import Path #Làm việc với đường dẫn file ví dụ: Path("abc.jpg") "dòng 45"

import cv2 # dùng để đọc ảnh, ghi ảnh, vẽ hình chữ nhật, hiển thị ảnh
import torch #thư viện pytorch, dùng để chạy YOLO, chạy ColorCNN, kiểm tra GPU

from util import clamp_xyxy, detect_plates, load_yolo_model, resolve_path, write_csv, draw_plate_label # import các hàm ở file util.py, resolve : Vì nếu bạn chạy chương trình từ thư mục khác, Python có thể không tìm thấy model.
from color_detector import ColorDetector, PLATE_PURPOSE, BOX_COLOR #thêm từ file color_detector.py, colordetector: nhận diện màu biển số xe, plate... tạo dictionary trang: dân sự..., box_color: màu khung khi vẽ
from paddle_ocr_engine import PaddlePlateOCR #dùng PaddlePlateOCR để đọc chữ trên biển


# ── Tham số dòng lệnh ────────────────────────────────────────
def build_args(): #Tạo đối tượng chứa tham số dòng lệnh.
    ap = argparse.ArgumentParser(description="Nhận diện biển số xe VN (YOLOv8 + PaddleOCR + ColorCNN+HSV)")
    ap.add_argument("--image",           default="Do2.jpg",                         help="Đường dẫn ảnh đầu vào")
    ap.add_argument("--video",           default=None,                              help="Đường dẫn video đầu vào")
    ap.add_argument("--plate-model",     default="models/license_plate_detector.pt", help="YOLO weights") #tham số model Yolo
    ap.add_argument("--color-weights",   default="weights/color_cnn_new.pth",       help="ColorCNN weights") #tham số model ColorCNN
    ap.add_argument("--output-image",    default="outputs/output.jpg",               help="Ảnh kết quả")
    ap.add_argument("--output-video",    default="outputs/output.mp4",               help="Video kết quả")
    ap.add_argument("--output-csv",      default="outputs/test.csv",                 help="CSV kết quả")
    ap.add_argument("--plate-threshold", default=0.25, type=float,                   help="Ngưỡng YOLO")
    ap.add_argument("--ocr-server",      action="store_true",                        help="Dùng PP-OCRv5 SERVER (~200MB, chính xác hơn)")
    ap.add_argument("--show",            action="store_true",                        help="Hiển thị kết quả trên màn hình")
    return ap.parse_args()


# ── Chương trình chính ───────────────────────────────────────
def process_frame(frame, frame_idx, plate_model, color_det, ocr, args, results):
    H, W = frame.shape[:2]
    plates = detect_plates(plate_model, frame, args.plate_threshold)
    print(f"[INFO] Frame {frame_idx}: tìm thấy {len(plates)} biển số")

    if not plates:
        return frame

    results[frame_idx] = {}
    for i, plate in enumerate(plates):
        x1, y1, x2, y2 = clamp_xyxy(frame, plate[:4])

        # Padding nhẹ quanh bbox để không cắt mất ký tự
        pad_x = max(2, int(0.05 * (x2 - x1)))
        pad_y = max(2, int(0.08 * (y2 - y1)))
        crop = frame[max(0, y1 - pad_y):min(H, y2 + pad_y),
                     max(0, x1 - pad_x):min(W, x2 + pad_x)]
        if crop.shape[0] < 10 or crop.shape[1] < 10:
            continue

        # 4a. Nhận màu (ColorCNN + HSV)
        color_label, color_conf = color_det.detect(crop)

        # 4b. Đọc text (PaddleOCR)
        text, text_score = ocr.recognize(crop, plate_color=color_label)

        purpose = PLATE_PURPOSE.get(color_label, "?")
        box_color = BOX_COLOR.get(color_label, (0, 255, 0))

        # 4c. Vẽ kết quả lên ảnh
        draw_plate_label(frame, (x1, y1, x2, y2), text, text_score,
                         color_label, color_conf, purpose, box_color)

        print(f"  [{i+1}] Biển: {text:12s} ({text_score:.0%}) | "
              f"Màu: {color_label:5s} ({color_conf:.0%}) - {purpose}")

        results[frame_idx][i] = {"license_plate": {
            "bbox": list(map(float, plate[:4])),
            "text": text,
            "text_score": text_score,
            "color": color_label,
            "purpose": purpose,
            "bbox_score": float(plate[4]) if len(plate) > 4 else 0,
        }}

    return frame


def main():
    args = build_args()  # đọc tham số
    base = Path(__file__).resolve().parent  # lấy thư mục project
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")
    print("=" * 60)

    # Bước 1 — Nạp 3 model
    plate_model = load_yolo_model(resolve_path(base, args.plate_model), "YOLO plate detector")
    color_det = ColorDetector(weights_path=str(resolve_path(base, args.color_weights)), device=device)
    ocr = PaddlePlateOCR(use_mobile=not args.ocr_server, score_thresh=0.5, verbose=True)
    print("=" * 60)

    # Bước 2 — Đọc đầu vào (ảnh hoặc video)
    if args.video:
        video_path = resolve_path(base, args.video)
        if not video_path.exists():
            print(f"[!!] Không tìm thấy video: {video_path}")
            return

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"[!!] Không mở được video: {video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_video = resolve_path(base, args.output_video)
        writer = None
        if width > 0 and height > 0:
            writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

        results = {}
        frame_idx = 0
        print(f"[INFO] Video: {video_path.name} -> {out_video.name}")
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            frame = process_frame(frame, frame_idx, plate_model, color_det, ocr, args, results)
            if writer is not None:
                writer.write(frame)
            if args.show:
                cv2.imshow("Kết Quả Nhận Diện Biển Số", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()

        out_csv = resolve_path(base, args.output_csv)
        write_csv(results, str(out_csv))
        print("=" * 60)
        print(f"[INFO] Video kết quả : {out_video}")
        print(f"[INFO] CSV kết quả   : {out_csv}")
        print("HOÀN THÀNH!")
        return

    # Ảnh đơn lẻ (giữ nguyên cách cũ)
    img_path = resolve_path(base, args.image)
    if not img_path.exists():
        print(f"[!!] Không tìm thấy ảnh: {img_path}")
        return
    frame = cv2.imread(str(img_path))
    if frame is None:
        print(f"[!!] Không đọc được: {img_path}")
        return
    H, W = frame.shape[:2]
    print(f"[INFO] Ảnh: {img_path.name}  ({W}x{H})")

    results = {0: {}}
    frame = process_frame(frame, 0, plate_model, color_det, ocr, args, results)

    # Lưu kết quả
    out_img = resolve_path(base, args.output_image)
    out_csv = resolve_path(base, args.output_csv)
    cv2.imwrite(str(out_img), frame)
    write_csv(results, str(out_csv))

    print("=" * 60)
    print(f"[INFO] Ảnh kết quả : {out_img}")
    print(f"[INFO] CSV kết quả : {out_csv}")
    print("HOÀN THÀNH!")

    if args.show:
        cv2.imshow("Kết Quả Nhận Diện Biển Số", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
