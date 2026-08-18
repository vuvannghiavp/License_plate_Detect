# — Nhận Diện Biển Số Xe Việt Nam

YOLOv8 phát hiện biển → PaddleOCR đọc chữ → ColorCNN nhận màu (Trắng / Vàng / Xanh / Đỏ)

---

## Cài đặt

**Bước 1 — Tạo môi trường ảo**

```
python -m venv venv
venv\Scripts\activate          # Windows
```
**Bước 2 — Cài thư viện**

```
pip install -r requirements.txt
```
---
## Chạy nhận diện

**Bước 1 — Chuẩn bị ảnh**
Chụp hoặc lấy một ảnh xe có biển số, đổi tên thành `anh.jpg` rồi copy vào thư mục dự án (cùng chỗ với `main.py`).

**Bước 2 — Chạy**

```
python main.py --image anh.jpg --show
```

Kết quả lưu tại `outputs/output.jpg` (ảnh có vẽ khung + nhãn) và `outputs/test.csv` (bảng dữ liệu).

**Một số tùy chọn:**

| Lệnh                     | Tác dụng                             |
|--------------------------|--------------------------------------|
| `--image <file>`         | Chỉ định ảnh đầu vào                 |
| `--show`                 | Hiện cửa sổ xem kết quả              |
| `--output-image <file>`  | Đặt tên file ảnh kết quả             |
| `--plate-threshold 0.15` | Giảm ngưỡng nếu bỏ sót biển          |
| `--ocr-server`           | OCR chính xác hơn (tải thêm ~200 MB) |

---

## Huấn luyện lại mô hình màu

Cần ít nhất **50 ảnh/lớp** (khuyến nghị 100+), đặt vào đúng thư mục:

```
train/Do/      ← biển đỏ (quân đội)
train/Trang/   ← biển trắng (cá nhân)
train/Vang/    ← biển vàng (vận tải)
train/Xanh/    ← biển xanh (nhà nước)
```

Sau đó chạy:

```
python train_color.py
```

Sau khi train xong, script sẽ tự lưu:

- `weights/color_cnn_new.pth` : model tốt nhất
- `weights/train_history.csv` : log từng epoch
- `weights/train_history.png` : biểu đồ loss / accuracy

### Cấu trúc dữ liệu

```text
dataset/
  train/
    Do/
    Trang/
    Vang/
    Xanh/
  valid/
    Do/
    Trang/
    Vang/
    Xanh/
  test/
    Do/
    Trang/
    Vang/
    Xanh/
```

Chạy script chia dữ liệu:

```bash
python prepare_dataset.py
```

Mặc định script sẽ tạo:
- `80%` train
- `10%` valid
- `10%` test

