'''How to train use command: python train_color.py'''

from __future__ import annotations

import argparse
import csv
import random
from math import ceil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from color_cnn import ColorCNN


CLASSES = ["Do", "Trang", "Vang", "Xanh"]
IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 80
LR = 5e-4
DATA_DIR = "dataset"
SAVE_PATH = "weights/color_cnn_new.pth"
HISTORY_CSV = "weights/train_history.csv"
HISTORY_PNG = "weights/train_history.png"
WARMUP_EPOCHS = 5


train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.1),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.08),
    transforms.RandomRotation(15),
    transforms.RandomPerspective(distortion_scale=0.15, p=0.4),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
])

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])


class ColorDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def load_samples(data_dir: str) -> list:
    """Load samples from one class-root folder like train/Do, train/Trang, ..."""
    samples = []
    print(f"\nLoading dataset from '{data_dir}/':")
    for label_idx, class_name in enumerate(CLASSES):
        folder = Path(data_dir) / class_name
        if not folder.exists():
            print(f"  [!!] Missing: {folder}")
            continue
        imgs = sorted(list(folder.glob("*.jpg")) + list(folder.glob("*.png")))
        for p in imgs:
            samples.append((str(p), label_idx))
        print(f"  {class_name:6s}: {len(imgs):4d} images")
    return samples


def detect_split_layout(data_root: Path) -> bool:
    """Return True when a split layout exists under data_root."""
    train_dir = data_root / "train"
    valid_dir = data_root / "valid"
    val_dir = data_root / "val"
    test_dir = data_root / "test"
    return train_dir.exists() and (valid_dir.exists() or val_dir.exists() or test_dir.exists())


def load_split_samples(data_root: Path) -> tuple[list, list, list]:
    """Load train/valid/test from a split layout."""
    train_samples = load_samples(str(data_root / "train"))

    valid_dir = data_root / "valid"
    if not valid_dir.exists():
        valid_dir = data_root / "val"
    valid_samples = load_samples(str(valid_dir)) if valid_dir.exists() else []

    test_dir = data_root / "test"
    test_samples = load_samples(str(test_dir)) if test_dir.exists() else []

    return train_samples, valid_samples, test_samples


def split_flat_samples(all_samples: list, train_ratio: float = 0.85) -> tuple[list, list]:
    """Split a flat dataset into train and val lists."""
    random.shuffle(all_samples)
    n_train = int(len(all_samples) * train_ratio)
    return all_samples[:n_train], all_samples[n_train:]


class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        ep = self.last_epoch
        if ep < self.warmup_epochs:
            factor = (ep + 1) / max(self.warmup_epochs, 1)
        else:
            progress = (ep - self.warmup_epochs) / max(self.total_epochs - self.warmup_epochs, 1)
            factor = (
                self.min_lr / self.base_lrs[0]
                + 0.5 * (1 - self.min_lr / self.base_lrs[0]) * (1 + np.cos(np.pi * progress))
            )
        return [base_lr * factor for base_lr in self.base_lrs]


def evaluate(model, loader, device) -> tuple[float, dict]:
    model.eval()
    correct, total = 0, 0
    class_correct = [0] * len(CLASSES)
    class_total = [0] * len(CLASSES)

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            for p, t in zip(preds, labels):
                class_total[t.item()] += 1
                class_correct[t.item()] += int(p == t)

    acc = correct / total * 100 if total else 0
    per_class = {
        CLASSES[i]: class_correct[i] / max(class_total[i], 1) * 100
        for i in range(len(CLASSES))
    }
    return acc, per_class


def save_training_history(history: list[dict], csv_path: str = HISTORY_CSV, png_path: str = HISTORY_PNG) -> None:
    """Save per-epoch metrics to CSV and render a simple PNG chart."""
    if not history:
        return

    csv_path = Path(csv_path)
    png_path = Path(png_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["epoch", "train_loss", "val_acc", "lr"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception as exc:  # pragma: no cover - Pillow is already a dependency
        print(f"[WARN] Could not create PNG chart: {exc}")
        return

    width, height = 1200, 720
    margin = 70
    gap = 40
    panel_h = (height - margin * 2 - gap) // 2
    panel_w = width - margin * 2
    bg = (248, 250, 252)
    fg = (30, 41, 59)
    grid = (203, 213, 225)
    loss_color = (239, 68, 68)
    acc_color = (37, 99, 235)
    lr_color = (16, 185, 129)

    img = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    def draw_rotated_text(target_img, xy, text, fill, angle):
        """Draw text rotated by angle degrees around the given top-left position."""
        txt = Image.new("RGBA", (260, 40), (255, 255, 255, 0))
        txt_draw = ImageDraw.Draw(txt)
        txt_draw.text((0, 0), text, fill=fill, font=font)
        rot = txt.rotate(angle, expand=1)
        target_img.paste(rot, xy, rot)

    epochs = [row["epoch"] for row in history]
    losses = [float(row["train_loss"]) for row in history]
    accs = [float(row["val_acc"]) for row in history]
    lrs = [float(row["lr"]) for row in history]

    def draw_panel(x0, y0, title, y_label, series, color, y_min=None, y_max=None, right_axis=False, right_series=None, right_color=None):
        x1 = x0 + panel_w
        y1 = y0 + panel_h
        draw.rounded_rectangle([x0, y0, x1, y1], radius=16, fill="white", outline=grid, width=2)
        draw.text((x0 + 16, y0 + 12), title, fill=fg, font=font)
        draw.text((x0 + 16, y0 + 30), y_label, fill=(100, 116, 139), font=font)

        left = x0 + 60
        top = y0 + 50
        right = x1 - 24
        bottom = y1 - 40

        draw.line([left, top, left, bottom], fill=fg, width=2)
        draw.line([left, bottom, right, bottom], fill=fg, width=2)
        draw.text((x0 + 8, y1 - 26), "Trục X: Epoch", fill=(70, 84, 100), font=font)
        draw_rotated_text(img, (x0 + 4, y0 + 90), y_label, (70, 84, 100), 90)

        values = list(series)
        if y_min is None:
            y_min = min(values)
        if y_max is None:
            y_max = max(values)
        if y_max == y_min:
            y_max = y_min + 1.0

        for i in range(5):
            yy = top + (bottom - top) * i / 4
            val = y_max - (y_max - y_min) * i / 4
            draw.line([left - 6, yy, right, yy], fill=grid, width=1)
            draw.text((10, yy - 6), f"{val:.2f}" if y_max <= 10 else f"{val:.0f}", fill=(100, 116, 139), font=font)

        n = len(values)
        if n == 1:
            xs = [left]
        else:
            xs = [left + (right - left) * i / (n - 1) for i in range(n)]

        def scale_y(v):
            return bottom - (v - y_min) * (bottom - top) / (y_max - y_min)

        pts = [(xs[i], scale_y(values[i])) for i in range(n)]
        for i in range(1, len(pts)):
            draw.line([pts[i - 1], pts[i]], fill=color, width=4)
        for px, py in pts:
            draw.ellipse([px - 4, py - 4, px + 4, py + 4], fill=color)

        max_epoch = max(epochs) if epochs else 1
        for i, ep in enumerate(epochs):
            if n <= 10 or i % max(1, ceil(n / 10)) == 0 or i == n - 1:
                tx = xs[i]
                draw.line([tx, bottom, tx, bottom + 6], fill=fg, width=1)
                draw.text((tx - 6, bottom + 8), str(ep), fill=(100, 116, 139), font=font)

        if right_axis and right_series is not None and right_color is not None:
            r_values = list(right_series)
            r_min, r_max = min(r_values), max(r_values)
            if r_max == r_min:
                r_max = r_min + 1.0
            r_pts = [(xs[i], bottom - (r_values[i] - r_min) * (bottom - top) / (r_max - r_min)) for i in range(n)]
            for i in range(1, len(r_pts)):
                draw.line([r_pts[i - 1], r_pts[i]], fill=right_color, width=3)
            for px, py in r_pts:
                draw.rectangle([px - 3, py - 3, px + 3, py + 3], fill=right_color)
            draw.text((x1 - 150, y0 + 12), "LR", fill=right_color, font=font)

    draw_panel(margin, margin, "Training Loss", "Trục Y: Loss", losses, loss_color)
    draw_panel(margin, margin + panel_h + gap, "Validation Accuracy", "Trục Y: Accuracy (%)", accs, acc_color, y_min=0, y_max=100, right_axis=True, right_series=lrs, right_color=lr_color)

    img.save(png_path)
    print(f"[OK] Saved training history: {csv_path}")
    print(f"[OK] Saved training chart   : {png_path}")


def train(optimizer_name: str = "adamw", epochs: int = EPOCHS, lr: float = LR, data_dir: str = DATA_DIR):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    print(f"Device    : {device}")
    print(f"Optimizer : {optimizer_name.upper()}")
    print(f"LR        : {lr}  |  Epochs: {epochs}  |  Warmup: {WARMUP_EPOCHS}")
    print(f"AMP (fp16): {use_amp}")

    save_path = Path(SAVE_PATH)
    history_csv = Path(HISTORY_CSV)
    history_png = Path(HISTORY_PNG)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    history_csv.parent.mkdir(parents=True, exist_ok=True)
    history_png.parent.mkdir(parents=True, exist_ok=True)

    data_root = Path(data_dir).resolve()
    if detect_split_layout(data_root):
        print("\n[INFO] Detected split layout: train/valid/test")
        train_samples, val_samples, test_samples = load_split_samples(data_root)
        if not train_samples:
            print("[ERROR] No training data found in split layout.")
            return
    else:
        print("\n[INFO] Detected flat layout: auto train/val split")
        train_root = data_root if data_root.name == "train" else data_root / "train"
        all_samples = load_samples(str(train_root))
        if not all_samples:
            print("[ERROR] No data found. Expected train/Do, train/Trang, train/Vang, train/Xanh")
            return
        train_samples, val_samples = split_flat_samples(all_samples)
        test_samples = []

    train_ds = ColorDataset(train_samples, train_tf)
    val_ds = ColorDataset(val_samples, val_tf)
    print(f"\nTrain: {len(train_ds)} | Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, BATCH_SIZE, shuffle=False, num_workers=0)

    model = ColorCNN(num_classes=4).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    if optimizer_name.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)

    scheduler = WarmupCosineScheduler(optimizer, WARMUP_EPOCHS, epochs, min_lr=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_acc = 0.0
    patience = 0
    max_patience = 20
    history: list[dict] = []

    print(f"\nStart training {epochs} epochs (early stop = {max_patience})...")
    print("-" * 65)

    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                loss = criterion(model(imgs), labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.item()

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        acc, per_class = evaluate(model, val_loader, device)
        mark = ""

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), save_path)
            mark = "  <- SAVED"
            patience = 0
        else:
            patience += 1

        if epoch % 10 == 0 or epoch == 1 or mark:
            avg_loss = loss_sum / max(len(train_loader), 1)
            print(f"Epoch {epoch:3d}/{epochs} | Loss:{avg_loss:.4f} | Val:{acc:.1f}% | LR:{current_lr:.2e}{mark}")
            if mark:
                detail = " | ".join(f"{k}:{v:.0f}%" for k, v in per_class.items())
                print(f"           Per-class: {detail}")

        avg_loss = loss_sum / max(len(train_loader), 1)
        history.append({
            "epoch": epoch,
            "train_loss": round(avg_loss, 6),
            "val_acc": round(acc, 4),
            "lr": round(current_lr, 8),
        })

        if patience >= max_patience:
            print(f"\n[INFO] Early stopping ({max_patience} epochs without improvement)")
            break

    print("-" * 65)
    print(f"Best Val: {best_acc:.1f}%  |  Saved: {save_path}")

    print("\n>> Per-class accuracy (best model):")
    best_model = ColorCNN(num_classes=4).to(device)
    best_model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    _, final_per = evaluate(best_model, val_loader, device)
    for cls, acc_c in final_per.items():
        status = "OK" if acc_c >= 70 else "CAN THEM DATA"
        print(f"  {cls:6s}: {acc_c:.1f}%  [{status}]")

    if test_samples:
        test_loader = DataLoader(ColorDataset(test_samples, val_tf), BATCH_SIZE, shuffle=False, num_workers=0)
        test_acc, test_per = evaluate(best_model, test_loader, device)
        print("\n>> Test accuracy (held-out test set):")
        print(f"  Overall: {test_acc:.1f}%")
        for cls, acc_c in test_per.items():
            print(f"  {cls:6s}: {acc_c:.1f}%")

    if best_acc >= 70:
        print("\n>> Target reached: >70%. Ready to use with main.py")
    else:
        print("\n>> Not yet at 70%. Try:")
        print("   1. Increase epochs: python train_color.py --epochs 120")
        print("   2. Try another optimizer: python train_color.py --optimizer adam --lr 1e-3")
        print("   3. Add more images to train/Do or train/Xanh if per-class is low")

    save_training_history(history, str(history_csv), str(history_png))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train ColorCNN for license plate color classification")
    ap.add_argument("--optimizer", default="adamw", choices=["adam", "adamw"])
    ap.add_argument("--epochs", default=EPOCHS, type=int)
    ap.add_argument("--lr", default=LR, type=float)
    ap.add_argument("--data-dir", default=DATA_DIR, help="Data root for either flat or split layout")
    ap.add_argument("--save-path", default=SAVE_PATH, help="Path to save best weights")
    ap.add_argument("--history-csv", default=HISTORY_CSV, help="Path to save training history CSV")
    ap.add_argument("--history-png", default=HISTORY_PNG, help="Path to save training history PNG")
    args = ap.parse_args()
    SAVE_PATH = args.save_path
    HISTORY_CSV = args.history_csv
    HISTORY_PNG = args.history_png
    train(optimizer_name=args.optimizer, epochs=args.epochs, lr=args.lr, data_dir=args.data_dir)
