

from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path


CLASSES = ["Do", "Trang", "Vang", "Xanh"]


def split_counts(n: int, train_ratio: float, valid_ratio: float, test_ratio: float) -> tuple[int, int, int]:
    if n <= 0:
        return 0, 0, 0

    ratios = [train_ratio, valid_ratio, test_ratio]
    total = sum(ratios)
    if total <= 0:
        raise ValueError("Ratios must sum to a positive number")

    ratios = [r / total for r in ratios]
    raw = [n * r for r in ratios]
    counts = [int(x) for x in raw]
    remainder = n - sum(counts)
    fracs = sorted(
        ((raw[i] - counts[i], i) for i in range(3)),
        reverse=True,
    )
    for _, idx in fracs[:remainder]:
        counts[idx] += 1
    return counts[0], counts[1], counts[2]


def collect_images(source_root: Path, class_name: str) -> list[Path]:
    folder = source_root / class_name
    if not folder.exists():
        return []
    files = sorted([*folder.glob("*.jpg"), *folder.glob("*.jpeg"), *folder.glob("*.png")])
    return files


def prepare_dataset(source_dir: str = "train", dest_dir: str = "dataset", train_ratio: float = 0.8, valid_ratio: float = 0.1, test_ratio: float = 0.1, seed: int = 42, force: bool = False) -> None:
    source_root = Path(source_dir)
    dest_root = Path(dest_dir)

    if not source_root.exists():
        raise FileNotFoundError(f"Source folder not found: {source_root}")

    if dest_root.exists() and any(dest_root.iterdir()):
        if not force:
            raise FileExistsError(f"Destination folder already exists and is not empty: {dest_root}")
        shutil.rmtree(dest_root)

    random.seed(seed)

    summary_rows = []
    for split in ("train", "valid", "test"):
        for cls in CLASSES:
            (dest_root / split / cls).mkdir(parents=True, exist_ok=True)

    for cls in CLASSES:
        files = collect_images(source_root, cls)
        random.shuffle(files)
        n_train, n_valid, n_test = split_counts(len(files), train_ratio, valid_ratio, test_ratio)
        train_files = files[:n_train]
        valid_files = files[n_train:n_train + n_valid]
        test_files = files[n_train + n_valid:n_train + n_valid + n_test]

        for src in train_files:
            shutil.copy2(src, dest_root / "train" / cls / src.name)
        for src in valid_files:
            shutil.copy2(src, dest_root / "valid" / cls / src.name)
        for src in test_files:
            shutil.copy2(src, dest_root / "test" / cls / src.name)

        summary_rows.append({
            "class": cls,
            "train": len(train_files),
            "valid": len(valid_files),
            "test": len(test_files),
            "total": len(files),
        })

    summary_path = dest_root / "split_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["class", "train", "valid", "test", "total"])
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"[OK] Split dataset created at: {dest_root}")
    for row in summary_rows:
        print(f"  {row['class']:6s}: train={row['train']:3d} valid={row['valid']:3d} test={row['test']:3d} total={row['total']:3d}")
    print(f"[OK] Summary saved to: {summary_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Split flat color dataset into train/valid/test")
    ap.add_argument("--source", default="train", help="Source folder with class subfolders")
    ap.add_argument("--dest", default="dataset", help="Destination dataset folder")
    ap.add_argument("--train-ratio", type=float, default=0.8)
    ap.add_argument("--valid-ratio", type=float, default=0.1)
    ap.add_argument("--test-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true", help="Overwrite destination folder if it exists")
    args = ap.parse_args()
    prepare_dataset(
        source_dir=args.source,
        dest_dir=args.dest,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        force=args.force,
    )
