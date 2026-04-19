"""Evalua el clasificador en el test de MangoDHDS.

Reporta precision/recall/F1 por clase y matriz de confusion. Guarda:
    reports/figures/confusion_matrix_cls.png
    reports/tables/metrics_cls.csv

Uso:
    python src/evaluation/evaluate_cls.py
    python src/evaluation/evaluate_cls.py --weights runs/classify/train_cls/weights/best.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import (
    CLASSES,
    DATASET_CLASSIFIER_ROOT,
    FIGURES_DIR,
    IMG_SIZE_CLS,
    TABLES_DIR,
    WEIGHTS_CLS,
)

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", type=Path, default=WEIGHTS_CLS)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--imgsz", type=int, default=IMG_SIZE_CLS)
    return p.parse_args()


def collect_test_samples(split_dir: Path) -> list[tuple[Path, int]]:
    samples: list[tuple[Path, int]] = []
    for idx, cls_name in enumerate(CLASSES):
        cls_dir = split_dir / cls_name
        if not cls_dir.is_dir():
            continue
        for img in sorted(cls_dir.iterdir()):
            if img.suffix.lower() in IMG_EXT:
                samples.append((img, idx))
    return samples


def main() -> int:
    args = parse_args()
    if not args.weights.is_file():
        print(f"ERROR: no existen los pesos {args.weights}")
        return 1

    split_dir = DATASET_CLASSIFIER_ROOT / args.split
    samples = collect_test_samples(split_dir)
    if not samples:
        print(f"ERROR: no se encontraron muestras en {split_dir}")
        return 1
    print(f"Evaluando {len(samples)} muestras en {split_dir}...")

    from ultralytics import YOLO

    model = YOLO(str(args.weights))
    y_true: list[int] = []
    y_pred: list[int] = []

    for img_path, true_idx in samples:
        result = model.predict(str(img_path), imgsz=args.imgsz, verbose=False)[0]
        names = result.names
        probs = result.probs
        pred_idx_in_model = int(probs.top1)
        pred_name = names[pred_idx_in_model]
        if pred_name in CLASSES:
            pred_idx = CLASSES.index(pred_name)
        else:
            pred_idx = -1
        y_true.append(true_idx)
        y_pred.append(pred_idx)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    report = classification_report(y_true, y_pred, target_names=CLASSES, output_dict=True, zero_division=0)
    df = pd.DataFrame(report).transpose()
    csv_path = TABLES_DIR / "metrics_cls.csv"
    df.to_csv(csv_path)
    print(f"OK: metricas guardadas en {csv_path}")
    print(classification_report(y_true, y_pred, target_names=CLASSES, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES))))
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title(f"Matriz de confusion ({args.split})")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    png_path = FIGURES_DIR / "confusion_matrix_cls.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"OK: matriz de confusion en {png_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
