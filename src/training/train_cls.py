"""Entrena el clasificador de enfermedad (Modelo 2) sobre MangoDHDS.

Flujo:
    1. Si `DatasetMango_YOLO_cls/` no existe, lo genera con make_cls_layout.
    2. Entrena `yolo11n-cls.pt` sobre ese layout.
    3. Copia el best.pt a `models/clasificador_enfermedad.pt`.

Uso:
    python src/training/train_cls.py
    python src/training/train_cls.py --epochs 2 --batch 4 --device cpu   # smoke test
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.training.make_cls_layout import main as build_cls_layout
from src.utils.config import (
    BATCH_CLS,
    DATASET_CLASSIFIER_ROOT,
    EPOCHS_CLS,
    IMG_SIZE_CLS,
    MODEL_CLS,
    MODELS_DIR,
    PATIENCE_CLS,
    SEED,
    WEIGHTS_CLS,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=EPOCHS_CLS)
    p.add_argument("--batch", type=int, default=BATCH_CLS)
    p.add_argument("--imgsz", type=int, default=IMG_SIZE_CLS)
    p.add_argument("--device", default="auto", help="'auto' | 'cpu' | '0' | '0,1'")
    p.add_argument("--model", default=MODEL_CLS)
    p.add_argument("--patience", type=int, default=PATIENCE_CLS)
    p.add_argument("--name", default="train_cls")
    return p.parse_args()


def ensure_layout() -> None:
    train_dir = DATASET_CLASSIFIER_ROOT / "train"
    if not train_dir.is_dir() or not any(train_dir.iterdir()):
        print("Layout de clasificacion no encontrado, generando...")
        build_cls_layout()
    else:
        print(f"Layout existente en {DATASET_CLASSIFIER_ROOT}, se reutiliza.")


def main() -> int:
    args = parse_args()
    ensure_layout()

    from ultralytics import YOLO

    model = YOLO(args.model)
    results = model.train(
        data=str(DATASET_CLASSIFIER_ROOT),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=(None if args.device == "auto" else args.device),
        seed=SEED,
        patience=args.patience,
        name=args.name,
        optimizer="auto",
    )

    save_dir = Path(results.save_dir) if hasattr(results, "save_dir") else None
    if save_dir is None:
        save_dir = Path("runs/classify") / args.name
    best = save_dir / "weights" / "best.pt"
    if best.is_file():
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, WEIGHTS_CLS)
        print(f"OK: pesos copiados a {WEIGHTS_CLS}")
    else:
        print(f"WARN: no se encontro {best}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
