"""Entrena el detector generico de mango (Modelo 1) sobre el dataset de
Luigui Cerna (Roboflow) ya colapsado a una sola clase.

Requiere:
    1. `python src/data_prep/download_roboflow.py`
    2. `python src/data_prep/collapse_detector_classes.py`

Escribe `data.yaml` con paths absolutos en runtime antes de entrenar para
que funcione tanto local como en Kaggle sin editar el archivo versionado.

Uso:
    python src/training/train_det_generic.py
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import (
    BATCH,
    DATASET_DETECTOR_DIR,
    DATASET_DETECTOR_YAML,
    EPOCHS,
    IMG_SIZE,
    MODEL_DET,
    MODELS_DIR,
    PATIENCE,
    SEED,
    WEIGHTS_DET,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch", type=int, default=BATCH)
    p.add_argument("--imgsz", type=int, default=IMG_SIZE)
    p.add_argument("--device", default="auto")
    p.add_argument("--model", default=MODEL_DET)
    p.add_argument("--patience", type=int, default=PATIENCE)
    p.add_argument("--name", default="train_det")
    return p.parse_args()


def write_runtime_yaml() -> Path:
    """Reescribe data.yaml con path absoluto al dataset. Idempotente."""
    if not DATASET_DETECTOR_DIR.is_dir():
        raise FileNotFoundError(
            f"No existe {DATASET_DETECTOR_DIR}. Correr download_roboflow.py y collapse_detector_classes.py primero."
        )
    content = (
        f"path: {DATASET_DETECTOR_DIR.resolve()}\n"
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n"
        "nc: 1\n"
        "names: ['mango']\n"
    )
    DATASET_DETECTOR_YAML.write_text(content)
    return DATASET_DETECTOR_YAML


def main() -> int:
    args = parse_args()
    yaml_path = write_runtime_yaml()
    print(f"data.yaml preparado: {yaml_path}")

    from ultralytics import YOLO

    model = YOLO(args.model)
    results = model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=(None if args.device == "auto" else args.device),
        seed=SEED,
        patience=args.patience,
        cos_lr=True,
        optimizer="auto",
        name=args.name,
    )

    save_dir = Path(results.save_dir) if hasattr(results, "save_dir") else None
    if save_dir is None:
        save_dir = Path("runs/detect") / args.name
    best = save_dir / "weights" / "best.pt"
    if best.is_file():
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, WEIGHTS_DET)
        print(f"OK: pesos copiados a {WEIGHTS_DET}")
    else:
        print(f"WARN: no se encontro {best}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
