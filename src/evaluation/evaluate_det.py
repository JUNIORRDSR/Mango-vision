"""Evalua el detector generico de mango.

Usa `model.val()` de Ultralytics sobre el dataset que se pase por --data.
Acepta tanto el test interno del dataset de Luigui Cerna como el test
externo con imagenes propias (cuando el usuario lo tenga listo).

Uso:
    python src/evaluation/evaluate_det.py
    python src/evaluation/evaluate_det.py --data DatasetMango_Propias/external_test.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import (
    DATASET_DETECTOR_YAML,
    IMG_SIZE,
    TABLES_DIR,
    WEIGHTS_DET,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", type=Path, default=WEIGHTS_DET)
    p.add_argument("--data", type=Path, default=DATASET_DETECTOR_YAML)
    p.add_argument("--imgsz", type=int, default=IMG_SIZE)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--out", type=Path, default=TABLES_DIR / "metrics_det.csv")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.weights.is_file():
        print(f"ERROR: no existen los pesos {args.weights}")
        return 1
    if not args.data.is_file():
        print(f"ERROR: no existe {args.data}")
        return 1

    from ultralytics import YOLO

    model = YOLO(str(args.weights))
    metrics = model.val(data=str(args.data), imgsz=args.imgsz, split=args.split)

    box = metrics.box
    row = {
        "data": str(args.data),
        "split": args.split,
        "mAP50": float(box.map50),
        "mAP50-95": float(box.map),
        "precision": float(box.mp),
        "recall": float(box.mr),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.is_file():
        df = pd.read_csv(args.out)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(args.out, index=False)
    print(f"OK: metricas guardadas/acumuladas en {args.out}")
    for k, v in row.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
