"""Genera mapas EigenCAM sobre N muestras del clasificador.

Salida: reports/figures/cam/<clase>/<nombre_imagen>.png

Uso:
    python src/evaluation/explain_cam.py --n 12
    python src/evaluation/explain_cam.py --weights runs/classify/train_cls/weights/best.pt --class Antracnosis
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import (
    CLASSES,
    DATASET_CLASSIFIER_ROOT,
    FIGURES_DIR,
    IMG_SIZE_CLS,
    WEIGHTS_CLS,
)

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", type=Path, default=WEIGHTS_CLS)
    p.add_argument("--split", default="test")
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--class", dest="only_class", default=None, help="Solo una clase.")
    p.add_argument("--imgsz", type=int, default=IMG_SIZE_CLS)
    return p.parse_args()


def pick_target_layer(model: torch.nn.Module) -> torch.nn.Module:
    """Selecciona una capa convolucional avanzada como target para CAM."""
    conv_layers = [m for m in model.modules() if isinstance(m, torch.nn.Conv2d)]
    if not conv_layers:
        raise RuntimeError("No se encontraron Conv2d en el modelo.")
    return conv_layers[-1]


def load_and_preprocess(path: Path, size: int) -> tuple[np.ndarray, torch.Tensor]:
    bgr = cv2.imread(str(path))
    bgr = cv2.resize(bgr, (size, size))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).contiguous()
    return rgb, tensor


def main() -> int:
    args = parse_args()
    if not args.weights.is_file():
        print(f"ERROR: no existen los pesos {args.weights}")
        return 1

    from ultralytics import YOLO

    yolo = YOLO(str(args.weights))
    torch_model = yolo.model
    torch_model.eval()
    target_layer = pick_target_layer(torch_model)

    split_dir = DATASET_CLASSIFIER_ROOT / args.split
    classes = [args.only_class] if args.only_class else CLASSES

    out_root = FIGURES_DIR / "cam"
    out_root.mkdir(parents=True, exist_ok=True)

    cam = EigenCAM(model=torch_model, target_layers=[target_layer])

    total = 0
    for cls_name in classes:
        cls_dir = split_dir / cls_name
        if not cls_dir.is_dir():
            continue
        out_cls = out_root / cls_name
        out_cls.mkdir(parents=True, exist_ok=True)
        images = [p for p in sorted(cls_dir.iterdir()) if p.suffix.lower() in IMG_EXT]
        for img_path in images[: args.n]:
            rgb, tensor = load_and_preprocess(img_path, args.imgsz)
            with torch.no_grad():
                grayscale = cam(input_tensor=tensor)[0]
            overlay = show_cam_on_image(rgb, grayscale, use_rgb=True)
            out_path = out_cls / f"{img_path.stem}_cam.png"
            plt.imsave(str(out_path), overlay)
            total += 1
    print(f"OK: {total} imagenes CAM generadas en {out_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
