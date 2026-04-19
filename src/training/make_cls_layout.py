"""Crea el layout de clasificacion que Ultralytics espera.

Ultralytics clasificacion (task='classify') requiere:

    <root>/<split>/<class_name>/<imagen>.jpg

MangoDHDS viene en formato deteccion YOLO (images/ + labels/) pero como
todos los bboxes son de imagen completa, cada imagen tiene exactamente
una clase. Este script lee `<split>/labels/*.txt`, toma el primer token
como class_id, y espeja `<split>/images/<nombre>.jpg` a
`DatasetMango_YOLO_cls/<split>/<class_name>/<nombre>.jpg`.

En Linux/Mac usa symlinks. En Windows usa shutil.copy2 (los symlinks ahi
requieren privilegios especiales).

Es idempotente: si el archivo destino ya existe, lo omite.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import CLASSES, DATASET_CLASSIFIER_ROOT, DATASET_MANGODHDS_ROOT

# En Ultralytics classify, el split de validacion se llama 'val' por defecto.
SPLIT_MAP = {"train": "train", "valid": "val", "test": "test"}
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

USE_SYMLINK = sys.platform != "win32"


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if USE_SYMLINK:
        dst.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dst)


def build_split(src_split: str, dst_split: str) -> int:
    src_images = DATASET_MANGODHDS_ROOT / src_split / "images"
    src_labels = DATASET_MANGODHDS_ROOT / src_split / "labels"
    if not src_images.is_dir() or not src_labels.is_dir():
        raise FileNotFoundError(f"Faltan carpetas en {DATASET_MANGODHDS_ROOT / src_split}")

    n = 0
    for img in sorted(src_images.iterdir()):
        if img.suffix.lower() not in IMG_EXT:
            continue
        label_path = src_labels / f"{img.stem}.txt"
        if not label_path.is_file():
            print(f"WARN: sin label {label_path.name}, se omite {img.name}")
            continue
        lines = [ln for ln in label_path.read_text().splitlines() if ln.strip()]
        if not lines:
            print(f"WARN: label vacia {label_path.name}, se omite {img.name}")
            continue
        cls_id = int(lines[0].split()[0])
        if not 0 <= cls_id < len(CLASSES):
            print(f"WARN: class_id {cls_id} fuera de rango en {label_path.name}")
            continue
        class_name = CLASSES[cls_id]
        dst = DATASET_CLASSIFIER_ROOT / dst_split / class_name / img.name
        link_or_copy(img, dst)
        n += 1
    return n


def main() -> int:
    DATASET_CLASSIFIER_ROOT.mkdir(parents=True, exist_ok=True)
    total = 0
    for src, dst in SPLIT_MAP.items():
        n = build_split(src, dst)
        print(f"  {src} -> {dst}: {n} imagenes")
        total += n
    print(f"OK: layout de clasificacion generado en {DATASET_CLASSIFIER_ROOT} ({total} total)")
    print(f"Metodo: {'symlink' if USE_SYMLINK else 'copia (Windows)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
