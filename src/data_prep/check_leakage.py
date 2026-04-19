"""Detecta fuga de datos entre splits de un dataset YOLO usando perceptual hash.

Compara las imágenes de train/valid/test por pares de splits. Si encuentra
pares con distancia Hamming < 5 sobre `imagehash.phash`, imprime la lista
y sale con código 1. No intenta arreglar nada automáticamente.

Por defecto analiza DatasetMango_YOLO/. Para validar el dataset limpio
generado por dedup_and_resplit.py:

    python src/data_prep/check_leakage.py --root DatasetMango_YOLO_clean
"""

from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import imagehash
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import DATASET_MANGODHDS_ROOT, REPO_ROOT

HASH_THRESHOLD = 5
SPLITS = ("train", "valid", "test")
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def hash_split(root: Path, split: str) -> dict[Path, imagehash.ImageHash]:
    images_dir = root / split / "images"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"No existe {images_dir}")
    hashes: dict[Path, imagehash.ImageHash] = {}
    for p in sorted(images_dir.iterdir()):
        if p.suffix.lower() not in IMG_EXT:
            continue
        with Image.open(p) as im:
            hashes[p] = imagehash.phash(im)
    return hashes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=DATASET_MANGODHDS_ROOT,
        help="Carpeta raiz del dataset a analizar (default: DatasetMango_YOLO).",
    )
    args = parser.parse_args()
    root = args.root if args.root.is_absolute() else (REPO_ROOT / args.root).resolve()

    print(f"Analizando {root}...")
    per_split: dict[str, dict[Path, imagehash.ImageHash]] = {}
    for split in SPLITS:
        per_split[split] = hash_split(root, split)
        print(f"  {split}: {len(per_split[split])} imagenes hasheadas")

    duplicates: list[tuple[str, str, Path, Path, int]] = []
    for a, b in combinations(SPLITS, 2):
        for path_a, h_a in per_split[a].items():
            for path_b, h_b in per_split[b].items():
                dist = h_a - h_b
                if dist < HASH_THRESHOLD:
                    duplicates.append((a, b, path_a, path_b, dist))

    if duplicates:
        print(f"\nFUGA DETECTADA: {len(duplicates)} pares sospechosos")
        print(f"(distancia Hamming < {HASH_THRESHOLD})\n")
        for a, b, pa, pb, d in duplicates:
            print(f"  [{a} <-> {b}] dist={d}  {pa.name}  <->  {pb.name}")
        print("\nNo se realiza ninguna accion automatica. Revisar manualmente.")
        return 1

    print(f"\nOK: no se encontraron duplicados entre splits (umbral {HASH_THRESHOLD}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
