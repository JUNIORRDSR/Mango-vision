"""Dedup y re-split de MangoDHDS para eliminar fuga entre train/valid/test.

Problema detectado por check_leakage.py: el split original de MangoDHDS
tiene 76 pares near-duplicate cross-split (imagenes identicas o casi
identicas repartidas entre train, valid y test con nombres distintos).
Entrenar asi infla las metricas de validacion porque el modelo memoriza
imagenes ya vistas. Para una evaluacion honesta hay que rehacer el split
deduplicando primero.

Este script:

1. Reune todas las imagenes de los tres splits originales en
   DatasetMango_YOLO/.
2. Calcula perceptual hash (phash) de cada una.
3. Construye clusters con union-find conectando pares con distancia
   Hamming < 5 (mismo criterio que check_leakage.py).
4. Asigna cada cluster a su clase mayoritaria (por primer token de
   label) y hace stratified split de clusters por clase con seed fija:
   70 % train, 15 % valid, 15 % test.
5. Escribe el dataset limpio en DatasetMango_YOLO_clean/ con la misma
   estructura que espera make_cls_layout.py (split/images, split/labels).
6. Copia data.yaml.

Al tratar el cluster entero como unidad atomica, ninguna imagen que
tenga un near-duplicate termina en dos splits a la vez.

Uso:
    python src/data_prep/dedup_and_resplit.py

Validacion posterior:
    python src/data_prep/check_leakage.py --root DatasetMango_YOLO_clean
"""

from __future__ import annotations

import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import imagehash
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import (
    CLASSES,
    DATASET_MANGODHDS_CLEAN,
    DATASET_MANGODHDS_ROOT,
    SEED,
)

HASH_THRESHOLD = 5
SPLITS_IN = ("train", "valid", "test")
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
RATIO_TRAIN = 0.70
RATIO_VALID = 0.15


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def collect_items() -> list[tuple[Path, Path, int]]:
    items: list[tuple[Path, Path, int]] = []
    for split in SPLITS_IN:
        images_dir = DATASET_MANGODHDS_ROOT / split / "images"
        labels_dir = DATASET_MANGODHDS_ROOT / split / "labels"
        if not images_dir.is_dir() or not labels_dir.is_dir():
            raise FileNotFoundError(f"Falta estructura en {DATASET_MANGODHDS_ROOT / split}")
        for p in sorted(images_dir.iterdir()):
            if p.suffix.lower() not in IMG_EXT:
                continue
            label = labels_dir / f"{p.stem}.txt"
            if not label.is_file():
                print(f"WARN: sin label {label.name}, se omite {p.name}")
                continue
            lines = [ln for ln in label.read_text().splitlines() if ln.strip()]
            if not lines:
                print(f"WARN: label vacia {label.name}, se omite {p.name}")
                continue
            cls_id = int(lines[0].split()[0])
            if not 0 <= cls_id < len(CLASSES):
                print(f"WARN: class_id {cls_id} fuera de rango en {label.name}")
                continue
            items.append((p, label, cls_id))
    return items


def main() -> int:
    if DATASET_MANGODHDS_CLEAN.exists():
        print(f"ERROR: {DATASET_MANGODHDS_CLEAN} ya existe.")
        print("Borralo manualmente antes de re-ejecutar para evitar mezclar splits.")
        return 1

    print(f"Leyendo {DATASET_MANGODHDS_ROOT}...")
    items = collect_items()
    print(f"Total imagenes con label valido: {len(items)}")

    print("Calculando phash de cada imagen...")
    hashes = []
    for img_path, _, _ in items:
        with Image.open(img_path) as im:
            hashes.append(imagehash.phash(im))

    print(f"Construyendo clusters union-find (phash Hamming < {HASH_THRESHOLD})...")
    uf = UnionFind(len(items))
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if hashes[i] - hashes[j] < HASH_THRESHOLD:
                uf.union(i, j)

    clusters: dict[int, list[int]] = defaultdict(list)
    for i in range(len(items)):
        clusters[uf.find(i)].append(i)

    print(f"Clusters formados: {len(clusters)}")
    print(f"Imagenes colapsables por near-duplicate: {len(items) - len(clusters)}")

    cluster_class: dict[int, int] = {}
    for root, members in clusters.items():
        counts: dict[int, int] = defaultdict(int)
        for m in members:
            counts[items[m][2]] += 1
        cluster_class[root] = max(counts, key=counts.get)

    by_class: dict[int, list[int]] = defaultdict(list)
    for root, cls_id in cluster_class.items():
        by_class[cls_id].append(root)

    rng = random.Random(SEED)
    split_assignment: dict[int, str] = {}
    print("\nStratified split por clase (sobre clusters, seed={}):".format(SEED))
    for cls_id in sorted(by_class.keys()):
        cluster_roots = by_class[cls_id]
        rng.shuffle(cluster_roots)
        n = len(cluster_roots)
        n_train = int(round(n * RATIO_TRAIN))
        n_valid = int(round(n * RATIO_VALID))
        train_roots = cluster_roots[:n_train]
        valid_roots = cluster_roots[n_train:n_train + n_valid]
        test_roots = cluster_roots[n_train + n_valid:]
        for r in train_roots:
            split_assignment[r] = "train"
        for r in valid_roots:
            split_assignment[r] = "valid"
        for r in test_roots:
            split_assignment[r] = "test"
        name = CLASSES[cls_id] if 0 <= cls_id < len(CLASSES) else f"cls{cls_id}"
        print(f"  [{name}] clusters={n}  train={len(train_roots)} valid={len(valid_roots)} test={len(test_roots)}")

    print(f"\nEscribiendo {DATASET_MANGODHDS_CLEAN}...")
    counts_written: dict[tuple[str, int], int] = defaultdict(int)
    for root, members in clusters.items():
        split = split_assignment[root]
        for m in members:
            img_path, label_path, cls_id = items[m]
            dst_img = DATASET_MANGODHDS_CLEAN / split / "images" / img_path.name
            dst_lbl = DATASET_MANGODHDS_CLEAN / split / "labels" / label_path.name
            dst_img.parent.mkdir(parents=True, exist_ok=True)
            dst_lbl.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img_path, dst_img)
            shutil.copy2(label_path, dst_lbl)
            counts_written[(split, cls_id)] += 1

    print("\nImagenes escritas por split y clase:")
    for split in ("train", "valid", "test"):
        total = 0
        for cls_id in range(len(CLASSES)):
            n = counts_written.get((split, cls_id), 0)
            total += n
            print(f"  [{split}] {CLASSES[cls_id]}: {n}")
        print(f"  [{split}] TOTAL: {total}\n")

    src_yaml = DATASET_MANGODHDS_ROOT / "data.yaml"
    if src_yaml.is_file():
        shutil.copy2(src_yaml, DATASET_MANGODHDS_CLEAN / "data.yaml")

    print(f"OK: dataset limpio en {DATASET_MANGODHDS_CLEAN}")
    print("Siguiente: python src/data_prep/check_leakage.py --root DatasetMango_YOLO_clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
