"""Colapsa todas las clases del dataset detector a una sola clase 'mango'.

Razon: el dataset original tiene dos clases ('Mango Exportable' y 'Mango
Industrial') con desbalance severo (~1298 vs 36) y la arquitectura en
cascada delega el diagnostico al Modelo 2. Para el Modelo 1 solo importa
localizar un mango.

Reescribe `class_id = 0` en todos los .txt de train/valid/test/labels y
regenera `data.yaml` con `nc: 1, names: ['mango']`. Idempotente: si ya
esta colapsado, no rompe ni duplica trabajo.

Uso:
    python src/data_prep/collapse_detector_classes.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import DATASET_DETECTOR_DIR

SPLITS = ("train", "valid", "test")


def remap_label_file(path: Path) -> tuple[int, bool]:
    """Reescribe un .txt. Devuelve (lineas_modificadas, archivo_modificado)."""
    original = path.read_text().splitlines()
    modified = 0
    new_lines = []
    for line in original:
        s = line.strip()
        if not s:
            new_lines.append(line)
            continue
        parts = s.split()
        if parts[0] != "0":
            parts[0] = "0"
            modified += 1
        new_lines.append(" ".join(parts))
    if modified > 0:
        path.write_text("\n".join(new_lines) + ("\n" if original and original[-1] == "" else ""))
    return modified, modified > 0


def write_data_yaml() -> Path:
    """Reescribe data.yaml con paths absolutos y clase unica."""
    yaml_path = DATASET_DETECTOR_DIR / "data.yaml"
    content = (
        f"path: {DATASET_DETECTOR_DIR.resolve()}\n"
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n"
        "nc: 1\n"
        "names: ['mango']\n"
    )
    yaml_path.write_text(content)
    return yaml_path


def main() -> int:
    if not DATASET_DETECTOR_DIR.is_dir():
        print(f"ERROR: no existe {DATASET_DETECTOR_DIR}")
        print("Ejecutar primero: python src/data_prep/download_roboflow.py")
        return 1

    total_labels = 0
    total_files_touched = 0
    for split in SPLITS:
        labels_dir = DATASET_DETECTOR_DIR / split / "labels"
        if not labels_dir.is_dir():
            print(f"WARN: no existe {labels_dir}, se omite.")
            continue
        for txt in sorted(labels_dir.glob("*.txt")):
            remapped, touched = remap_label_file(txt)
            total_labels += remapped
            if touched:
                total_files_touched += 1

    yaml_path = write_data_yaml()
    print(f"OK: {total_labels} labels remapped, {total_files_touched} files updated.")
    print(f"data.yaml reescrito en: {yaml_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
