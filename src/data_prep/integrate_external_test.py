"""Valida y registra el test externo (imagenes propias anotadas).

Asume que el usuario exporto desde Roboflow un zip YOLOv8 y lo descomprimio
en `DatasetMango_Propias/`, con la estructura esperada:

    DatasetMango_Propias/
        test/images/*.jpg
        test/labels/*.txt        # class_id siempre 0 (mango)

Este script valida esa estructura, verifica que todas las labels tengan
unicamente clase 0, y genera `external_test.yaml` apuntando al test externo.
Es idempotente y NO modifica imagenes ni labels.

Uso (no ejecutar hasta que el usuario tenga las imagenes anotadas):
    python src/data_prep/integrate_external_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import DATASET_EXTERNAL_TEST_DIR, DATASET_EXTERNAL_TEST_YAML


def main() -> int:
    if not DATASET_EXTERNAL_TEST_DIR.is_dir():
        print(f"ERROR: no existe {DATASET_EXTERNAL_TEST_DIR}")
        print("Exportar el dataset anotado desde Roboflow (formato YOLOv8)")
        print("y descomprimirlo en esa ruta.")
        return 1

    images_dir = DATASET_EXTERNAL_TEST_DIR / "test" / "images"
    labels_dir = DATASET_EXTERNAL_TEST_DIR / "test" / "labels"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        print("ERROR: estructura invalida. Se espera test/images y test/labels.")
        return 1

    bad_classes: list[Path] = []
    n_labels = 0
    for txt in sorted(labels_dir.glob("*.txt")):
        for line in txt.read_text().splitlines():
            s = line.strip()
            if not s:
                continue
            n_labels += 1
            if s.split()[0] != "0":
                bad_classes.append(txt)
                break

    if bad_classes:
        print(f"ERROR: {len(bad_classes)} archivos con clases distintas de 0 (mango):")
        for p in bad_classes[:20]:
            print(f"  {p.name}")
        return 1

    content = (
        f"path: {DATASET_EXTERNAL_TEST_DIR.resolve()}\n"
        "train: test/images\n"
        "val: test/images\n"
        "test: test/images\n"
        "nc: 1\n"
        "names: ['mango']\n"
    )
    DATASET_EXTERNAL_TEST_YAML.write_text(content)
    print(f"OK: {n_labels} labels validadas, yaml escrito en {DATASET_EXTERNAL_TEST_YAML}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
