"""Descarga el dataset de Luigui Cerna desde Roboflow Universe.

Requiere que `ROBOFLOW_API_KEY` este en el entorno. La API key nunca se
hardcodea ni se versiona.

Linux/Mac:   export ROBOFLOW_API_KEY="tu_key"
Windows CMD: set ROBOFLOW_API_KEY=tu_key
PowerShell:  $env:ROBOFLOW_API_KEY = "tu_key"

Uso:
    python src/data_prep/download_roboflow.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import (
    DATASET_DETECTOR_DIR,
    ROBOFLOW_FORMAT,
    ROBOFLOW_PROJECT,
    ROBOFLOW_WORKSPACE,
)


def main() -> int:
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        print("ERROR: la variable de entorno ROBOFLOW_API_KEY no esta definida.")
        print("Configurar:")
        print('  Linux/Mac:   export ROBOFLOW_API_KEY="tu_key"')
        print("  Windows CMD: set ROBOFLOW_API_KEY=tu_key")
        print('  PowerShell:  $env:ROBOFLOW_API_KEY = "tu_key"')
        return 1

    try:
        from roboflow import Roboflow
    except ImportError:
        print("ERROR: instalar dependencias primero (pip install -r requirements.txt)")
        return 1

    DATASET_DETECTOR_DIR.mkdir(parents=True, exist_ok=True)

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
    versions = project.versions()
    if not versions:
        print(f"ERROR: el proyecto {ROBOFLOW_PROJECT} no tiene versiones publicadas.")
        return 1
    latest = versions[0]
    print(f"Descargando version {latest.version} de {ROBOFLOW_PROJECT} en formato {ROBOFLOW_FORMAT}...")
    latest.download(ROBOFLOW_FORMAT, location=str(DATASET_DETECTOR_DIR))
    print(f"OK: dataset descargado en {DATASET_DETECTOR_DIR}")
    print("Siguiente paso: python src/data_prep/collapse_detector_classes.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
