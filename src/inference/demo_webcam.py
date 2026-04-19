"""Demo frente al jurado: webcam + cascada con defaults de presentacion.

Wrapper de `infer_cascade.py` con los defaults del proyecto. Se pueden
sobreescribir por CLI.

Uso:
    python src/inference/demo_webcam.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import CONF_CLASSIFICATION, CONF_DETECTION, WEIGHTS_CLS, WEIGHTS_DET


def main() -> int:
    # Reutilizamos directamente infer_cascade con los defaults ya seteados.
    from src.inference import infer_cascade

    # Construimos argv minimo para argparse.
    base_argv = sys.argv[1:]
    injected = []
    if "--source" not in base_argv:
        injected += ["--source", "0"]
    if "--det-weights" not in base_argv:
        injected += ["--det-weights", str(WEIGHTS_DET)]
    if "--cls-weights" not in base_argv:
        injected += ["--cls-weights", str(WEIGHTS_CLS)]
    if "--conf-det" not in base_argv:
        injected += ["--conf-det", str(CONF_DETECTION)]
    if "--conf-cls" not in base_argv:
        injected += ["--conf-cls", str(CONF_CLASSIFICATION)]
    sys.argv = [sys.argv[0]] + injected + base_argv
    return infer_cascade.main()


if __name__ == "__main__":
    sys.exit(main())
