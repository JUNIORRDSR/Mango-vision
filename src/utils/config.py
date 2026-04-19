"""Configuración central del proyecto Mango-vision.

Todas las rutas y constantes viven aquí para que los scripts individuales
no hardcodeen paths. Importar desde otros módulos como:

    from src.utils.config import MODEL_CLS, DATASET_CLASSIFIER_ROOT
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# --- Datasets ---------------------------------------------------------------
# MangoDHDS (local, versionado en el repo). Labels son bboxes sintéticos de
# imagen completa, por lo cual solo se usa para clasificación.
DATASET_MANGODHDS_ROOT = REPO_ROOT / "DatasetMango_YOLO"

# Versión deduplicada y re-splitteada de MangoDHDS. Se genera con
# dedup_and_resplit.py porque el split original tiene near-duplicates
# cross-split que inflan las métricas de validación (ver README).
# Ésta es la fuente real para entrenar el clasificador.
DATASET_MANGODHDS_CLEAN = REPO_ROOT / "DatasetMango_YOLO_clean"

# Espejo regenerado por make_cls_layout.py con layout de clasificación
# esperado por Ultralytics: <root>/<split>/<class_name>/*.jpg
DATASET_CLASSIFIER_ROOT = REPO_ROOT / "DatasetMango_YOLO_cls"

# Dataset externo de Roboflow (Luigui Cerna) para el detector genérico de
# mango. Se descarga con download_roboflow.py; no se versiona en git.
DATASET_DETECTOR_DIR = REPO_ROOT / "DatasetMango_Detector"
DATASET_DETECTOR_YAML = DATASET_DETECTOR_DIR / "data.yaml"

# Test externo con imágenes propias anotadas por el usuario en Roboflow.
DATASET_EXTERNAL_TEST_DIR = REPO_ROOT / "DatasetMango_Propias"
DATASET_EXTERNAL_TEST_YAML = DATASET_EXTERNAL_TEST_DIR / "external_test.yaml"

# --- Modelos y pesos --------------------------------------------------------
MODEL_CLS = "yolo11n-cls.pt"
MODEL_DET = "yolo11n.pt"

MODELS_DIR = REPO_ROOT / "models"
WEIGHTS_CLS = MODELS_DIR / "clasificador_enfermedad.pt"
WEIGHTS_DET = MODELS_DIR / "detector_mango.pt"

# --- Hiperparámetros --------------------------------------------------------
IMG_SIZE = 640
IMG_SIZE_CLS = 224
EPOCHS = 100
EPOCHS_CLS = 50
BATCH = 16
BATCH_CLS = 32
SEED = 42
PATIENCE = 20
PATIENCE_CLS = 15

# --- Umbrales de inferencia -------------------------------------------------
CONF_DETECTION = 0.5
CONF_CLASSIFICATION = 0.4

# --- Clases -----------------------------------------------------------------
CLASSES = [
    "Saludable",
    "Antracnosis",
    "Cancro_bacteriano",
    "Costras",
    "Podredumbre_Extremo_tallo",
]

# BGR para OpenCV (no RGB).
COLORS_BY_CLASS = {
    "Saludable": (0, 200, 0),
    "Antracnosis": (0, 0, 255),
    "Cancro_bacteriano": (0, 255, 255),
    "Costras": (0, 140, 255),
    "Podredumbre_Extremo_tallo": (200, 0, 200),
}

# --- Roboflow (trazabilidad de fuente) --------------------------------------
ROBOFLOW_WORKSPACE = "luigui-andre-cerna-grados-dpsrr"
ROBOFLOW_PROJECT = "clasificacion-de-mangos"
ROBOFLOW_FORMAT = "yolov8"

# --- Reportes ---------------------------------------------------------------
REPORTS_DIR = REPO_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"
