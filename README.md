# Mango-vision

Sistema de vision por computadora para detectar enfermedades en fruto de mango mediante una arquitectura en cascada de dos modelos YOLO. Sub-proyecto del Semillero TI de la Universidad Libre seccional Barranquilla.

> Guia paso a paso end-to-end (setup, datos, entrenamiento, parametros, errores comunes): [`docs/WORKFLOW.md`](docs/WORKFLOW.md).

## Arquitectura en cascada

```
  Frame (webcam / video / imagen)
            |
            v
  +----------------------+      Modelo 1 (deteccion)
  | detector_mango.pt    |  --> bboxes de mangos
  +----------------------+      yolo11n sobre dataset de Luigui Cerna
            |
            v   crops
  +----------------------+      Modelo 2 (clasificacion)
  | clasificador_enf.pt  |  --> diagnostico por mango
  +----------------------+      yolo11n-cls sobre MangoDHDS
            |
            v
  Overlay: bbox + <clase_enfermedad>: <confianza>%
```

Cada recorte de mango se redimensiona y pasa al clasificador. El color del bbox depende de la clase predicha por el Modelo 2 (verde si `Saludable`, otros colores por enfermedad — ver `src/utils/config.py`).

## Estructura del repositorio

```
Mango-vision/
|-- DatasetMango_YOLO/            # MangoDHDS local raw (versionado)
|-- DatasetMango_YOLO_clean/      # MangoDHDS deduplicado y re-splitteado (regenerado, no versionado)
|-- DatasetMango_YOLO_cls/        # layout de clasificacion (regenerado, no versionado)
|-- DatasetMango_Detector/        # dataset de Luigui Cerna (descargado, no versionado)
|-- DatasetMango_Propias/         # test externo anotado por el usuario (no versionado)
|-- models/                       # pesos finales (no versionado)
|-- notebooks/
|   |-- 01_exploracion_dataset.ipynb
|   |-- 02_entrenamiento_cls.ipynb
|   |-- 03_entrenamiento_det.ipynb
|   `-- 04_evaluacion.ipynb
|-- reports/
|   |-- figures/
|   `-- tables/
|-- src/
|   |-- data_prep/                # descarga, limpieza, colapso de clases, leakage
|   |-- training/                 # train_cls, train_det_generic, make_cls_layout
|   |-- evaluation/               # evaluate_cls, evaluate_det, explain_cam
|   |-- inference/                # infer_cascade, demo_webcam
|   `-- utils/config.py
|-- requirements.txt
`-- README.md
```

## Datasets

**MangoDHDS** (local, versionado). Fuente: Mendeley, licencia CC BY 4.0. Procesado a formato YOLO en `DatasetMango_YOLO/` con 5 clases: `Saludable`, `Antracnosis`, `Cancro_bacteriano`, `Costras`, `Podredumbre_Extremo_tallo`. Todas las anotaciones son bboxes de imagen completa (`cls 0.5 0.5 1.0 1.0`), por eso se usa para clasificacion, no deteccion.

### Fuga de datos en MangoDHDS y re-split

Al correr `src/data_prep/check_leakage.py` sobre `DatasetMango_YOLO/` (umbral `imagehash.phash` Hamming < 5) se detectaron **76 pares near-duplicate cross-split** entre `train`, `valid` y `test`. La mayoria con distancia `0`, es decir imagenes perceptualmente identicas repartidas entre splits con nombres distintos (por ejemplo `He105.jpg` en train y `He70.jpg` en valid apuntan a la misma foto).

Entrenar directamente sobre ese split invalida la evaluacion: el modelo memoriza imagenes vistas en entrenamiento y las vuelve a ver en validacion, de modo que las metricas reportadas no reflejan generalizacion.

**Mitigacion aplicada.** `src/data_prep/dedup_and_resplit.py` regenera el dataset:

1. Hashea todas las imagenes con `imagehash.phash`.
2. Construye clusters con union-find conectando pares con Hamming < 5.
3. Trata cada cluster como unidad atomica y hace **stratified split por clase** (70 / 15 / 15) con `seed=42` sobre los clusters.
4. Escribe la version limpia en `DatasetMango_YOLO_clean/` con la misma estructura `split/{images,labels}/`.

El clasificador se entrena sobre `DatasetMango_YOLO_clean/` (configurado en `src/utils/config.py` via `DATASET_MANGODHDS_CLEAN` y consumido por `src/training/make_cls_layout.py`). `DatasetMango_YOLO/` permanece como fuente historica no usada en entrenamiento.

**Flujo de preparacion del clasificador:**

```bash
python src/data_prep/check_leakage.py                                  # 1. reporta fuga sobre el split original
python src/data_prep/dedup_and_resplit.py                              # 2. genera DatasetMango_YOLO_clean
python src/data_prep/check_leakage.py --root DatasetMango_YOLO_clean   # 3. valida 0 pares cross-split
```

El paso 3 debe imprimir `OK: no se encontraron duplicados entre splits` y salir con codigo 0.

**Clasificacion de Mangos** (Luigui Cerna, Roboflow Universe). Licencia CC BY 4.0. ~1334 imagenes en campo abierto. Las dos clases originales (`Mango Exportable`, `Mango Industrial`) se colapsan a una sola clase `mango` para el Modelo 1.

**Test externo propio** (futuro). 50-80 imagenes tomadas en la finca de Sabanalarga, anotadas en Roboflow con clase unica `mango`. Se integra con `integrate_external_test.py` cuando este listo.

Citaciones BibTeX recomendadas:

```bibtex
@dataset{mangodhds,
  title  = {MangoDHDS: Mango Disease Hybrid Dataset},
  author = {Mendeley Data contributors},
  note   = {Licencia CC BY 4.0},
  url    = {https://data.mendeley.com/}
}

@dataset{cerna_mango_roboflow,
  title  = {Clasificacion de Mangos},
  author = {Cerna Grados, Luigui Andre},
  note   = {Roboflow Universe, licencia CC BY 4.0},
  url    = {https://universe.roboflow.com/luigui-andre-cerna-grados-dpsrr/clasificacion-de-mangos}
}
```

## Setup

```bash
python -m venv .venv
# Linux/Mac
source .venv/bin/activate
# Windows
# .venv\Scripts\activate

pip install -r requirements.txt
```

Variable de entorno para descargar el dataset del detector:

```bash
# Linux/Mac
export ROBOFLOW_API_KEY="tu_key"
# Windows CMD
set ROBOFLOW_API_KEY=tu_key
# PowerShell
$env:ROBOFLOW_API_KEY = "tu_key"
```

## Descargar el dataset del detector

```bash
python src/data_prep/download_roboflow.py
python src/data_prep/collapse_detector_classes.py
```

La primera descarga deja `DatasetMango_Detector/` con las clases originales; la segunda reemplaza todos los `class_id` por `0` y reescribe `data.yaml` con `nc: 1, names: ['mango']`. Idempotente.

## Entrenar

```bash
# Clasificador (Modelo 2) sobre MangoDHDS
python src/training/train_cls.py --epochs 50 --batch 32 --imgsz 224

# Detector generico (Modelo 1) sobre dataset de Luigui Cerna
python src/training/train_det_generic.py --epochs 100 --batch 16 --imgsz 640
```

Los pesos finales se copian a `models/clasificador_enfermedad.pt` y `models/detector_mango.pt`.

## Evaluar

```bash
python src/evaluation/evaluate_cls.py
python src/evaluation/evaluate_det.py
# Con test externo propio (cuando exista):
python src/evaluation/evaluate_det.py --data DatasetMango_Propias/external_test.yaml
```

Genera:

- `reports/figures/confusion_matrix_cls.png`
- `reports/tables/metrics_cls.csv`
- `reports/tables/metrics_det.csv`

## Explicabilidad

```bash
python src/evaluation/explain_cam.py --n 12
# o solo para una clase
python src/evaluation/explain_cam.py --n 12 --class Antracnosis
```

Genera mapas EigenCAM en `reports/figures/cam/<clase>/`.

## Demo en vivo

```bash
python src/inference/demo_webcam.py
# equivalente a:
python src/inference/infer_cascade.py \
    --det-weights models/detector_mango.pt \
    --cls-weights models/clasificador_enfermedad.pt \
    --source 0
```

Salir con `q`. Para grabar la demo: agregar `--save-video runs/cascade/demo.mp4`.

## Integrar test externo anotado

```bash
# Descomprimir export YOLOv8 de Roboflow en DatasetMango_Propias/
python src/data_prep/integrate_external_test.py
python src/evaluation/evaluate_det.py --data DatasetMango_Propias/external_test.yaml
```

## Muestreo de frames desde video

```bash
python src/data_prep/video_to_frames.py --video captura.mp4 --output data/frames/ --every 15
```

Muestrea cada 15 frames y descarta duplicados por perceptual hash. Util para alimentar el dataset propio a partir de videos en finca.

## Limitaciones conocidas

- **Bboxes sinteticos en MangoDHDS.** Todas las labels son imagen completa. Por eso el Modelo 2 se entrena en modo clasificacion, no deteccion.
- **Fuga cross-split en el MangoDHDS original.** 76 pares de imagenes near-duplicate repartidas entre train/valid/test. Se mitiga regenerando el dataset con `dedup_and_resplit.py` (ver seccion "Fuga de datos en MangoDHDS y re-split"). El entrenamiento del clasificador se corre siempre sobre `DatasetMango_YOLO_clean/`.
- **Sesgo out-of-distribution.** Los datasets publicos estan tomados en condiciones controladas. En campo abierto la precision bajara hasta que haya fine-tuning con datos propios.
- **Detector de una sola clase.** El Modelo 1 solo localiza mangos; el diagnostico depende enteramente del Modelo 2 y de los crops.
- **Confianzas heterogeneas entre modelos.** `--conf-det` y `--conf-cls` se ajustan por separado en `config.py`.

## Trabajo futuro

- Integracion RGB-D con Intel RealSense D435i para estimar distancia y tamano real del fruto.
- Captura y anotacion propia en la finca de Sabanalarga (Atlantico).
- Enriquecer con dataset especifico de podredumbre severa postcosecha.
- Despliegue edge en Raspberry Pi 4 con modelo cuantizado.

## Autores

Daniela Villa Bastidas, Jorge Solano Romero. Semillero TI, Universidad Libre seccional Barranquilla.
