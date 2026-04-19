# Workflow — Mango-vision

Guía end-to-end del proyecto: desde clonar el repo hasta presentar la demo en vivo. Incluye el razonamiento detrás de cada decisión, los parámetros por defecto, cómo escalar el entrenamiento y qué hacer ante errores comunes.

Público: cualquier integrante del semillero que necesite reproducir, auditar o extender el pipeline.

---

## Índice

1. [Arquitectura en cascada](#1-arquitectura-en-cascada)
2. [Setup inicial](#2-setup-inicial)
3. [Datasets y su procedencia](#3-datasets-y-su-procedencia)
4. [Fase 1 — MangoDHDS: validar, deduplicar, re-splitear](#4-fase-1--mangodhds-validar-deduplicar-re-splitear)
5. [Fase 2 — Dataset del detector: descargar y colapsar](#5-fase-2--dataset-del-detector-descargar-y-colapsar)
6. [Fase 3 — Test externo propio](#6-fase-3--test-externo-propio)
7. [Fase 4 — Entrenamiento del clasificador (Modelo 2)](#7-fase-4--entrenamiento-del-clasificador-modelo-2)
8. [Fase 5 — Entrenamiento del detector (Modelo 1)](#8-fase-5--entrenamiento-del-detector-modelo-1)
9. [Fase 6 — Evaluación](#9-fase-6--evaluación)
10. [Fase 7 — Inferencia en cascada y demo](#10-fase-7--inferencia-en-cascada-y-demo)
11. [Parámetros: por qué son esos y cómo escalar](#11-parámetros-por-qué-son-esos-y-cómo-escalar)
12. [Errores comunes y diagnóstico](#12-errores-comunes-y-diagnóstico)
13. [Checklist final antes de defensa](#13-checklist-final-antes-de-defensa)

---

## 1. Arquitectura en cascada

Dos modelos YOLO especializados:

```
Frame (webcam / video / imagen)
       │
       ▼
┌──────────────────────┐
│ Modelo 1 — Detector  │  yolo11n sobre dataset de Luigui Cerna (Roboflow)
│ detector_mango.pt    │  Tarea: localizar mangos en escenas reales
└──────────────────────┘  Output: bboxes con clase única "mango"
       │
       ▼ crops (recorte por bbox)
┌──────────────────────┐
│ Modelo 2 — Clasif.   │  yolo11n-cls sobre MangoDHDS
│ clasificador_enf.pt  │  Tarea: diagnosticar enfermedad por fruto
└──────────────────────┘  Output: Saludable / Antracnosis / Cancro / Costras / Podredumbre
       │
       ▼
Overlay: bbox + "<clase>: <confianza>%"
```

**Por qué cascada y no un solo modelo multiclase:**

- MangoDHDS es en realidad un dataset de clasificación. Los bboxes del YOLO son sintéticos (imagen completa, `cls 0.5 0.5 1.0 1.0`). No sirve para entrenar detección.
- El dataset de Luigui Cerna tiene mangos en campo abierto pero sin anotaciones de enfermedad.
- Separar responsabilidades hace cada modelo más pequeño, rápido y auditable. Reemplazar el detector o el clasificador por una versión mejor no rompe el otro.

---

## 2. Setup inicial

### Requisitos

- Python 3.10 o superior.
- 4 GB de RAM libre para el smoke test en CPU. Entrenamiento real pide GPU.
- Git + cuenta en Roboflow (correo universitario).
- Windows 10/11, Linux o macOS. En Windows los scripts usan copias en vez de symlinks automáticamente.

### Pasos

```bash
git clone https://github.com/JUNIORRDSR/Mango-vision.git
cd Mango-vision

python -m venv .venv
# Linux / Mac
source .venv/bin/activate
# Windows PowerShell
.venv\Scripts\Activate.ps1
# Windows Git Bash
source .venv/Scripts/activate

pip install -r requirements.txt
```

### Variable de entorno para Roboflow

```bash
# Linux / Mac
export ROBOFLOW_API_KEY="tu_key"
# Windows PowerShell
$env:ROBOFLOW_API_KEY = "tu_key"
# Windows CMD
set ROBOFLOW_API_KEY=tu_key
```

Verificar: `echo $ROBOFLOW_API_KEY` (Linux/Git Bash) o `echo %ROBOFLOW_API_KEY%` (CMD) debe imprimir la key.

**Por qué variable de entorno y no hardcoded:** API keys en el repo = leak público. `download_roboflow.py` valida que la variable exista y aborta con mensaje claro si falta.

---

## 3. Datasets y su procedencia

| Dataset | Carpeta | Uso | Versionado |
|---|---|---|---|
| MangoDHDS (Mendeley) | `DatasetMango_YOLO/` | Modelo 2 — clasificación | Sí |
| MangoDHDS dedup | `DatasetMango_YOLO_clean/` | Modelo 2 — clasificación real | No (regenera) |
| Cls layout | `DatasetMango_YOLO_cls/` | Fuente que Ultralytics consume | No (regenera) |
| Clasificación Mangos (Luigui Cerna) | `DatasetMango_Detector/` | Modelo 1 — detección | No (descarga) |
| Test externo propio | `DatasetMango_Propias/` | Evaluación OOD del detector | No |

Licencia de ambos datasets públicos: **CC BY 4.0**. Citación en el README.

---

## 4. Fase 1 — MangoDHDS: validar, deduplicar, re-splitear

### Paso 1.1. Detectar fuga en el split original

```bash
python src/data_prep/check_leakage.py
```

Compara todas las imágenes de `train`, `valid`, `test` con perceptual hash (`imagehash.phash`) y reporta pares cuyo Hamming sea menor a 5 (duplicados o casi-duplicados).

**Resultado esperado en el MangoDHDS original:** 76 pares cross-split (mayoría distancia 0 = imágenes idénticas con nombres diferentes). El script sale con código 1.

**Por qué esto importa:** el clasificador entrenado sobre un split con leakage memoriza imágenes ya vistas. Las métricas de validación se vuelven espejismo y no predicen cómo se comportará el modelo en producción.

### Paso 1.2. Regenerar el dataset sin leakage

```bash
python src/data_prep/dedup_and_resplit.py
```

Qué hace, en orden:

1. Lee todas las imágenes de los tres splits originales.
2. Calcula `imagehash.phash` de cada una.
3. Construye clusters con union-find conectando pares con Hamming < 5.
4. Trata cada cluster como unidad atómica. Cada imagen con near-duplicate cae en un solo split.
5. Split estratificado por clase con `seed=42`: 70 % train, 15 % valid, 15 % test sobre los clusters.
6. Escribe `DatasetMango_YOLO_clean/{train,valid,test}/{images,labels}/`.

**Parámetros fijos (no cambiar sin razón):**

| Parámetro | Valor | Por qué |
|---|---|---|
| `HASH_THRESHOLD` | 5 | Mismo umbral que usa `check_leakage.py`, consistente |
| `RATIO_TRAIN` / `RATIO_VALID` | 0.70 / 0.15 | Estándar académico; el 15 % restante queda en test |
| `SEED` | 42 | Configurable en `src/utils/config.py`; fijo para reproducibilidad |

Si cambias el seed, documenta el cambio y re-corre todo el pipeline.

### Paso 1.3. Validar el dataset limpio

```bash
python src/data_prep/check_leakage.py --root DatasetMango_YOLO_clean
```

Debe imprimir `OK: no se encontraron duplicados entre splits` y salir con código 0.

Si todavía encuentra pares, el union-find no los conectó (típicamente cuando la distancia justo supera el umbral). En ese caso:

- Subir `HASH_THRESHOLD` a 6 en ambos scripts y volver a correr, o
- Inspeccionar manualmente los pares reportados y decidir si son realmente duplicados.

---

## 5. Fase 2 — Dataset del detector: descargar y colapsar

### Paso 2.1. Descargar

```bash
python src/data_prep/download_roboflow.py
```

Descarga la versión más reciente del dataset de Luigui Cerna en formato YOLOv8 a `DatasetMango_Detector/`.

**Bug conocido (ya mitigado):** la versión 1.3.3 de `roboflow` salta el download si la carpeta destino existe y tiene contenido. El script actual detecta carpeta vacía y fuerza `overwrite=True`. Si ves que el script dice "OK" pero la carpeta sigue vacía, borra `DatasetMango_Detector/` y vuelve a correr.

### Paso 2.2. Colapsar a una sola clase

```bash
python src/data_prep/collapse_detector_classes.py
```

Reescribe cada label cambiando el primer token (class_id) a `0` y regenera `data.yaml` con `nc: 1`, `names: ['mango']`.

**Por qué colapsar Mango Exportable (1298 imgs) + Mango Industrial (36 imgs) a `mango`:**

- Desbalance 36:1. La clase minoritaria nunca aprendería.
- La arquitectura en cascada delega el diagnóstico al Modelo 2. El detector solo necesita decir "aquí hay un mango".
- Colapsar reduce la complejidad del detector sin perder capacidad útil.

Debe imprimir algo como `X labels remapped, Y files updated`. Idempotente: correrlo dos veces no rompe nada.

### Paso 2.3. Verificar

```bash
cat DatasetMango_Detector/data.yaml
# Esperado: nc: 1   names: ['mango']

# Confirma que todas las labels son clase 0:
awk '{print $1}' DatasetMango_Detector/train/labels/*.txt | sort -u
# Esperado: solo "0"
```

---

## 6. Fase 3 — Test externo propio

El usuario anota 50-80 imágenes de las ~180 propias en Roboflow como dataset de test externo. Una sola clase `mango`. Detalles en el README. Cuando el zip exportado se descomprime en `DatasetMango_Propias/`:

```bash
python src/data_prep/integrate_external_test.py
```

Valida estructura (`train/images`, `valid/images`, `test/images` según el export) y genera `DatasetMango_Propias/external_test.yaml` apuntando al split que funcione como test. Este yaml se usa en la fase de evaluación.

**Por qué separar este dataset:** las 180 imágenes propias fueron tomadas con celular en contexto real del semillero (Universidad Libre Barranquilla, mangos de Sabanalarga). Son la mejor aproximación a cómo el modelo se comportará en producción. No se usan en entrenamiento para no contaminar la evaluación.

---

## 7. Fase 4 — Entrenamiento del clasificador (Modelo 2)

### Paso 4.1. Smoke test (CPU, 2 epochs)

Antes de entrenar real, confirma que el pipeline no esté roto:

```bash
python src/training/train_cls.py --epochs 2 --batch 4 --device cpu
```

Qué pasa por dentro:

1. Si no existe `DatasetMango_YOLO_cls/`, el script llama a `make_cls_layout.py`, que espeja `DatasetMango_YOLO_clean/{split}/images/<nombre>.jpg` a `DatasetMango_YOLO_cls/<split_dst>/<class_name>/<nombre>.jpg` (split `valid` se renombra a `val` porque Ultralytics así lo espera).
2. Carga `yolo11n-cls.pt` y entrena 2 epochs.
3. Copia `runs/classify/train_cls/weights/best.pt` a `models/clasificador_enfermedad.pt`.

**Resultado aceptable:** termina sin error, métricas bajas pero finitas, archivo copiado. Solo valida pipeline end-to-end, no calidad del modelo.

### Paso 4.2. Entrenamiento real

CPU es inviable para la configuración por defecto (50 epochs × 1064 imgs). Opciones:

**Local con GPU (NVIDIA + CUDA):**

```bash
python src/training/train_cls.py
# Equivalente a:
python src/training/train_cls.py --epochs 50 --batch 32 --imgsz 224 --device 0
```

**Kaggle (gratuito, T4 × 2):**

Abrir `notebooks/02_entrenamiento_cls.ipynb`, conectar GPU, correr todas las celdas. El notebook clona el repo, instala deps, corre `train_cls.py`.

**Parámetros y por qué:**

| Parámetro | Default | Razón |
|---|---|---|
| `--epochs 50` | Suficiente para converger con dataset de ~1k imgs. `patience=15` detiene antes si val no mejora |
| `--batch 32` | Cabe en 6 GB VRAM con imgsz 224. Subir a 64 si hay 12 GB+ |
| `--imgsz 224` | Estándar para clasificación; red ligera, entrenamiento rápido |
| `--device auto` | Detecta GPU si hay, fallback CPU |
| `seed=42` | Reproducibilidad |
| `patience=15` | Early stopping generoso para un dataset pequeño |
| `optimizer="auto"` | Ultralytics elige AdamW o SGD según dataset |

### Paso 4.3. Entrenamiento más fuerte (si la val top1 estanca)

Subir capacidad del modelo y resolución:

```bash
python src/training/train_cls.py \
    --model yolo11s-cls.pt \
    --epochs 100 \
    --batch 16 \
    --imgsz 320 \
    --patience 25
```

**Por qué cada cambio:**

- `yolo11s-cls.pt`: modelo "small", más parámetros, mejor capacidad de representación. El `n` (nano) es para edge; `s` es el sweet spot si hay GPU.
- `--imgsz 320`: más detalle de la lesión. MangoDHDS tiene lesiones pequeñas (puntos de antracnosis) que se pierden a 224×224.
- `--batch 16`: se reduce porque 320 consume más VRAM.
- `--epochs 100 --patience 25`: dataset más chico → converge lento → más paciencia.

Si sigue sin mejorar, el cuello de botella no es capacidad sino datos. Agregar más imágenes propias o augmentation más agresiva en el futuro.

### Paso 4.4. Validar artefactos

```bash
ls models/clasificador_enfermedad.pt
ls runs/classify/train_cls/
```

`runs/classify/train_cls/` contiene `results.csv`, `confusion_matrix.png`, `train_batch0.jpg`. Todos útiles para el informe.

---

## 8. Fase 5 — Entrenamiento del detector (Modelo 1)

### Paso 5.1. Comando

```bash
python src/training/train_det_generic.py
# Equivalente a:
python src/training/train_det_generic.py --epochs 100 --batch 16 --imgsz 640 --device auto
```

Qué hace:

1. Regenera `DatasetMango_Detector/data.yaml` con paths absolutos en runtime (evita que el yaml versionado rompa al cambiar de máquina).
2. Carga `yolo11n.pt` y entrena.
3. Copia `best.pt` a `models/detector_mango.pt`.

### Paso 5.2. Parámetros y razones

| Parámetro | Default | Razón |
|---|---|---|
| `--epochs 100` | Detección necesita más epochs que clasificación; convergencia de bbox loss es lenta |
| `--batch 16` | Cabe en 8 GB VRAM a 640×640 |
| `--imgsz 640` | Estándar YOLO para detección. Resolución suficiente para mangos medianos en escena |
| `cos_lr=True` | Learning rate con coseno; mejora convergencia final |
| `patience=20` | Early stopping si val mAP no mejora en 20 epochs |
| `seed=42` | Reproducibilidad |

### Paso 5.3. Entrenamiento más robusto

```bash
python src/training/train_det_generic.py \
    --model yolo11s.pt \
    --epochs 150 \
    --batch 8 \
    --imgsz 832
```

**Por qué:**

- `yolo11s.pt`: +parámetros, mejor mAP en objetos pequeños o escenas con clutter.
- `--imgsz 832`: mangos lejanos o en árboles grandes se detectan mejor con más resolución.
- `--batch 8`: la GPU típica no aguanta 16 × 832. Baja batch antes de subir imgsz.
- `--epochs 150`: dataset de 1334 imgs → red más grande necesita más pasadas.

**Criterio para considerar el detector listo:** mAP50 sobre el val de Luigui Cerna arriba de 0.70. Abajo de eso, revisar augmentation o más data.

### Paso 5.4. GPU requerida

No hay manera realista de entrenar esto en CPU. Opciones:

- **Local CUDA:** si tienes GPU NVIDIA con ≥ 6 GB, corre `--batch 8 --imgsz 640` como punto de partida.
- **Kaggle:** `notebooks/03_entrenamiento_det.ipynb`. Sessions de 9 horas por semana gratis con T4. Suficiente para 100 epochs.
- **Colab Pro:** alternativa.

### Paso 5.5. Validar

```bash
ls models/detector_mango.pt
```

---

## 9. Fase 6 — Evaluación

### Clasificador (MangoDHDS test set limpio)

```bash
python src/evaluation/evaluate_cls.py
```

Genera:
- `reports/figures/confusion_matrix_cls.png`
- `reports/tables/metrics_cls.csv` (precision, recall, F1 por clase)

**Métrica clave:** macro-F1. Accuracy simple no sirve porque las clases pueden quedar sutilmente desbalanceadas.

### Detector (test de Luigui Cerna)

```bash
python src/evaluation/evaluate_det.py
```

Genera `reports/tables/metrics_det.csv` con mAP50, mAP50-95, precision, recall.

### Detector sobre test externo propio (**el importante**)

```bash
python src/evaluation/evaluate_det.py --data DatasetMango_Propias/external_test.yaml
```

Estas métricas son las que importan para la defensa. Demuestran generalización a condiciones reales del semillero. Esperable que sean más bajas que las de Luigui Cerna — eso es honesto y se reporta sin maquillar.

### Explicabilidad

```bash
python src/evaluation/explain_cam.py --n 12
# Solo una clase:
python src/evaluation/explain_cam.py --n 12 --class Antracnosis
```

EigenCAM sobre N muestras del clasificador. Salida en `reports/figures/cam/<clase>/`. Útil para justificar que el modelo mira la lesión, no el fondo.

---

## 10. Fase 7 — Inferencia en cascada y demo

### Demo con webcam

```bash
python src/inference/demo_webcam.py
```

Wrapper con defaults para presentación. Abre la webcam, detecta mangos, los clasifica, dibuja bbox coloreado según la enfermedad, muestra FPS. Salir con `q`.

### Inferencia manual con más control

```bash
python src/inference/infer_cascade.py \
    --det-weights models/detector_mango.pt \
    --cls-weights models/clasificador_enfermedad.pt \
    --source 0 \
    --save-video runs/cascade/demo.mp4
```

`--source` acepta:
- Índice de cámara (`0`, `1`).
- Ruta a video (`.mp4`, `.avi`).
- Ruta a imagen (`.jpg`).
- Ruta a carpeta con imágenes.

### Umbrales

Configurados en `src/utils/config.py`:

```python
CONF_DETECTION = 0.5       # detector: mínima confianza para considerar bbox
CONF_CLASSIFICATION = 0.4  # clasificador: mínima confianza para mostrar clase
```

Si en la demo salen bboxes falsos, sube `CONF_DETECTION` a 0.6. Si el clasificador se ve dubitativo, baja `CONF_CLASSIFICATION` a 0.3 para ver qué está prediciendo.

---

## 11. Parámetros: por qué son esos y cómo escalar

### Matriz de decisión

| Situación | Cambio recomendado |
|---|---|
| Smoke test en CPU | `--epochs 2 --batch 4 --device cpu` |
| GPU local 6 GB | defaults |
| GPU local 12 GB+ | `--batch 64` cls, `--batch 32` det |
| Kaggle T4 | defaults funcionan; `--batch 64` si no OOM |
| Val top1 / mAP estanca | `yolo11s` en vez de `yolo11n`, `--imgsz` mayor |
| Dataset crece al doble | `--epochs` +50, subir `--patience` |
| Ruido en inferencia | `CONF_DETECTION` 0.6; `CONF_CLASSIFICATION` 0.5 |
| Muchos falsos negativos | `CONF_DETECTION` 0.35; revisar augmentation del detector |

### Qué NO escalar sin motivo

- `seed=42`. Cambiarlo sin documentar rompe reproducibilidad.
- `imgsz` para clasificación más allá de 320 rara vez ayuda y multiplica tiempo de entrenamiento.
- `epochs` sin subir `patience`. Si `patience` no se ajusta, corta antes y el aumento no tiene efecto.

### Modelos `yolo11n-cls` vs `yolo11s-cls` vs más grandes

| Modelo | Parámetros | Cuándo usar |
|---|---|---|
| `yolo11n-cls` | ~2.5M | Edge, Raspberry, baseline rápido |
| `yolo11s-cls` | ~9M | Default razonable con GPU |
| `yolo11m-cls` | ~23M | Dataset grande, GPU sobrada |
| `yolo11l-cls` / `yolo11x-cls` | +50M | Investigación, overkill para 1k imgs |

Mismo criterio para detección (`yolo11n.pt`, `yolo11s.pt`, etc.).

---

## 12. Errores comunes y diagnóstico

### "loading Roboflow workspace..." OK pero `DatasetMango_Detector/` vacío

**Causa:** SDK 1.3.3 salta download si la carpeta destino existe. El script actual ya fuerza `overwrite=True` cuando la carpeta está vacía.

**Fix:** borrar la carpeta y re-ejecutar.

```bash
rm -rf DatasetMango_Detector
python src/data_prep/download_roboflow.py
```

### `ERROR: la variable de entorno ROBOFLOW_API_KEY no esta definida`

Exportar la variable (sección 2) **en la misma terminal** donde correrás el script. En PowerShell cerrar y reabrir invalida la variable si no está en `$PROFILE`.

### `No se encontraron duplicados entre splits` pero aún ves números raros

Leakage también puede venir de labels mal asignadas (no de imágenes duplicadas). Revisar `notebooks/01_exploracion_dataset.ipynb`, sección de muestreo visual por clase.

### Smoke test del clasificador falla con `FileNotFoundError` en `DatasetMango_YOLO_clean`

Te saltaste la Fase 1. Correr primero:

```bash
python src/data_prep/dedup_and_resplit.py
```

### `FileNotFoundError: DatasetMango_Detector/` al entrenar el detector

Fases 2.1 y 2.2 no ejecutadas. Ver sección 5.

### OOM (out of memory) en GPU durante entrenamiento

Bajar `--batch` a la mitad. Si persiste, bajar también `--imgsz`.

### mAP50 del detector < 0.5

Datos insuficientes o dataset muy desbalanceado para la tarea. Inspeccionar `runs/detect/train_det/train_batch*.jpg` y `val_batch*.jpg` para ver si hay bboxes mal dibujados o imágenes ruidosas. Considerar ampliar con más imágenes propias anotadas.

### Demo en webcam abre cámara equivocada

Cambiar `--source 0` por `--source 1` (o 2). En Windows a veces la webcam integrada es índice 1 si hay drivers de cámara virtual.

### `cv2.error: OpenCV(...) !_src.empty()` en inferencia

Ruta de imagen/video incorrecta o archivo corrupto. Verificar con:

```bash
python -c "import cv2; print(cv2.imread('ruta/al/archivo.jpg') is None)"
```

### Near-duplicates aún aparecen tras `dedup_and_resplit.py`

Umbral 5 no capturó pares borderline. Subir a 6 en ambos scripts (`check_leakage.py` y `dedup_and_resplit.py`) y re-correr. Documentar el cambio.

---

## 13. Checklist final antes de defensa

- [ ] `check_leakage.py --root DatasetMango_YOLO_clean` → OK, 0 pares.
- [ ] `models/clasificador_enfermedad.pt` existe y proviene de entrenamiento completo (no smoke test).
- [ ] `models/detector_mango.pt` existe y `evaluate_det.py` reporta mAP50 ≥ 0.70 en val de Luigui Cerna.
- [ ] `evaluate_det.py --data DatasetMango_Propias/external_test.yaml` ejecutado y métricas registradas.
- [ ] `reports/figures/confusion_matrix_cls.png` generado.
- [ ] `reports/tables/metrics_cls.csv` y `metrics_det.csv` generados.
- [ ] `reports/figures/cam/` poblado con EigenCAM de al menos 12 muestras.
- [ ] Demo probada en la máquina de presentación con la cámara que se va a usar.
- [ ] Lista de muestras "buenas" para la demo preparada (pool de mangos de la finca que el modelo clasifique correctamente).
- [ ] README actualizado con resultados finales (no solo placeholders).
- [ ] Sección "Limitaciones conocidas" leída y asumida (bboxes sintéticos, sesgo OOD, detector single-class).

---

## Trabajo futuro (fuera del alcance actual)

- Integración RGB-D con Intel RealSense D435i para estimar tamaño real del fruto.
- Captura y anotación propia continua en Sabanalarga.
- Dataset de podredumbre severa postcosecha.
- Despliegue edge en Raspberry Pi 4 con modelo cuantizado a INT8.
- Curación activa con `video_to_frames.py` sobre videos de supervisión de la finca.

Estas no se implementan ahora; el guion del jurado las menciona como continuación natural.
