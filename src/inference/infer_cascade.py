"""Inferencia en cascada: detector generico -> clasificador de enfermedad.

Flujo por frame:
    1. Detector (Modelo 1) encuentra bboxes de mangos.
    2. Para cada bbox con conf > CONF_DETECTION, se recorta y redimensiona.
    3. El crop pasa al clasificador (Modelo 2) que devuelve clase y confianza.
    4. Se dibuja bbox con color segun clase diagnosticada y texto:
       '<clase_enfermedad>: <confianza>%'.
    5. FPS en esquina superior izquierda.

Uso:
    # webcam
    python src/inference/infer_cascade.py --det-weights models/detector_mango.pt --cls-weights models/clasificador_enfermedad.pt --source 0

    # video
    python src/inference/infer_cascade.py --det-weights models/detector_mango.pt --cls-weights models/clasificador_enfermedad.pt --source path/al/video.mp4

    # imagen
    python src/inference/infer_cascade.py --det-weights ... --cls-weights ... --source path/a/imagen.jpg

    # grabar
    python src/inference/infer_cascade.py ... --save-video runs/cascade/demo.mp4
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import (
    COLORS_BY_CLASS,
    CONF_CLASSIFICATION,
    CONF_DETECTION,
    IMG_SIZE_CLS,
    WEIGHTS_CLS,
    WEIGHTS_DET,
)

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--det-weights", type=Path, default=WEIGHTS_DET)
    p.add_argument("--cls-weights", type=Path, default=WEIGHTS_CLS)
    p.add_argument("--source", default="0", help="Webcam index, ruta a video, o ruta a imagen.")
    p.add_argument("--conf-det", type=float, default=CONF_DETECTION)
    p.add_argument("--conf-cls", type=float, default=CONF_CLASSIFICATION)
    p.add_argument("--imgsz-cls", type=int, default=IMG_SIZE_CLS)
    p.add_argument("--save-video", type=Path, default=None)
    p.add_argument("--no-show", action="store_true", help="No abrir ventana cv2.")
    return p.parse_args()


def draw_box(frame, xyxy, label: str, color: tuple[int, int, int]) -> None:
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
    cv2.putText(frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)


def process_frame(frame, detector, classifier, args) -> any:
    det_results = detector.predict(frame, conf=args.conf_det, verbose=False)[0]
    boxes = det_results.boxes
    if boxes is None or len(boxes) == 0:
        return frame

    for i in range(len(boxes)):
        xyxy = boxes.xyxy[i].cpu().numpy()
        x1, y1, x2, y2 = [max(0, int(v)) for v in xyxy]
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        cls_result = classifier.predict(crop, imgsz=args.imgsz_cls, verbose=False)[0]
        probs = cls_result.probs
        top1 = int(probs.top1)
        conf = float(probs.top1conf)
        cls_name = cls_result.names[top1]
        if conf < args.conf_cls:
            label_cls = f"Desconocido: {conf * 100:.0f}%"
            color = (128, 128, 128)
        else:
            label_cls = f"{cls_name}: {conf * 100:.0f}%"
            color = COLORS_BY_CLASS.get(cls_name, (255, 255, 255))
        draw_box(frame, (x1, y1, x2, y2), label_cls, color)
    return frame


def draw_fps(frame, fps: float) -> None:
    text = f"FPS: {fps:.1f}"
    cv2.putText(frame, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
    cv2.putText(frame, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


def open_source(source: str):
    """Devuelve (cap, is_image, single_image_or_None)."""
    p = Path(source)
    if p.is_file() and p.suffix.lower() in IMG_EXT:
        img = cv2.imread(str(p))
        return None, True, img
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
    else:
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la fuente {source}")
    return cap, False, None


def main() -> int:
    args = parse_args()
    if not args.det_weights.is_file():
        print(f"ERROR: no existen los pesos del detector {args.det_weights}")
        return 1
    if not args.cls_weights.is_file():
        print(f"ERROR: no existen los pesos del clasificador {args.cls_weights}")
        return 1

    from ultralytics import YOLO

    detector = YOLO(str(args.det_weights))
    classifier = YOLO(str(args.cls_weights))

    cap, is_image, single_img = open_source(args.source)

    if is_image:
        out = process_frame(single_img, detector, classifier, args)
        if args.save_video:
            args.save_video.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(args.save_video.with_suffix(".jpg")), out)
        if not args.no_show:
            cv2.imshow("Mango Cascade", out)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return 0

    writer = None
    if args.save_video:
        args.save_video.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps_src = cap.get(cv2.CAP_PROP_FPS) or 20.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(str(args.save_video), fourcc, fps_src, (w, h))

    t_prev = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            out = process_frame(frame, detector, classifier, args)
            now = time.time()
            fps = 1.0 / max(1e-6, (now - t_prev))
            t_prev = now
            draw_fps(out, fps)
            if writer is not None:
                writer.write(out)
            if not args.no_show:
                cv2.imshow("Mango Cascade", out)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if not args.no_show:
            cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
