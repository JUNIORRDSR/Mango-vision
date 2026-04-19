"""Muestrea frames de un video y deduplica por perceptual hash.

Uso (no se ejecuta en este pase de scaffolding):
    python src/data_prep/video_to_frames.py \
        --video path/al/video.mp4 \
        --output data/frames/ \
        --every 15 \
        --hash-threshold 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import imagehash
from PIL import Image


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--every", type=int, default=15, help="Muestrear cada N frames.")
    p.add_argument("--hash-threshold", type=int, default=5)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.video.is_file():
        print(f"ERROR: no existe el video {args.video}")
        return 1
    args.output.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(args.video))
    kept_hashes: list[imagehash.ImageHash] = []
    kept = 0
    seen = 0
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        seen += 1
        if seen % args.every != 0:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        h = imagehash.phash(pil)
        if any(h - prev < args.hash_threshold for prev in kept_hashes):
            continue
        kept_hashes.append(h)
        out = args.output / f"frame_{idx:06d}.jpg"
        cv2.imwrite(str(out), frame)
        kept += 1
        idx += 1
    cap.release()
    print(f"OK: {seen} frames leidos, {kept} unicos guardados en {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
