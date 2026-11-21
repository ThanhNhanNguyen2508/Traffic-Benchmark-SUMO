#!/usr/bin/env python3
"""Run YOLO on a list of images and cache every box (step shared by calibration and inference).

The detector is the COCO-pretrained YOLO11x from Ultralytics, NOT fine-tuned. City-camera images are
512x288; they are upscaled to --imgsz 1280 inside the model (small motorcycles are detected far better).
Boxes are kept down to conf 0.08 so that confidence thresholds can be tuned later without re-running YOLO.
Output coordinates are scaled to 1920x1080 space (x3.75 for 512-wide images).

    python -m flow_estimation.detect --list frames.jsonl --root <image root> --out dets.jsonl.gz
    (frames.jsonl: one {"image": "<path relative to --root>", ...} per line)
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

COCO = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
REPO = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = REPO / "models" / "yolo11x.pt"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", required=True, type=Path, help="jsonl with an 'image' key per line")
    ap.add_argument("--root", required=True, type=Path, help="folder the 'image' paths are relative to")
    ap.add_argument("--out", required=True, type=Path, help="output .jsonl or .jsonl.gz")
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS) if DEFAULT_WEIGHTS.is_file() else "yolo11x.pt",
                    help="YOLO weights (default: models/yolo11x.pt, else Ultralytics downloads yolo11x.pt)")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default=None, help="e.g. 0, cuda:1, cpu")
    a = ap.parse_args()

    from ultralytics import YOLO
    keys = [json.loads(l)["image"] for l in open(a.list, encoding="utf-8")]
    keys = list(dict.fromkeys(keys))
    model = YOLO(a.weights)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if a.out.suffix == ".gz" else open
    with opener(a.out, "wt", encoding="utf-8") as fo:
        for i in range(0, len(keys), a.batch):
            chunk = keys[i:i + a.batch]
            res = model.predict([str(a.root / k) for k in chunk], imgsz=a.imgsz, conf=0.08, iou=0.6,
                                classes=list(COCO), verbose=False, half=True, max_det=400, device=a.device)
            for k, r in zip(chunk, res):
                s = 1920 / r.orig_shape[1]                     # back to 1080p pixel space
                b = r.boxes
                xyxy = b.xyxy.cpu().numpy() * s
                dets = [[COCO[int(c)], round(float((x0 + x1) / 2), 1), round(float(y1), 1), round(float(y1 - y0), 1),
                         round(float(x1 - x0), 1), round(float(p), 3)]
                        for (x0, y0, x1, y1), c, p in zip(xyxy, b.cls.cpu().numpy(), b.conf.cpu().numpy())]
                fo.write(json.dumps({"image": k, "dets": dets}) + "\n")
            if i % (a.batch * 50) == 0:
                print(f"{i}/{len(keys)}", flush=True)
    print("done:", a.out)


if __name__ == "__main__":
    main()
