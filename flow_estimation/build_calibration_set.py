#!/usr/bin/env python3
"""Build the calibration set: ground-truth flow per video + snapshot frames degraded like the city feed.

Input (not redistributed, see data/README.md):
  --tracks   one JSON per video with tracked objects: {"video", "camera", "objects": [{"cls", "id", "h_med",
             "path": [[t, x_footpoint, y_footpoint, box_h], ...]}]}  (t = frame index / 25.0014, 1920x1080 px)
  --mapkf    one CSV per video with columns frame_id, timestamp (keyframe -> presentation time, s)
  --keyframes  keyframe images <keyframes>/<lot>/<video>/<frame_id:06d>.webp (1920x1080)

Output (--out):
  videos.json            {video: {camera, duration_s, flow: {motorcycle, car, truck, bus}}}  ground truth
  frames.jsonl           one line per snapshot on a 5 s grid: {video, camera, T, pts, image}
  images/<video>/<id>.jpg  the snapshots resized to 512x288, JPEG quality 88 (= city camera images)

Ground truth: a tracked vehicle is MOVING if its footpoint travels more than max(40 px, 0.8 x median box
height); the true flow of a video is the number of moving vehicles per class. Parked vehicles never count.

    python -m flow_estimation.build_calibration_set --tracks .../acts --mapkf .../map-keyframes \
        --keyframes .../keyframes --out data/calibration
"""
from __future__ import annotations

import argparse
import bisect
import csv
import glob
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .common import CLASSES

GRID_S = 5.0          # one snapshot every 5 s of video; sequences with gaps 5/10/15/20/30 s are subsets


def _resize(job):
    src, dst = job
    if not os.path.isfile(dst):
        from PIL import Image
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        Image.open(src).convert("RGB").resize((512, 288), Image.LANCZOS).save(dst, quality=88)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tracks", required=True, type=Path)
    ap.add_argument("--mapkf", required=True, type=Path)
    ap.add_argument("--keyframes", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    videos, frames, jobs = {}, [], []
    for f in sorted(glob.glob(str(a.tracks / "*.json"))):
        d = json.load(open(f, encoding="utf-8")); v = d["video"]; lot = v.split("-")[0]
        flow = {c: 0 for c in CLASSES}
        for o in d["objects"]:
            if o["cls"] not in flow or not o["path"]:
                continue
            P = np.asarray(o["path"], dtype=np.float32)
            h = float(o.get("h_med") or np.median(P[:, 3]))
            if float(np.max(np.hypot(P[:, 1] - P[0, 1], P[:, 2] - P[0, 2]))) > max(40.0, 0.8 * h):
                flow[o["cls"]] += 1
        rows = [(int(r["frame_id"]), float(r["timestamp"])) for r in csv.DictReader(open(a.mapkf / f"{v}.csv"))]
        pts = [p for _, p in rows]
        videos[v] = {"camera": d["camera"], "duration_s": round(pts[-1], 1), "flow": flow}
        T = 0.0
        while T <= pts[-1] + 1e-6:                       # keyframe nearest to each 5 s mark
            i = bisect.bisect_left(pts, T)
            j = min((x for x in (i - 1, i) if 0 <= x < len(pts)), key=lambda x: abs(pts[x] - T))
            if abs(pts[j] - T) <= 1.5:
                fid = rows[j][0]; src = a.keyframes / lot / v / f"{fid:06d}.webp"
                if src.is_file():
                    key = f"{v}/{fid:06d}.jpg"
                    jobs.append((str(src), str(a.out / "images" / key)))
                    frames.append({"video": v, "camera": d["camera"], "T": T, "pts": pts[j], "image": key})
            T += GRID_S
    with ProcessPoolExecutor(a.workers) as ex:
        list(ex.map(_resize, jobs, chunksize=64))
    json.dump(videos, open(a.out / "videos.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    with open(a.out / "frames.jsonl", "w", encoding="utf-8") as fo:
        for r in frames:
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(videos)} videos, {len(frames)} snapshots -> {a.out}")


if __name__ == "__main__":
    main()
