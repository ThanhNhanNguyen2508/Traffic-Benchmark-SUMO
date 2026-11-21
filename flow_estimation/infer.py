#!/usr/bin/env python3
"""INFERENCE: vehicles / hour per class for real city cameras, using a fitted calibration (calibrate.py).

Input: one or more session folders as written by the snapshot collector:
    <session>/<Cam_name>/metadata.csv   (columns: filename, received_at_vn, ...)
    <session>/<Cam_name>/frame_*.jpg     (512x288 JPEG, one every ~10-15 s)

Per camera, images are sorted by receive time and for every image k:
  1. YOLO boxes (same model, same thresholds as calibration);
  2. new vehicles vs image k-1 (Hungarian matching; parked-cell mask recomputed per 10-min block);
  3. dt = real time since image k-1, rounded to the nearest calibrated gap (5/10/15/20/30 s);
     flow_k = new_k * beta(class, dt); pairs with dt > 35 s are missing data (no flow, no time);
  4. per bin: vehicles/hour = sum(flow) / sum(covered seconds) * 3600.
Nothing is re-fitted on the real cameras (they have no ground truth).

    python -m flow_estimation.infer --root data/city_cam --sessions 2026-10-29_0850-1250 2026-11-07_1306-1706 \
        --exclude Cam_To_Hien_Thanh_Su_Van_Hanh --dets data/detections/city_cam_yolo11x_1280.jsonl.gz \
        --calibration flow_estimation/calibration/yolo11x_1280.json --out results/flow
If --dets does not exist, YOLO is run first (needs the images and a GPU).
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt_
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from .common import BLOCK_S, CLASSES, MAX_GAP_S, STATIC_FRAC, count_new, keep, load_detections, nearest_gap

TZ = dt_.timezone(dt_.timedelta(hours=7))


def list_frames(root: Path, sessions, exclude):
    rows = []
    for s in sessions:
        for cam_dir in sorted((root / s).glob("Cam_*")):
            if cam_dir.name in exclude or not (cam_dir / "metadata.csv").is_file():
                continue
            for r in csv.DictReader(open(cam_dir / "metadata.csv", encoding="utf-8-sig")):
                p = cam_dir / r["filename"]
                if p.is_file():
                    rows.append({"session": s, "camera": cam_dir.name, "t": r["received_at_vn"],
                                 "image": str(p.relative_to(root))})
    return rows


def per_frame_flow(frames, D, cal):
    """frames of one camera in time order -> ts, dt, valid, new[K,4], flow[K,4]"""
    ts = np.array([dt_.datetime.fromisoformat(r["t"]).timestamp() for r in frames])
    beta = {int(g): np.array(b) for g, b in cal["beta"].items()}
    cm, co = cal["conf"]["motorcycle"], cal["conf"]["other"]
    block = ((ts - ts[0]) // BLOCK_S).astype(int)
    new = np.zeros((len(frames), 4))
    for conf, cols in ((cm, slice(0, 1)), (co, slice(1, 4))):      # same per-threshold runs as calibration
        dets = [[d for d in D[r["image"]] if d[5] >= conf and d[0] in ("motorcycle", "bicycle", "car", "truck", "bus")]
                for r in frames]
        n_all = np.zeros_like(new)
        for b in np.unique(block):
            ii = np.where(block == b)[0]; lo = max(ii[0] - 1, 0)    # keep the previous image for matching
            n = count_new(dets[lo:ii[-1] + 1], STATIC_FRAC)
            n_all[ii] = n[ii[0] - lo:]
        new[:, cols] = n_all[:, cols]
    dt = np.diff(ts, prepend=np.nan)
    valid = ~np.isnan(dt) & (dt <= MAX_GAP_S)
    g = nearest_gap(np.nan_to_num(dt), beta.keys())
    flow = np.where(valid[:, None], new * np.array([beta[x] for x in g]), 0.0)
    return ts, np.nan_to_num(dt), valid, new, flow


def bins(ts, dt, valid, flow, size_s):
    out, t0 = [], ts[0]
    idx = ((ts - t0) // size_s).astype(int)
    for h in range(int(idx.max()) + 1):
        m = idx == h
        cov = float(dt[valid & m].sum())
        if m.sum() < 10 or cov < 60:
            continue
        rate = flow[m].sum(0) / cov * 3600
        out.append({"from": dt_.datetime.fromtimestamp(t0 + size_s * h, TZ).strftime("%H:%M"),
                    "to": dt_.datetime.fromtimestamp(min(t0 + size_s * (h + 1), ts[-1]), TZ).strftime("%H:%M"),
                    "covered_min": round(cov / 60, 1), "images": int(m.sum()),
                    "veh_per_h": {c: int(round(x)) for c, x in zip(CLASSES, rate)}, "total": int(round(rate.sum()))})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("data/city_cam"), help="folder holding the session folders")
    ap.add_argument("--sessions", nargs="+", required=True)
    ap.add_argument("--index", type=Path, default=None,
                    help="frame index jsonl (session, camera, t, image) to use instead of reading the session folders; "
                         "lets inference run from the detection cache without the images "
                         "(data/detections/city_cam_frames.jsonl)")
    ap.add_argument("--write-index", type=Path, default=None, help="save the frame index read from the folders")
    ap.add_argument("--exclude", nargs="*", default=[], help="camera folder names to skip")
    ap.add_argument("--dets", type=Path, required=True, help="detection cache; created with YOLO if missing")
    ap.add_argument("--calibration", type=Path, required=True)
    ap.add_argument("--bin-min", type=int, default=60, help="aggregation bin (minutes) for the hourly table")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    if a.index:
        frames = [r for r in map(json.loads, open(a.index, encoding="utf-8"))
                  if r["session"] in a.sessions and r["camera"] not in set(a.exclude)]
    else:
        frames = list_frames(a.root, a.sessions, set(a.exclude))
    if a.write_index:
        with open(a.write_index, "w", encoding="utf-8") as fo:
            for r in frames:
                fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    if not a.dets.is_file():
        lst = a.out / "frames.jsonl"; a.out.mkdir(parents=True, exist_ok=True)
        with open(lst, "w", encoding="utf-8") as fo:
            for r in frames:
                fo.write(json.dumps(r, ensure_ascii=False) + "\n")
        subprocess.run([sys.executable, "-m", "flow_estimation.detect", "--list", str(lst), "--root", str(a.root),
                        "--out", str(a.dets)], check=True)
    D = load_detections(a.dets); cal = json.load(open(a.calibration))
    groups = {}
    for r in frames:
        groups.setdefault((r["session"], r["camera"]), []).append(r)

    a.out.mkdir(parents=True, exist_ok=True)
    flow_out, perframe, table = {}, {}, []
    for (s, c), rows in sorted(groups.items()):
        rows.sort(key=lambda r: r["t"])
        ts, dt, valid, new, flow = per_frame_flow(rows, D, cal)
        key = f"{s}|{c}"
        hours = bins(ts, dt, valid, flow, a.bin_min * 60)
        ten = bins(ts, dt, valid, flow, 600)
        flow_out[key] = {"start": dt_.datetime.fromtimestamp(ts[0], TZ).isoformat(), "bins": hours, "series_10min": ten}
        perframe[key] = {"t": [r["t"] for r in rows], "dt": dt.round(3).tolist(), "valid": valid.tolist(),
                         "new": new.astype(int).tolist(), "flow": flow.round(4).tolist()}
        for h in hours:
            table.append([s, c, h["from"], h["to"], h["covered_min"], *h["veh_per_h"].values(), h["total"]])
        print(key)
        for h in hours:
            print(f"  {h['from']}-{h['to']} ({h['covered_min']:.0f} min) " + "  ".join(f"{k} {v:6d}" for k, v in h["veh_per_h"].items()) + f" | total {h['total']}")
    json.dump(flow_out, open(a.out / "flow.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(perframe, open(a.out / "perframe.json", "w", encoding="utf-8"))
    with open(a.out / "flow_bins.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["session", "camera", "from", "to", "covered_min", *CLASSES, "total"]); w.writerows(table)
    print("->", a.out)


if __name__ == "__main__":
    main()
