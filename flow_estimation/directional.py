#!/usr/bin/env python3
"""DIRECTIONAL flow (FORWARD / REVERSE) per camera: infer.py's per-frame estimator run independently per ROI.

Every detection footpoint (x_center, y_bottom) is assigned to a label by roi.py (polygons in directional_roi.json);
EXCLUDE detections are never counted. Matching (Hungarian) and new-vehicle counting run separately per label; the
parked-cell mask is still computed on the FULL frame of each 600 s block (baseline semantics) and passed to every ROI.
Confidence filters, dt / valid / nearest-gap / beta are exactly those of infer.per_frame_flow.
Detections must already be cached (--dets); YOLO is never run here.

    python -m flow_estimation.directional infer --index data/detections/city_cam_frames.jsonl \
        --sessions 2026-10-29_0850-1250 2026-11-07_1306-1706 2026-11-12_0530-0700 \
        --exclude Cam_To_Hien_Thanh_Su_Van_Hanh --dets data/detections/city_cam_yolo11x_1280.jsonl.gz \
        --calibration flow_estimation/calibration/yolo11x_1280.json \
        --roi flow_estimation/directional_roi.json --out <dir>            # -> <dir>/perframe_directional.json
    python -m flow_estimation.directional window --perframe <dir_baseline>/perframe.json \
        --perframe-directional <dir>/perframe_directional.json \
        --setting light 2026-11-12_0530-0700 05:30 06:30 --setting heavy 2026-10-29_0850-1250 10:50 11:50 \
        --out <dir>/demand_settings_directional.json
`fullframe` in the window output is window.window() of the baseline perframe (identical to demand_settings.json).
"""
from __future__ import annotations

import argparse
import datetime as dt_
import hashlib
import json
from pathlib import Path

import numpy as np

from .common import BLOCK_S, CLASSES, MAX_GAP_S, STATIC_FRAC, count_new, load_detections, nearest_gap, static_cells
from .infer import bins, list_frames
from .window import window

VEHICLES = ("motorcycle", "bicycle", "car", "truck", "bus")


def per_frame_flow_directional(frames, D, cal, roi_cam):
    """frames of one camera in time order -> ts, dt, valid, new{label: [K,4]}, flow{label: [K,4]} (label in roi_cam.directions)"""
    ts = np.array([dt_.datetime.fromisoformat(r["t"]).timestamp() for r in frames])
    beta = {int(g): np.array(b) for g, b in cal["beta"].items()}
    cm, co = cal["conf"]["motorcycle"], cal["conf"]["other"]
    block = ((ts - ts[0]) // BLOCK_S).astype(int)
    labels = list(roi_cam.directions)
    new = {lb: np.zeros((len(frames), 4)) for lb in labels}
    for conf, cols in ((cm, slice(0, 1)), (co, slice(1, 4))):      # same per-threshold runs as calibration
        dets = [[d for d in D[r["image"]] if d[5] >= conf and d[0] in VEHICLES] for r in frames]
        parts = [roi_cam.split(fr) for fr in dets]                   # label -> dets of that frame
        n_all = {lb: np.zeros((len(frames), 4)) for lb in labels}
        for b in np.unique(block):
            ii = np.where(block == b)[0]; lo = max(ii[0] - 1, 0)    # keep the previous image for matching
            mask = static_cells(dets[lo:ii[-1] + 1], STATIC_FRAC)    # static mask on the FULL frame
            for lb in labels:
                n = count_new([p[lb] for p in parts[lo:ii[-1] + 1]], mask=mask)
                n_all[lb][ii] = n[ii[0] - lo:]
        for lb in labels:
            new[lb][:, cols] = n_all[lb][:, cols]
    dt = np.diff(ts, prepend=np.nan)
    valid = ~np.isnan(dt) & (dt <= MAX_GAP_S)
    g = nearest_gap(np.nan_to_num(dt), beta.keys())
    bg = np.array([beta[x] for x in g])
    flow = {lb: np.where(valid[:, None], new[lb] * bg, 0.0) for lb in labels}
    return ts, np.nan_to_num(dt), valid, new, flow


def dets_per_label(frames, D, cal, roi_cam):
    """number of detections (counted classes, after the confidence filters) falling in each label, all frames"""
    cm, co = cal["conf"]["motorcycle"], cal["conf"]["other"]
    out = {lb: 0 for lb in roi_cam.labels}
    for r in frames:
        dets = [d for d in D[r["image"]] if d[0] in VEHICLES and d[5] >= (cm if d[0] in ("motorcycle", "bicycle") else co)]
        for lb, ds in roi_cam.split(dets).items():
            out[lb] += len(ds)
    return out


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cmd_infer(a):
    from .roi import load_roi
    roi = load_roi(a.roi)
    if not a.dets.is_file():
        raise SystemExit(f"--dets {a.dets} not found: directional inference needs the detection cache "
                         f"(run flow_estimation.infer / detect first; YOLO is not run here)")
    if a.index:
        frames = [r for r in map(json.loads, open(a.index, encoding="utf-8"))
                  if r["session"] in a.sessions and r["camera"] not in set(a.exclude)]
    else:
        frames = list_frames(a.root, a.sessions, set(a.exclude))
    D = load_detections(a.dets)
    with open(a.calibration, encoding="utf-8") as f:
        cal = json.load(f)
    groups = {}
    for r in frames:
        groups.setdefault((r["session"], r["camera"]), []).append(r)

    a.out.mkdir(parents=True, exist_ok=True)
    series = {}
    for (s, c), rows in sorted(groups.items()):
        if c not in roi:
            print(f"skip {s}|{c}: camera not in ROI file")
            continue
        rows.sort(key=lambda r: r["t"])
        ts, dt, valid, new, flow = per_frame_flow_directional(rows, D, cal, roi[c])
        key = f"{s}|{c}"
        series[key] = {"t": [r["t"] for r in rows], "dt": dt.round(3).tolist(), "valid": valid.tolist(),
                       "directions": {lb: {"new": new[lb].astype(int).tolist(), "flow": flow[lb].round(4).tolist()}
                                      for lb in roi[c].directions},
                       "dets_per_label": dets_per_label(rows, D, cal, roi[c])}
        print(key, "dets_per_label", series[key]["dets_per_label"])
        for lb in roi[c].directions:
            print(f"  {lb}")
            for h in bins(ts, dt, valid, flow[lb], a.bin_min * 60):
                print(f"    {h['from']}-{h['to']} ({h['covered_min']:.0f} min) " +
                      "  ".join(f"{k} {v:6d}" for k, v in h["veh_per_h"].items()) + f" | total {h['total']}")
    out = {"schema_version": "j02-perframe-directional-1",
           "provenance": {"dets": str(a.dets), "dets_sha256": sha256(a.dets),
                          "calibration": str(a.calibration), "calibration_sha256": sha256(a.calibration),
                          "roi": str(a.roi), "roi_sha256": sha256(a.roi), "classes": CLASSES,
                          "anchor": "footpoint (x_center, y_bottom)",
                          "static_mask": "full-frame per 600 s block (baseline semantics)",
                          "matching": "independent per ROI"},
           "series": series}
    with open(a.out / "perframe_directional.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print("->", a.out / "perframe_directional.json")


def cmd_window(a):
    with open(a.perframe, encoding="utf-8") as f:
        P = json.load(f)
    with open(a.perframe_directional, encoding="utf-8") as f:
        PD = json.load(f)
    labels = sorted({lb for d in PD["series"].values() for lb in d["directions"]})
    res = {"schema_version": "j02-demand-settings-1",
           "provenance": {"perframe": str(a.perframe), "perframe_sha256": sha256(a.perframe),
                          "perframe_directional": str(a.perframe_directional),
                          "perframe_directional_sha256": sha256(a.perframe_directional),
                          "directional_provenance": PD["provenance"]}}
    for name, s, t0, t1 in a.setting:
        cams = {c: {"fullframe": v} for c, v in window(P, s, t0, t1).items()}
        for lb in labels:                                            # per-direction views for window.window()
            view = {k: {"t": d["t"], "dt": d["dt"], "valid": d["valid"], "flow": d["directions"][lb]["flow"]}
                    for k, d in PD["series"].items() if lb in d["directions"]}
            for c, v in window(view, s, t0, t1).items():
                cams.setdefault(c, {})[lb] = v
        res[name] = {"session": s, "from": t0, "to": t1, "cameras": cams}
        print(f"{name:7s} {s} {t0}-{t1}")
        for c, v in cams.items():
            print(f"  {c}: " + "  ".join(f"{k} {x['total']:,}" for k, x in v.items()))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("infer", help="per-frame directional flow -> perframe_directional.json")
    p.add_argument("--root", type=Path, default=Path("data/city_cam"), help="folder holding the session folders")
    p.add_argument("--sessions", nargs="+", required=True)
    p.add_argument("--index", type=Path, default=None, help="frame index jsonl (session, camera, t, image)")
    p.add_argument("--exclude", nargs="*", default=[], help="camera folder names to skip")
    p.add_argument("--dets", type=Path, required=True, help="existing detection cache (never created here)")
    p.add_argument("--calibration", type=Path, required=True)
    p.add_argument("--roi", type=Path, required=True, help="directional_roi.json")
    p.add_argument("--bin-min", type=int, default=60, help="aggregation bin (minutes) for the hourly summary")
    p.add_argument("--out", type=Path, required=True, help="output folder")
    p.set_defaults(fn=cmd_infer)
    p = sub.add_parser("window", help="FORWARD/REVERSE/fullframe rates over clock windows")
    p.add_argument("--perframe", type=Path, required=True, help="baseline perframe.json from infer.py")
    p.add_argument("--perframe-directional", type=Path, required=True)
    p.add_argument("--setting", nargs=4, action="append", metavar=("NAME", "SESSION", "FROM", "TO"), required=True)
    p.add_argument("--out", type=Path, required=True, help="demand_settings_directional.json")
    p.set_defaults(fn=cmd_window)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
