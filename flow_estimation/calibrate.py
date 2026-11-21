#!/usr/bin/env python3
"""TRAINING: fit the conversion factors beta(class, gap) and the confidence thresholds on the calibration set,
and measure the error by 3-fold cross-validation over cameras.

For every video and every snapshot gap dt in {5, 10, 15, 20, 30} s (plus a random 10/15 s mix like the real
morning feed), the 5 s snapshot grid is subsampled, YOLO boxes are filtered at a confidence threshold, and the
NEW vehicles of each class are counted (Hungarian matching to the previous snapshot + parked-cell mask, see
common.py). Then

    beta(class, dt) = median over videos of  true_flow(class) / counted_new(class)

Confidence thresholds are chosen on a grid {0.10 ... 0.35}: first the motorcycle threshold (others fixed at
0.25), then one shared threshold for car/truck/bus, each minimising WAPE on the training videos.

Cross-validation: cameras -> 3 fixed folds (data/calibration/camera_folds.json). For each fold: thresholds and
beta are fitted on the 2 other folds only, then applied unchanged to the held-out cameras. The reported error
is pooled over the 3 held-out folds. The final calibration (used for inference) is refitted on all videos.

    python -m flow_estimation.calibrate --calib data/calibration --dets data/detections/calibration_yolo11x_1280.jsonl.gz \
        --out flow_estimation/calibration/yolo11x_1280.json
"""
from __future__ import annotations

import argparse
import collections
import json
import random
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from .common import CLASSES, GROUP, count_new, load_detections, wape

GAPS = (5, 10, 15, 20, 30)
CONFS = (0.1, 0.15, 0.2, 0.25, 0.3, 0.35)
MIX_P10 = 0.58          # share of 10 s gaps in the real morning feed (rest 15 s)
_G = {}                 # worker globals


def schedules(frames, seed_key):
    """Snapshot sequences of one video for every gap (frames: list of dicts with 'T', 'image')."""
    byT = {r["T"]: r for r in frames}; tmax = max(byT)
    out = {str(dt): [byT[t] for t in np.arange(0, tmax + 1, dt) if t in byT] for dt in GAPS}
    rng = random.Random(seed_key); seq, t = [], 0.0
    while t <= tmax:
        if t in byT:
            seq.append(byT[t])
        t += 10 if rng.random() < MIX_P10 else 15
    out["mix"] = seq
    return out


def _work(v):
    """new-vehicle counts of video v for every (sequence, threshold): {(seq, conf): (new[K,4], dts[K])}"""
    D, sch = _G["dets"], _G["sch"][v]
    res = {}
    for name, seq in sch.items():
        dts = np.diff([r["T"] for r in seq], prepend=np.nan)
        for conf in CONFS:
            frames = [[d for d in D[r["image"]] if d[5] >= conf and d[0] in GROUP] for r in seq]
            n = count_new(frames); n[0] = 0                     # first image: no previous one
            res[(name, conf)] = (n, dts)
    return v, res


class Fitter:
    def __init__(self, R, Y):
        self.R, self.Y = R, Y

    def N(self, v, seq, cm, co):
        """counted new vehicles: motorcycles at threshold cm, car/truck/bus at threshold co"""
        return np.r_[self.R[v][(seq, cm)][0][:, 0].sum(), self.R[v][(seq, co)][0][:, 1:].sum(0)]

    def beta(self, vids, seq, cm, co):
        A = np.array([self.N(v, seq, cm, co) for v in vids]); Y = np.array([self.Y[v] for v in vids])
        return np.array([float(np.median(Y[A[:, c] > 5, c] / A[A[:, c] > 5, c])) for c in range(4)])

    def err(self, vids, seq, b, cm, co):
        P = np.array([self.N(v, seq, cm, co) * b for v in vids]); Y = np.array([self.Y[v] for v in vids])
        return [wape(Y[:, c], P[:, c]) for c in range(4)]

    def thresholds(self, vids, seq):
        cm = min(CONFS, key=lambda c: self.err(vids, seq, self.beta(vids, seq, c, 0.25), c, 0.25)[0])
        co = min(CONFS, key=lambda c: sum(self.err(vids, seq, self.beta(vids, seq, cm, c), cm, c)[1:]))
        return cm, co


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calib", type=Path, default=Path("data/calibration"), help="folder with videos.json, frames.jsonl, camera_folds.json")
    ap.add_argument("--dets", type=Path, required=True, help="detections of the calibration snapshots (detect.py)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()

    V = json.load(open(a.calib / "videos.json", encoding="utf-8"))
    folds = json.load(open(a.calib / "camera_folds.json", encoding="utf-8"))
    fr = collections.defaultdict(list)
    for l in open(a.calib / "frames.jsonl", encoding="utf-8"):
        r = json.loads(l); fr[r["video"]].append(r)
    vids = sorted(v for v in fr if v in V)
    FOLD = {v: folds[V[v]["camera"]] for v in vids}
    Y = {v: np.array([V[v]["flow"][c] for c in CLASSES], float) for v in vids}

    _G["dets"] = load_detections(a.dets)
    _G["sch"] = {v: schedules(sorted(fr[v], key=lambda r: r["T"]), v) for v in vids}
    with Pool(a.workers) as p:
        R = dict(p.map(_work, vids, chunksize=4))
    F = Fitter(R, Y)

    out = {"detector": {"model": "yolo11x (COCO, Ultralytics)", "imgsz": 1280, "input": "512x288 JPEG"},
           "classes": CLASSES, "gaps_s": list(GAPS), "n_videos": len(vids),
           "n_cameras": len({V[v]["camera"] for v in vids}), "cv": {}, "folds": {}}
    for seq in [str(g) for g in GAPS] + ["mix"]:
        ys, ps = [], []
        for f in sorted(set(FOLD.values())):
            tr = [v for v in vids if FOLD[v] != f]; te = [v for v in vids if FOLD[v] == f]
            fit_seq = "10" if seq == "mix" else seq
            cm, co = F.thresholds(tr, fit_seq)
            if seq == "mix":                      # mixed gaps: beta of each pair chosen by its own gap
                b = {dt: F.beta(tr, dt, cm, co) for dt in ("10", "15")}
                for v in te:
                    p = np.zeros(4)
                    for src, sl, c in (("m", slice(0, 1), cm), ("o", slice(1, 4), co)):
                        n, dts = R[v][("mix", c)]
                        bb = np.where((dts == 10)[:, None], b["10"][None, :], b["15"][None, :])
                        p[sl] = (n * bb).sum(0)[sl]
                    ys.append(Y[v]); ps.append(p)
            else:
                b = F.beta(tr, seq, cm, co)
                for v in te:
                    ys.append(Y[v]); ps.append(F.N(v, seq, cm, co) * b)
                if seq == "10":
                    out["folds"][str(f)] = {"train_videos": len(tr), "test_videos": len(te),
                                            "test_cameras": sum(1 for x in folds.values() if x == f),
                                            "conf_motorcycle": cm, "conf_other": co, "beta_10s": np.round(b, 3).tolist(),
                                            "test_wape": np.round(F.err(te, "10", b, cm, co), 1).tolist()}
        ys, ps = np.array(ys), np.array(ps)
        out["cv"][seq] = {"wape": [round(wape(ys[:, c], ps[:, c]), 2) for c in range(4)] + [round(wape(ys.sum(1), ps.sum(1)), 2)],
                          "bias": [round(float((ps[:, c].sum() / ys[:, c].sum() - 1) * 100), 2) for c in range(4)]
                                  + [round(float((ps.sum() / ys.sum() - 1) * 100), 2)]}
        print(f"gap {seq:>3}: WAPE moto/car/truck/bus/total " + " ".join(f"{x:5.1f}" for x in out["cv"][seq]["wape"]), flush=True)

    cm, co = F.thresholds(vids, "10")       # final: thresholds at 10 s on all videos, beta of every gap at them
    out["conf"] = {"motorcycle": cm, "other": co}
    out["beta"] = {str(g): np.round(F.beta(vids, str(g), cm, co), 4).tolist() for g in GAPS}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print("final conf", out["conf"], "beta", out["beta"], "->", a.out)


if __name__ == "__main__":
    main()
