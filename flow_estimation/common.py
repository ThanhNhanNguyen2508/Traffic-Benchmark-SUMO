"""Shared pieces of the flow estimator: classes, detection loading, new-vehicle counting, metrics.

Coordinates: every detection is stored in 1920x1080 pixel space (detections made on 512x288 images are
scaled by 3.75), so all pixel thresholds below mean the same thing in calibration and inference.
A detection is [cls, x_center, y_bottom, height, width, conf]; (x_center, y_bottom) is the footpoint.
"""
from __future__ import annotations

import collections
import gzip
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

CLASSES = ["motorcycle", "car", "truck", "bus"]                  # output classes, in this order everywhere
CI = {c: i for i, c in enumerate(CLASSES)}
# matching groups: a box can only be paired with a box of the same group (detector often swaps truck/bus)
GROUP = {"motorcycle": 0, "bicycle": 0, "car": 1, "truck": 2, "bus": 2}
CELL = 48              # static-mask grid cell, px in 1080p space
STATIC_FRAC = 0.6      # a cell occupied in >= 60% of the images of a block is a parking spot
BLOCK_S = 600          # static-mask block length (s); calibration videos are ~10 min long
MAX_GAP_S = 35         # image pairs further apart than this are treated as missing data


def open_text(path):
    path = Path(path)
    return gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else open(path, encoding="utf-8")


def load_detections(path) -> dict[str, list]:
    """{image key: [[cls, x, y_bottom, h, w, conf], ...]} from a (gzipped) jsonl written by detect.py."""
    out = {}
    with open_text(path) as fh:
        for line in fh:
            r = json.loads(line)
            out[r["image"]] = r["dets"]
    return out


def keep(dets, conf_moto: float, conf_other: float):
    """Box-confidence filter: one threshold for the two-wheel group (motorcycle, and bicycle, which takes part in
    matching but is never counted) and one for car/truck/bus. Persons are dropped."""
    return [d for d in dets if (d[0] in ("motorcycle", "bicycle") and d[5] >= conf_moto) or
            (d[0] in ("car", "truck", "bus") and d[5] >= conf_other)]


def _cell(d):
    return GROUP[d[0]], int(d[1] // CELL), int(d[2] // CELL)


def static_cells(frames, frac=STATIC_FRAC) -> set:
    """Cells (group, cx, cy) occupied in >= frac of the frames: parked vehicles."""
    cnt = collections.Counter()
    for fr in frames:
        cnt.update({_cell(d) for d in fr})
    return {c for c, m in cnt.items() if m >= frac * len(frames)}


def match_prev(cur, prev):
    """Hungarian matching of footpoints of image k (cur) to image k-1 (prev).
    A pair is allowed only for the same group and distance < 0.5 * max(box heights) + 4 px.
    Returns {index in cur: index in prev}."""
    if not cur or not prev:
        return {}
    C = np.array([[d[1], d[2], d[3], GROUP[d[0]]] for d in cur])
    P = np.array([[d[1], d[2], d[3], GROUP[d[0]]] for d in prev])
    dist = np.hypot(C[:, None, 0] - P[None, :, 0], C[:, None, 1] - P[None, :, 1])
    tol = 0.5 * np.maximum(C[:, None, 2], P[None, :, 2]) + 4
    cost = np.where((dist < tol) & (C[:, None, 3] == P[None, :, 3]), dist / tol, 9.0)
    ri, ci = linear_sum_assignment(cost)
    return {int(i): int(j) for i, j in zip(ri, ci) if cost[i, j] < 1}


def count_new(frames, frac=STATIC_FRAC, mask=None):
    """frames: list of detection lists in time order. Returns new[K, 4]: vehicles of each class in image k
    that were not in image k-1 (after removing parked cells). new[0] counts every vehicle of image 0;
    callers drop it because there is no previous image."""
    if mask is None:
        mask = static_cells(frames, frac) if frac is not None else set()
    new = np.zeros((len(frames), 4))
    prev = []
    for k, fr in enumerate(frames):
        fr = [d for d in fr if _cell(d) not in mask]
        m = match_prev(fr, prev)
        for i, d in enumerate(fr):
            if i not in m and d[0] in CI:
                new[k, CI[d[0]]] += 1
        prev = fr
    return new


def wape(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.abs(p - y).sum() / max(y.sum(), 1) * 100)


def nearest_gap(dt, gaps):
    gaps = np.asarray(sorted(gaps))
    return gaps[np.argmin(np.abs(np.asarray(dt)[:, None] - gaps[None, :]), axis=1)]
