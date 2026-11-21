#!/usr/bin/env python3
"""Flow (vehicles / hour per class) of every camera over a clock window, from infer.py's perframe.json.

Used to define the benchmark demand settings (light / medium / heavy):

    python -m flow_estimation.window --perframe results/flow/perframe.json \
        --setting light  2026-11-12_0530-0700 05:30 06:30 \
        --setting medium 2026-11-07_1306-1706 13:06 14:06 \
        --setting heavy  2026-10-29_0850-1250 10:50 11:50 \
        --out results/flow/demand_settings.json
Window = images received in (from, to]; rate = sum(flow) / covered seconds * 3600.
"""
from __future__ import annotations

import argparse
import datetime as dt_
import json
from pathlib import Path

import numpy as np

from .common import CLASSES

TZ = dt_.timezone(dt_.timedelta(hours=7))


def window(P, session, a, b):
    out = {}
    for key, d in P.items():
        s, cam = key.split("|")
        if s != session:
            continue
        ts = np.array([dt_.datetime.fromisoformat(t).timestamp() for t in d["t"]])
        day = dt_.datetime.fromtimestamp(ts[0], TZ).date()
        t0, t1 = (dt_.datetime.combine(day, dt_.time(*map(int, x.split(":"))), TZ).timestamp() for x in (a, b))
        m = (ts > t0) & (ts <= t1); v = np.array(d["valid"]) & m
        cov = float(np.array(d["dt"])[v].sum())
        rate = np.array(d["flow"])[m].sum(0) / max(cov, 1) * 3600
        out[cam] = {c: int(round(x)) for c, x in zip(CLASSES, rate)} | {"total": int(round(rate.sum())), "covered_min": round(cov / 60, 1)}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--perframe", type=Path, required=True)
    ap.add_argument("--setting", nargs=4, action="append", metavar=("NAME", "SESSION", "FROM", "TO"), required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    P = json.load(open(a.perframe))
    res = {}
    for name, s, t0, t1 in a.setting:
        cams = window(P, s, t0, t1)
        mean = {c: int(round(np.mean([v[c] for v in cams.values()]))) for c in CLASSES + ["total"]}
        res[name] = {"session": s, "from": t0, "to": t1, "cameras": cams, "mean_of_cameras": mean}
        print(f"{name:7s} {s} {t0}-{t1}  mean: " + "  ".join(f"{c} {mean[c]:,}" for c in CLASSES + ["total"]))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
