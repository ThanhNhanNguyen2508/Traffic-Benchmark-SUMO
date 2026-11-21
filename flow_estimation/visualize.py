#!/usr/bin/env python3
"""Illustrations of the counting step on real city-camera images.

  matching: two consecutive images; GREEN = paired with image k-1 by Hungarian matching (stood still or already
            counted, +0), ORANGE = new vehicle (+1), GREY = inside a parked cell (ignored)
  parked:   one row per example: the parked cell (orange square = 48 px cell holding the footpoint) at the start,
            middle and end of its 10-min block, with the share of images in which a vehicle sits there

    python -m flow_estimation.visualize matching --root data/city_cam --session 2026-10-29_0850-1250 \
        --camera Cam_Thanh_Thai_To_Hien_Thanh --time 10:52 --out docs/figures/hungarian_matching.png
    python -m flow_estimation.visualize parked --root data/city_cam \
        --example 2026-11-07_1306-1706 Cam_Thanh_Thai_To_Hien_Thanh 16:18 --out docs/figures/parked_vehicles.png
(both need --dets and --calibration, defaults below)
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt_
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .common import BLOCK_S, CELL, GROUP, STATIC_FRAC, _cell, keep, load_detections, match_prev, static_cells
from .infer import list_frames

GN = {0: "motorcycle", 1: "car", 2: "truck/bus"}
BLUE, GREEN, ORANGE, GREY, INK = (42, 120, 214), (27, 175, 122), (235, 104, 52), (170, 170, 170), (20, 33, 61)


def font(size, bold=True):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "DejaVuSans-Bold.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


def camera_frames(a, session, camera):
    rows = sorted((r for r in list_frames(a.root, [session], set()) if r["camera"] == camera), key=lambda r: r["t"])
    ts = np.array([dt_.datetime.fromisoformat(r["t"]).timestamp() for r in rows])
    return rows, ts, ((ts - ts[0]) // BLOCK_S).astype(int)


def cmd_matching(a, D, cal):
    rows, ts, block = camera_frames(a, a.session, a.camera)
    dets = [keep(D[r["image"]], cal["conf"]["motorcycle"], cal["conf"]["other"]) for r in rows]
    k = next(i for i, r in enumerate(rows) if r["t"][11:16] >= a.time)
    mask = static_cells([dets[i] for i in np.where(block == block[k])[0]], STATIC_FRAC)
    split = lambda fr: ([d for d in fr if _cell(d) not in mask], [d for d in fr if _cell(d) in mask])
    prev, _ = split(dets[k - 1]); cur, parked = split(dets[k]); m = match_prev(cur, prev)
    S = 1280 / 1920
    load = lambda i: Image.open(a.root / rows[i]["image"]).convert("RGB").resize((1280, 720), Image.LANCZOS)
    A, B = load(k - 1), load(k); da, db = ImageDraw.Draw(A), ImageDraw.Draw(B)

    def box(dr, d, col, w=3):
        x, y, h, ww = d[1] * S, d[2] * S, d[3] * S, d[4] * S
        dr.rectangle([x - ww / 2, y - h, x + ww / 2, y], outline=col, width=w); return x, y
    for d in prev:
        box(da, d, BLUE)
    for d in parked:
        box(db, d, GREY, 2)
    for i, d in enumerate(cur):
        col = GREEN if i in m else ORANGE; x, y = box(db, d, col)
        db.ellipse([x - 5, y - 5, x + 5, y + 5], fill=col)
        if i in m:
            p = prev[m[i]]; px, py = p[1] * S, p[2] * S
            db.ellipse([px - 6, py - 6, px + 6, py + 6], outline=BLUE, width=3); db.line([px, py, x, y], fill=(255, 255, 255), width=3)
    W = Image.new("RGB", (2590, 880), (251, 251, 248)); W.paste(A, (0, 110)); W.paste(B, (1310, 110)); dw = ImageDraw.Draw(W)
    F, Fs = font(22), font(16)
    dw.text((10, 20), f"Image k-1 ({rows[k-1]['t'][11:19]}) - blue = boxes in the previous image", fill=INK, font=F)
    dw.text((1320, 20), f"Image k ({rows[k]['t'][11:19]}, dt = {ts[k]-ts[k-1]:.1f} s)", fill=INK, font=F)
    dw.text((1320, 55), "GREEN = matched to k-1 (stood still / already counted, +0)   ORANGE = NEW (+1)", fill=INK, font=Fs)
    dw.text((1320, 80), "blue ring = position in k-1, white line = Hungarian pair   GREY = parked cell (ignored)", fill=INK, font=Fs)
    new = collections.Counter(d[0] for i, d in enumerate(cur) if i not in m)
    dw.text((10, 845), f"k-1: {len(prev)} vehicles   k: {len(cur)} vehicles -> matched {len(m)}, NEW {len(cur)-len(m)} {dict(new)}, parked {len(parked)}", fill=INK, font=F)
    a.out.parent.mkdir(parents=True, exist_ok=True); W.save(a.out); print("->", a.out)


def cmd_parked(a, D, cal):
    S = 1024 / 1920; Fs, F = font(19, False), font(22); panels = []
    for session, camera, t in a.example:
        rows, ts, block = camera_frames(a, session, camera)
        dets = [keep(D[r["image"]], cal["conf"]["motorcycle"], cal["conf"]["other"]) for r in rows]
        k = next(i for i, r in enumerate(rows) if r["t"][11:16] >= t); ii = np.where(block == block[k])[0]
        cnt = collections.Counter()
        for i in ii:
            cnt.update({_cell(d) for d in dets[i]})
        cells = [(c, n) for c, n in cnt.items() if n >= STATIC_FRAC * len(ii)]
        if not cells:
            print(f"no parked cell in {session} {camera} {t}"); continue
        (g, cx, cy), n = max(cells, key=lambda x: x[1]); x0, y0, cs = cx * CELL * S, cy * CELL * S, CELL * S
        P = Image.new("RGB", (1880, 470), (251, 251, 248)); dp = ImageDraw.Draw(P)
        dp.text((10, 10), f"{camera} ({session}, block from {rows[ii[0]]['t'][11:16]}): {GN[g]} in the orange cell in {n}/{len(ii)} images = {n/len(ii)*100:.0f}% -> masked", fill=INK, font=F)
        for j, i in enumerate([ii[0], ii[len(ii) // 2], ii[-1]]):
            im = Image.open(a.root / rows[i]["image"]).convert("RGB").resize((1024, 576), Image.LANCZOS)
            L = max(0, min(1024 - 300, x0 + cs / 2 - 150)); T = max(0, min(576 - 200, y0 + cs / 2 - 100))
            crop = im.crop((L, T, L + 300, T + 200)).resize((600, 400), Image.LANCZOS); dr = ImageDraw.Draw(crop)
            dr.rectangle([(x0 - L) * 2, (y0 - T) * 2, (x0 + cs - L) * 2, (y0 + cs - T) * 2], outline=ORANGE, width=4)
            hit = any(_cell(d) == (g, cx, cy) for d in dets[i])
            dr.rectangle([0, 0, 600, 34], fill=INK); dr.text((8, 6), f"{rows[i]['t'][11:19]}  {'detected' if hit else 'NOT detected'}", fill=(247, 247, 244), font=Fs)
            P.paste(crop, (10 + j * 620, 55))
        panels.append(P)
    out = Image.new("RGB", (1880, 470 * len(panels)), (251, 251, 248))
    for j, p in enumerate(panels):
        out.paste(p, (0, 470 * j))
    a.out.parent.mkdir(parents=True, exist_ok=True); out.save(a.out); print("->", a.out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("matching", "parked"):
        sp = sub.add_parser(name)
        sp.add_argument("--root", type=Path, default=Path("data/city_cam"))
        sp.add_argument("--dets", type=Path, default=Path("data/detections/city_cam_yolo11x_1280.jsonl.gz"))
        sp.add_argument("--calibration", type=Path, default=Path("flow_estimation/calibration/yolo11x_1280.json"))
        sp.add_argument("--out", type=Path, required=True)
    sub.choices["matching"].add_argument("--session", required=True)
    sub.choices["matching"].add_argument("--camera", required=True)
    sub.choices["matching"].add_argument("--time", required=True, help="HH:MM: first image at or after this time is image k")
    sub.choices["parked"].add_argument("--example", nargs=3, action="append", required=True, metavar=("SESSION", "CAMERA", "HH:MM"))
    a = ap.parse_args()
    D = load_detections(a.dets); cal = json.load(open(a.calibration))
    {"matching": cmd_matching, "parked": cmd_parked}[a.cmd](a, D, cal)


if __name__ == "__main__":
    main()
