#!/usr/bin/env python3
"""Figures for the report (English labels): hourly tables, hourly stacked bars, 10-min series, vehicle mix,
validation error, demand settings.

    python -m flow_estimation.figures --flow results/flow --calibration flow_estimation/calibration/yolo11x_1280.json \
        --out results/figures
"""
from __future__ import annotations

import argparse
import datetime as dt_
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates          # noqa: E402
import matplotlib.pyplot as plt            # noqa: E402
import numpy as np                         # noqa: E402

CL = ["motorcycle", "car", "truck", "bus"]; CLN = ["Motorcycle", "Car", "Truck", "Bus"]
SER = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
NAME = {"Cam_Ly_Thuong_Kiet_Bac_Hai": "Ly Thuong Kiet – Bac Hai", "Cam_Thanh_Thai_To_Hien_Thanh": "Thanh Thai – To Hien Thanh",
        "Cam_To_Hien_Thanh_1": "To Hien Thanh 1", "Cam_To_Hien_Thanh_2": "To Hien Thanh 2"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})
fmt = lambda x: f"{int(round(x)):,}"
thousands = matplotlib.ticker.FuncFormatter(lambda v, _: f"{int(v):,}")


def session_label(s):
    d, t = s.split("_"); return f"{dt_.date.fromisoformat(d).strftime('%d %b %Y')}, {t[:2]}:{t[2:4]}–{t[5:7]}:{t[7:]}"


def table(rows, cols, title, note, path, numeric_from=2, X=None):
    n = len(rows); rh = 0.46; W = 10.4
    X = np.array(X if X is not None else [0.0, 3.35, 5.55, 6.95, 8.05, 9.05, 10.4]); W = X[-1]
    H = rh * (n + 1)
    fig = plt.figure(figsize=(11.2, rh * (n + 1) + 1.3)); ax = fig.add_axes([0.02, 0.02, 0.96, 0.96]); ax.axis("off")
    ax.set_xlim(-0.2, W + 0.2); ax.set_ylim(-0.8, H + 0.75)
    ax.text(-0.1, H + 0.45, title, fontsize=15, weight="bold", color=INK, va="center")
    ax.add_patch(plt.Rectangle((-0.1, H - rh), W + 0.2, rh, color="#eef3fb", lw=0))
    for j, c in enumerate(cols):
        ax.text(X[j], H - rh / 2, c, ha="left" if j < numeric_from else "right", va="center", weight="bold", color=INK, fontsize=12)
    per = max(1, sum(1 for r in rows if r[0]) and n // max(1, sum(1 for r in rows if r[0])))
    for i, r in enumerate(rows):
        y = H - rh * (i + 1.5)
        if (i // per) % 2:
            ax.add_patch(plt.Rectangle((-0.1, y - rh / 2), W + 0.2, rh, color="#f6f6f4", lw=0))
        for j, v in enumerate(r):
            ax.text(X[j], y, v, ha="left" if j < numeric_from else "right", va="center", fontsize=12, color=INK,
                    weight="bold" if j in (0, len(cols) - 1) else "normal")
    ax.plot([-0.1, W + 0.1], [0, 0], color=GRID, lw=1)
    ax.text(-0.1, -0.45, note, color=INK2, fontsize=10, va="center")
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.15); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--flow", type=Path, default=Path("results/flow"))
    ap.add_argument("--calibration", type=Path, default=Path("flow_estimation/calibration/yolo11x_1280.json"))
    ap.add_argument("--out", type=Path, default=Path("results/figures"))
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    F = json.load(open(a.flow / "flow.json")); cal = json.load(open(a.calibration))
    sessions = sorted({k.split("|")[0] for k in F}); cams = [c for c in NAME if any(k.endswith(c) for k in F)]

    for s in sessions:
        rows = []
        for c in cams:
            for i, h in enumerate(F[f"{s}|{c}"]["bins"]):
                v = h["veh_per_h"]; rows.append([NAME[c] if i == 0 else "", f"{h['from']}–{h['to']}", *[fmt(v[k]) for k in CL], fmt(h["total"])])
        table(rows, ["Camera", "Time", *CLN, "Total"], f"{session_label(s)} – estimated flow (vehicles / hour)",
              "All directions through the camera view combined. Snapshots every 10–15 s (512×288).", a.out / f"table_{s}.png")
        # hourly stacked bars
        fig, axs = plt.subplots(1, len(cams), figsize=(14, 4.6), sharey=True)
        for ax, c in zip(np.atleast_1d(axs), cams):
            H = F[f"{s}|{c}"]["bins"]; x = np.arange(len(H)); bot = np.zeros(len(H))
            for k, cl in enumerate(CL):
                v = np.array([h["veh_per_h"][cl] for h in H]); ax.bar(x, v, bottom=bot, width=0.7, color=SER[k], label=CLN[k], edgecolor=SURF, linewidth=1.2); bot += v
            for xi, h in zip(x, H):
                ax.text(xi, h["total"] + 150, fmt(h["total"]), ha="center", fontsize=9, color=INK2)
            ax.set_xticks(x, [h["from"] for h in H]); ax.set_title(NAME[c], fontsize=11.5, color=INK, weight="bold", loc="left")
            ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True); ax.yaxis.set_major_formatter(thousands)
        np.atleast_1d(axs)[0].set_ylabel("vehicles / hour")
        h_, l_ = np.atleast_1d(axs)[0].get_legend_handles_labels(); fig.legend(h_, l_, loc="upper right", ncol=4, frameon=False, bbox_to_anchor=(0.995, 0.97))
        fig.suptitle(f"{session_label(s)} – flow by camera", x=0.01, ha="left", fontsize=14, weight="bold", color=INK)
        fig.tight_layout(rect=(0, 0, 1, 0.88)); fig.savefig(a.out / f"bars_{s}.png", dpi=220); plt.close(fig)
        # 10-min series
        fig, axs = plt.subplots(2, 2, figsize=(13, 7.2), sharex=True)
        for j, (ax, cn) in enumerate(zip(axs.flat, CLN)):
            for ci, c in enumerate(cams):
                d = F[f"{s}|{c}"]; day = dt_.date.fromisoformat(d["start"][:10])
                xs = [dt_.datetime.combine(day, dt_.time(*map(int, h["from"].split(":")))) + dt_.timedelta(minutes=5) for h in d["series_10min"]]
                ax.plot(xs, [h["veh_per_h"][CL[j]] for h in d["series_10min"]], color=SER[ci], lw=2, marker="o", ms=3.5, label=NAME[c])
            ax.set_title(cn, loc="left", color=INK, weight="bold"); ax.grid(axis="y", color=GRID, lw=0.8); ax.set_ylim(bottom=0)
            ax.yaxis.set_major_formatter(thousands); ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            if j % 2 == 0:
                ax.set_ylabel("vehicles / hour")
        h_, l_ = axs[0, 0].get_legend_handles_labels(); fig.legend(h_, l_, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.955))
        fig.suptitle(f"{session_label(s)} – flow by class, 10-minute resolution", x=0.01, ha="left", fontsize=14, weight="bold", color=INK)
        fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(a.out / f"series_{s}.png", dpi=220); plt.close(fig)

    # vehicle mix over all camera-hours
    allh = [h for k, d in F.items() for h in d["bins"]]
    avg = np.array([np.mean([h["veh_per_h"][c] for h in allh]) for c in CL]); sh = avg / avg.sum() * 100
    fig, ax = plt.subplots(figsize=(10, 6.2))
    w, _ = ax.pie(sh, colors=SER, startangle=90, counterclock=False, wedgeprops=dict(edgecolor=SURF, linewidth=2.5))
    for k, wd in enumerate(w):
        ang = np.deg2rad((wd.theta1 + wd.theta2) / 2)
        if sh[k] >= 8:
            ax.text(0.62 * np.cos(ang), 0.62 * np.sin(ang), f"{sh[k]:.1f}%", ha="center", va="center", fontsize=15, color="#ffffff", weight="bold")
    ax.legend(w, [f"{n}  {p:.1f}%  (~{fmt(v)} veh/h)" for n, p, v in zip(CLN, sh, avg)], loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=13)
    ax.set_title("Average vehicle mix (all cameras and sessions)", loc="left", fontsize=15, weight="bold", color=INK)
    fig.savefig(a.out / "vehicle_mix.png", dpi=220, bbox_inches="tight"); plt.close(fig)

    # validation error (cross-validation, mixed 10/15 s gaps)
    cv = cal["cv"]["mix"]; fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.2)); x = np.arange(5); cols = SER + [INK2]
    for ax, key, ttl in ((axs[0], "wape", "Absolute error (WAPE, %)"), (axs[1], "bias", "Systematic bias (%)")):
        v = np.array(cv[key]); ax.bar(x, v, color=cols, width=0.62)
        for xi, vi in zip(x, v):
            ax.text(xi, vi + (1 if vi >= 0 else -1), f"{vi:+.0f}%" if key == "bias" else f"{vi:.0f}%", ha="center", va="bottom" if vi >= 0 else "top", fontsize=10, color=INK)
        ax.set_xticks(x, CLN + ["All vehicles"], fontsize=10); ax.set_title(ttl, loc="left", color=INK, weight="bold"); ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)
        if key == "bias":
            ax.axhline(0, color=INK2, lw=1); ax.set_ylim(min(v) - 6, 6)
        else:
            ax.set_ylim(0, max(v) + 8)
    fig.suptitle(f"Cross-validation on {cal['n_videos']} ground-truth videos ({cal['n_cameras']} cameras, 512×288, 10/15 s snapshots, ~10-min videos)",
                 x=0.01, ha="left", fontsize=12, weight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(a.out / "validation_error.png", dpi=220); plt.close(fig)

    ds = a.flow / "demand_settings.json"
    if ds.is_file():
        S = json.load(open(ds)); rows = []
        for name, v in S.items():
            m = v["mean_of_cameras"]; rows.append([name.capitalize(), f"{v['from']}–{v['to']}, {session_label(v['session'])[:11]}", *[fmt(m[c]) for c in CL], fmt(m["total"])])
        table(rows, ["Setting", "Time window", *CLN, "Total"], "Demand settings – vehicles / hour, mean of the cameras",
              "Each SUMO scenario uses the per-camera values of its window (results/flow/demand_settings.json).", a.out / "demand_settings.png",
              X=[0.0, 1.7, 6.4, 7.6, 8.7, 9.7, 11.2])
    print("figures ->", a.out)


if __name__ == "__main__":
    main()
