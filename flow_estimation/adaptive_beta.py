"""Density-adaptive conversion factor (pure numpy, no SUMO/torch).

    beta(frame k, class c, gap g) = beta_star[g][c] * clip(D[k,c] / dbar[g][c], 0.25, 3) ** gamma[c]

D[k,c] = non-static detections of class c in snapshot k (exactly what count_new sees). Frame 0 has no "new"
count, so frames k >= 1 are used for dbar and for every weighted sum. gamma == 0 gives the fixed beta.
Used by calibrate.py (fit) and, later, by the inference scripts (apply).
"""
from __future__ import annotations

import numpy as np

from .common import CLASSES, CI, _cell, wape

CLIP = (0.25, 3.0)
GAMMA_GRID = tuple(round(x, 3) for x in np.arange(0, 0.5 + 1e-9, 0.025))
MIN_GAIN = 0.5       # fitted gamma is kept only if it lowers the mean CV WAPE (over gaps) by at least this many points
MIN_NEW = 5          # a video needs more than this many counted new vehicles to enter the beta median


def frame_density(frames, mask) -> np.ndarray:
    """[K, 4] number of detections per class in each frame, static-mask cells removed."""
    dens = np.zeros((len(frames), len(CLASSES)))
    for k, fr in enumerate(frames):
        for d in fr:
            if d[0] in CI and _cell(d) not in mask:
                dens[k, CI[d[0]]] += 1
    return dens


def factor(dens, dbar, gamma, clip=CLIP):
    """clip(dens / dbar) ** gamma, broadcast over arrays. dbar <= 0 -> factor 1; gamma == 0 -> exactly ones."""
    dens = np.asarray(dens, float)
    dbar = np.asarray(dbar, float)
    gamma = np.asarray(gamma, float)
    out = np.ones(np.broadcast(dens, dbar, gamma).shape)
    ok = np.broadcast_to(dbar > 0, out.shape) & np.broadcast_to(gamma != 0, out.shape)
    if ok.any():
        x = np.clip(dens / np.where(dbar > 0, dbar, 1.0), *clip)
        out = np.where(ok, np.broadcast_to(x, out.shape) ** gamma, 1.0)
    return out


def weighted_new(new, dens, dbar, gamma):
    """sum over frames k >= 1 of new[k] * factor(dens[k]). 1-D inputs (one class) -> scalar; [K,4] -> [4]."""
    new, dens = np.asarray(new, float), np.asarray(dens, float)
    return (new[1:] * factor(dens[1:], dbar, gamma)).sum(axis=0)


def _beta_median(ws):
    """ws: [(Y, weighted_new, raw N)]. Median of Y / weighted_new over videos with N > MIN_NEW; NaN if none."""
    r = [y / w for y, w, n in ws if n > MIN_NEW and w > 0]
    return float(np.median(r)) if r else float("nan")


def _class_stats(S, c, dbar, gamma):
    """per video: (Y, weighted_new, raw N) of class c."""
    return [(float(s["Y"][c]), float(weighted_new(s["new"][:, c], s["dens"][:, c], dbar, gamma)),
             float(np.asarray(s["new"])[1:, c].sum())) for s in S]


def _dbar(S, c):
    d = np.concatenate([np.asarray(s["dens"])[1:, c] for s in S]) if S else np.zeros(0)
    return float(d.mean()) if d.size else 0.0


def _cv_class(S, c, gamma, folds):
    """pooled 3-fold CV of class c on samples S of one gap: dbar and beta from train folds only. -> (y[], p[])."""
    ys, ps = [], []
    for f in folds:
        tr = [s for s in S if s["fold"] != f]
        te = [s for s in S if s["fold"] == f]
        db = _dbar(tr, c)
        st = _class_stats(tr, c, db, gamma)
        beta = _beta_median(st)
        for s in te:
            ys.append(float(s["Y"][c]))
            ps.append(beta * float(weighted_new(s["new"][:, c], s["dens"][:, c], db, gamma)))
    return np.array(ys), np.array(ps)


def _score(S, c, gamma, folds):
    y, p = _cv_class(S, c, gamma, folds)
    return wape(y, p) if np.isfinite(p).all() else float("inf")


def fit(samples, gaps, classes=CLASSES, gamma=None, grid=GAMMA_GRID):
    """samples: [{video, fold, gap (str), new[K,4], dens[K,4], Y[4]}, ...]. Returns the JSON-ready block."""
    gaps = [str(g) for g in gaps]
    by = {g: [s for s in samples if str(s["gap"]) == g] for g in gaps}
    folds = sorted({s["fold"] for s in samples})
    nc = len(classes)

    gscores, gam, gain = {}, {}, {}
    for c, name in enumerate(classes):
        if gamma is not None:
            gam[name] = float(gamma.get(name, 0.0))
            continue
        sc = []
        for g in grid:
            sc.append((float(g), float(np.mean([_score(by[k], c, g, folds) for k in gaps]))))
        gscores[name] = sc
        base = sc[0][1] if sc[0][0] == 0 else _mean_score(by, gaps, c, 0.0, folds)
        best = min(sc, key=lambda t: (t[1], t[0]))      # ties -> smaller gamma
        gain[name] = base - best[1]
        gam[name] = best[0] if gain[name] >= MIN_GAIN else 0.0
    garr = np.array([gam[n] for n in classes])

    block = {"form": "beta_star[gap][c] * clip(D/dbar[gap][c], clip[0], clip[1]) ** gamma[c]",
             "clip": list(CLIP), "gamma": {n: round(gam[n], 3) for n in classes},
             "gamma_source": "cli" if gamma is not None else "fit", "gamma_grid": list(grid),
             "density_def": ("non-static detections of the class per frame, after conf filter (motorcycle at "
                             "conf.motorcycle, others at conf.other) and static-cell mask; frames k>=1; "
                             "full frame (no ROI)"),
             "dbar": {}, "beta_star": {}, "cv": {}, "cv_gamma_scores": {}, "density_stats": {},
             "min_gain": MIN_GAIN, "gamma_gain": {n: round(v, 2) for n, v in gain.items()},
             "note": "cv uses the final thresholds for every fold"}
    for n, sc in gscores.items():
        block["cv_gamma_scores"][n] = [[g, round(w, 2)] for g, w in sc]

    for g in gaps:
        S = by[g]
        dbar = np.array([round(_dbar(S, c), 6) for c in range(nc)])     # stored value == value used for beta_star
        block["dbar"][g] = {n: float(dbar[c]) for c, n in enumerate(classes)}
        block["beta_star"][g] = [round(_beta_median(_class_stats(S, c, dbar[c], garr[c])), 4)
                                 for c in range(nc)]
        cv, Y = {}, {}
        for tag, gg in (("fixed", np.zeros(nc)), ("adaptive", garr)):
            res = [_cv_class(S, c, gg[c], folds) for c in range(nc)]
            Y, P = np.array([r[0] for r in res]).T, np.array([r[1] for r in res]).T      # [videos, classes]
            for c, n in enumerate(classes):
                cv.setdefault(n, {})["wape_" + tag] = round(wape(Y[:, c], P[:, c]), 2)
                cv[n]["bias_" + tag] = round(float((P[:, c].sum() / max(Y[:, c].sum(), 1) - 1) * 100), 2)
            cv.setdefault("total", {})["wape_" + tag] = round(wape(Y.sum(1), P.sum(1)), 2)
            cv["total"]["bias_" + tag] = round(float((P.sum() / max(Y.sum(), 1) - 1) * 100), 2)
        block["cv"][g] = cv
        block["density_stats"][g] = {}
        for c, n in enumerate(classes):
            d = np.concatenate([np.asarray(s["dens"])[1:, c] for s in S])
            q = np.percentile(d, [5, 25, 50, 75, 95]) if d.size else [float("nan")] * 5
            block["density_stats"][g][n] = {"n_frames": int(d.size), "mean": round(float(d.mean()) if d.size else float("nan"), 2),
                                            **{k: round(float(v), 2) for k, v in zip(("p5", "p25", "p50", "p75", "p95"), q)}}
    return block


def _mean_score(by, gaps, c, g, folds):
    return float(np.mean([_score(by[k], c, g, folds) for k in gaps]))


def apply(new, dens, gap_per_frame, block):
    """per-frame beta [K,4] = beta_star[gap_k] * factor(dens_k, dbar[gap_k], gamma). `new` only gives the shape."""
    new, dens = np.asarray(new, float), np.asarray(dens, float)
    classes = CLASSES
    gam = np.array([block["gamma"][c] for c in classes], float)
    clip = tuple(block.get("clip", CLIP))
    beta = np.zeros(new.shape)
    for k, g in enumerate(gap_per_frame):
        g = str(g)
        db = np.array([block["dbar"][g][c] for c in classes], float)
        beta[k] = np.asarray(block["beta_star"][g], float) * factor(dens[k], db, gam, clip)
    return beta
