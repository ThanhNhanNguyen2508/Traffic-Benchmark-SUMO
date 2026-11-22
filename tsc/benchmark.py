"""Benchmark post-processing helpers (stdlib only; no SUMO / torch import).

Used by the Full-mode notebook after ``run_tsc.py train`` / ``run_tsc.py eval``:
  * select_mplight_checkpoint - pick the MPLight checkpoint by validation travel time (never the test split);
  * aggregate_episodes        - per (mode, controller) mean / sample std over evaluation episodes;
  * write_rows                - CSV (+ optional JSON) writer for result tables;
  * format_mean_std_table     - printable mean+/-std table.
"""
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path


def _sha256_file(path):
    """SHA-256 hex digest of a file (streamed)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def select_mplight_checkpoint(run_dir, sumo_seed=None, out_path=None, mode=None):
    """Select the MPLight checkpoint of one training run from ``run_dir/progress.csv`` and write a selection JSON.

    Only rows with a ``test_travel_time`` value are candidates (greedy test on the VAL split, written every
    ``--test-every`` rounds). Key = (test_travel_time, test_teleports or +inf, round); the minimum wins.
    The test split is never used. Returns the selection dict (also written to ``out_path``,
    default ``run_dir.parent / "mplight_selection.json"``).
    """
    run_dir = Path(run_dir)
    with open(run_dir / "progress.csv", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    cand = []
    for r in rows:
        tt = (r.get("test_travel_time") or "").strip()
        if not tt:
            continue
        tp = (r.get("test_teleports") or "").strip()
        cand.append((float(tt), float(tp) if tp else float("inf"), int(r["round"]), r))
    if not cand:
        raise RuntimeError(f"{mode}: progress.csv has no validation row (test_travel_time); "
                           f"check that --test-every (TRAIN_TEST_EVERY) is > 0 and <= rounds")
    best = min(cand, key=lambda x: (x[0], x[1], x[2]))
    ckpt = run_dir / "models" / f"round_{best[2]}.pt"
    if not ckpt.is_file():
        raise FileNotFoundError(f"checkpoint round_{best[2]}.pt not found: {ckpt}")
    sel = dict(mode=mode,
               criterion="min test_travel_time (greedy trên split VAL, seed SUMO %s); hoà: ít test_teleports hơn, rồi round nhỏ hơn; KHÔNG dùng split test" % (sumo_seed,),
               round=best[2], checkpoint=str(ckpt), checkpoint_sha256=_sha256_file(ckpt),
               val_metrics={k: v for k, v in best[3].items() if k.startswith("test_")},
               n_val_points=len(cand), rounds_trained=len(rows), run_dir=str(run_dir), test_split_used_for_selection=False)
    out = Path(out_path) if out_path else run_dir.parent / "mplight_selection.json"
    out.write_text(json.dumps(sel, indent=1, ensure_ascii=False), encoding="utf-8")
    return sel


def aggregate_episodes(runs, ep_rows, metrics, key=("mode", "controller")):
    """One row per run base dict (given order): ``dict(base, n, seeds)`` + for each metric ``m`` (fmean) and
    ``m_std`` (sample std, ddof=1; NaN when n == 1). Episodes are matched to a run on the ``key`` fields."""
    agg = []
    for r0 in runs:
        eps = [e for e in ep_rows if all(e[k] == r0[k] for k in key)]
        row = dict(r0, n=len(eps), seeds=";".join(str(e["seed"]) for e in eps))
        for m in metrics:
            v = [float(e[m]) for e in eps]
            row[m] = statistics.fmean(v)
            row[m + "_std"] = statistics.stdev(v) if len(v) > 1 else math.nan
        agg.append(row)
    return agg


def _nan_to_none(d):
    """Replace top-level float NaN values of a dict by None (strict JSON)."""
    return {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in d.items()}


def write_rows(rows, csv_path, json_path=None, nan_to_none=True):
    """Write ``rows`` (list of dicts, columns = keys of the first row) as CSV and optionally JSON (indent=1).
    With ``nan_to_none`` top-level NaN floats become null in the JSON; pass False to dump the rows unchanged."""
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    if json_path is not None:
        data = [_nan_to_none(r) for r in rows] if nan_to_none else rows
        Path(json_path).write_text(json.dumps(data, indent=1), encoding="utf-8")


def format_mean_std_table(rows, metrics, n_label=None):
    """Printable table (string): title line, column header, one line per row; ``mean+/-std`` when n > 1 else mean."""
    fv = lambda row, m: (f"{row[m]:.2f}±{row[m + '_std']:.2f}" if row["n"] > 1 else f"{row[m]:.2f}")
    lines = ["mean±std (std mẫu, ddof=1; n=1 => không có std)" + (f" | {n_label}" if n_label else ""),
             f"{'mode':11s}{'controller':12s}{'n':>3s}" + "".join(f"{m:>{max(20, len(m) + 2)}s}" for m in metrics)]
    for row in rows:
        lines.append(f"{row['mode']:11s}{row['controller']:12s}{row['n']:3d}"
                     + "".join(f"{fv(row, m):>{max(20, len(m) + 2)}s}" for m in metrics))
    return "\n".join(lines)


# ---------------------------------------------------------------------------- J05: protocol helpers
LOWER_IS_BETTER = ("travel_time", "travel_time_arrived", "avg_queue", "teleports", "pending_insertion",
                   "in_network_at_end", "delay", "delay_with_depart", "trip_duration", "waiting_time",
                   "delay_all", "delay_with_depart_all", "waiting_time_all")
# Only "arrived" (throughput within the horizon). "departed" is demand/insertion-driven (what SUMO managed to
# insert), not a controller-quality score, so it is left out of the improvement table.
HIGHER_IS_BETTER = ("arrived",)


def _pct(base, x, lower):
    try:
        base, x = float(base), float(x)
    except (TypeError, ValueError):
        return math.nan
    if math.isnan(base) or math.isnan(x) or base == 0:
        return math.nan
    return ((base - x) if lower else (x - base)) / base * 100.0


def improvement_table(agg_rows, baseline="fixedtime", metrics=None, key="mode"):
    """Per ``key`` group (default mode): each controller row vs the ``baseline`` controller row of the same group.
    ``<m>_impr_pct`` = (base-x)/base*100 for lower-is-better, (x-base)/base*100 for higher-is-better
    (positive = better than baseline); NaN if base is 0/NaN. Raw mean ``<m>`` is included.
    Metrics default to those of LOWER/HIGHER_IS_BETTER present in the rows. Baseline rows are included (0 %)."""
    known = LOWER_IS_BETTER + HIGHER_IS_BETTER
    if metrics is None:
        metrics = [m for m in known if agg_rows and all(m in r for r in agg_rows)]
    out = []
    for r in agg_rows:
        base = next((b for b in agg_rows if b[key] == r[key] and b["controller"] == baseline), None)
        row = {key: r[key], "controller": r["controller"], "baseline": baseline}
        for m in metrics:
            row[m] = r[m]
            lower = m not in HIGHER_IS_BETTER
            row[m + "_impr_pct"] = _pct(base[m], r[m], lower) if base is not None else math.nan
        out.append(row)
    return out


def plot_learning_curves(progress_csvs, out_png, title=None):
    """Learning curves from MPLight ``progress.csv`` files: x = round, solid = train_travel_time, dashed =
    test_travel_time (greedy on VAL, rounds where present); one colour per label. Returns ``out_png``.
    Raises ImportError if matplotlib is missing."""
    try:
        from matplotlib.figure import Figure   # object API: no pyplot, does not change the caller's backend
    except ImportError as e:
        raise ImportError("plot_learning_curves needs matplotlib (pip install matplotlib)") from e
    fig = Figure(figsize=(8, 4.5))
    ax = fig.subplots()
    for i, (label, path) in enumerate(progress_csvs.items()):
        with open(path, encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        c = f"C{i % 10}"
        tr = [(int(r["round"]), float(r["train_travel_time"])) for r in rows if (r.get("train_travel_time") or "").strip()]
        te = [(int(r["round"]), float(r["test_travel_time"])) for r in rows if (r.get("test_travel_time") or "").strip()]
        if tr:
            ax.plot(*zip(*tr), color=c, label=f"{label} train")
        if te:
            ax.plot(*zip(*te), color=c, ls="--", marker="o", ms=3, label=f"{label} val")
    ax.set_xlabel("round")
    ax.set_ylabel("travel time (s)")
    if title:
        ax.set_title(title)
    ax.grid(alpha=.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    return out_png
