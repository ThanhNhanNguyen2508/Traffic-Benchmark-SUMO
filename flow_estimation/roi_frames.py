"""Per-frame vehicle counts inside the directional ROIs, with audit overlays (J02E).

MEASUREMENT: instantaneous count per frame = vehicles visible in each ROI in ONE image at time t.
It is NOT a flow and NOT veh/h (no matching between images, no beta). Use it to review ROI geometry and
detector behaviour; the flow estimate comes from flow_estimation.directional.

Inputs
    --index    JSONL, one line per frame: {"session","camera","t","image"[, "width","height"]}; "image" is relative to --root
               (and is the key of the detection cache). Extra keys are ignored.
    --dets     detection cache (jsonl / jsonl.gz) {"image": key, "dets": [[cls, x_center, y_bottom, h, w, conf], ...]}
               in 1920-wide space. A frame missing from the cache is an ERROR (never treated as empty).
    --roi      directional_roi.json (flow_estimation/roi.py). A camera without ROI is counted as FULL_FRAME only.
    --calibration CAL.json  thresholds conf.motorcycle / conf.other  (label "CALIBRATED:<sha256[:12]>")
    --review-conf X         both thresholds = X, label "REVIEW_THRESHOLD_NOT_CALIBRATED"  (exactly one of the two is required)
    --dets-provenance P     sidecar with "backend" (default: <dets>.provenance.json if it exists); else --backend must be given.
    --backend yolo11x|hybrid_cleaned   used only when no provenance; if both are present they must agree.

Usage
    python -m flow_estimation.roi_frames --index frames.jsonl --dets dets.jsonl.gz --roi flow_estimation/directional_roi.json \\
        --review-conf 0.25 --root IMG_ROOT --overlays 2 --out OUT_DIR
    python -m flow_estimation.roi_frames --index idx.jsonl --dets d.jsonl.gz --roi roi.json --calibration cal.json \\
        --backend yolo11x --root IMG_ROOT --overlay-images "S/Cam_X/frame_1.jpg" --overlay-scale 2 --out OUT_DIR

Outputs (OUT_DIR): detections.csv, counts_per_frame.csv/.json, summary_camera_roi.csv (sum_* vs mean_* per frame),
roi_frames_provenance.json, overlays/<session>/<camera>/<stem>.png (needs --root).
source_model: hybrid_cleaned -> motorcycle "yolo12x_bmd45", car/truck/bus "yolo11x"; yolo11x -> "yolo11x" for all.
Thresholds: motorcycle (and bicycle) use conf["motorcycle"]; car/truck/bus use conf["other"]. bicycle rows are written but never counted.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from .common import CLASSES, open_text
from .roi import load_roi

SCHEMA = "roi-frames-1"
MEASUREMENT = "instantaneous_count_per_frame - vehicles visible in each ROI in one image at time t; NOT a flow and NOT veh/h"
TWO_WHEEL = ("motorcycle", "bicycle")
ALL_CLASSES = ("motorcycle", "car", "truck", "bus", "bicycle")
DET_W = 1920.0
BACKENDS = ("yolo11x", "hybrid_cleaned")
COLORS = {"FORWARD": (40, 190, 60), "REVERSE": (50, 110, 255), "EXCLUDE": (150, 150, 150), "FULL_FRAME": (255, 150, 0)}


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def source_model(backend: str, cls: str) -> str:
    if backend == "hybrid_cleaned":
        return "yolo12x_bmd45" if cls == "motorcycle" else "yolo11x"
    return "yolo11x"


def read_index(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            r = json.loads(line)
            for k in ("session", "camera", "image"):
                if k not in r:
                    raise ValueError(f"{path}:{i}: index line lacks {k!r}")
            r.setdefault("t", "")
            rows.append(r)
    return rows


def load_cache(path):
    out = {}
    with open_text(path) as fh:
        for line in fh:
            r = json.loads(line)
            out[r["image"]] = r["dets"]
    return out


def _font(size):
    from PIL import ImageFont
    for p in ("arial.ttf", "DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _image_size(r, root):
    if r.get("width") and r.get("height"):
        return int(r["width"]), int(r["height"])
    if root is not None:
        p = Path(root) / r["image"]
        if p.is_file():
            from PIL import Image
            with Image.open(p) as im:
                return im.size
    return None


def draw_overlay(r, size, root, out_path, roi, dets_rows, counts, thr_label, scale, backend):
    from PIL import Image, ImageDraw
    im = Image.open(Path(root) / r["image"]).convert("RGB")
    W, H = im.size
    im = im.resize((int(W * scale), int(H * scale)), Image.LANCZOS)
    W2, H2 = im.size
    fs = int(6 + 3 * scale)
    font = _font(fs)
    ov = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    if roi is not None:
        for lb, poly in roi.polygons.items():
            pts = [(x * W2, y * H2) for x, y in poly]
            col = COLORS.get(lb, (255, 255, 0))
            d.polygon(pts, fill=col + (35,))
            d.line(pts + [pts[0]], fill=col + (255,), width=2)
            d.text((pts[0][0] + 3, pts[0][1] + 3), lb, fill=col + (255,), font=font)
    for x in dets_rows:
        if not x["passes_threshold"] or x["class"] not in CLASSES or x["foot_x"] == "":
            continue
        col = COLORS.get(x["roi"], (255, 255, 0))
        sx = W2 / W
        x1, y1, x2, y2 = (float(x[k]) * sx for k in ("x1", "y1", "x2", "y2"))
        d.rectangle([x1, y1, x2, y2], outline=col + (255,), width=2)
        short = {"yolo11x": "y11", "yolo12x_bmd45": "bmd45"}.get(x["source_model"], x["source_model"])
        d.text((x1 + 1, max(fs + 4, y1 - fs - 1)), f"{x['class']} {float(x['conf']):.2f} [{short}]", fill=col + (255,), font=font)
        fx, fy = float(x["foot_x"]) * sx, float(x["foot_y"]) * sx
        d.ellipse([fx - 3, fy - 3, fx + 3, fy + 3], fill=col + (255,), outline=(0, 0, 0, 255))
    # count table (per ROI x class), bottom-left
    labels = [lb for lb in counts]
    line_h = fs + 3
    tw = int(fs * 20)
    th = line_h * (len(labels) + 1) + 6
    d.rectangle([4, H2 - th - 4, 4 + tw, H2 - 4], fill=(0, 0, 0, 190))
    d.text((8, H2 - th - 1), f"{'ROI':<10} m  c  t  b  tot", fill=(255, 255, 255, 255), font=font)
    for i, lb in enumerate(labels):
        c = counts[lb]
        d.text((8, H2 - th - 1 + line_h * (i + 1)),
               f"{lb:<10} {c['motorcycle']:>2} {c['car']:>2} {c['truck']:>2} {c['bus']:>2} {c['total']:>3}",
               fill=COLORS.get(lb, (255, 255, 0)) + (255,), font=font)
    d.rectangle([0, 0, W2, fs + 5], fill=(0, 0, 0, 220))
    d.text((4, 2), f"{r['camera']} | {r['t']} | {thr_label} | {backend} | INSTANTANEOUS COUNT (not flow)",
           fill=(255, 255, 255, 255), font=font)
    out = Image.alpha_composite(im.convert("RGBA"), ov).convert("RGB")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(out_path)


def _write_csv(path, cols, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _evenly(n, k):
    if n == 0 or k <= 0:
        return []
    return sorted({int(round(v)) for v in np.linspace(0, n - 1, min(k, n))})


def run(index, dets, roi, out, calibration=None, review_conf=None, dets_provenance=None, backend=None, root=None,
        overlays=0, overlay_images=(), overlay_scale=2.0, exclude=()):
    if (calibration is None) == (review_conf is None):
        raise ValueError("exactly one of calibration / review_conf is required")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    # threshold
    if calibration is not None:
        cal = json.loads(Path(calibration).read_text(encoding="utf-8"))
        thr = {"motorcycle": float(cal["conf"]["motorcycle"]), "other": float(cal["conf"]["other"])}
        thr_label = "CALIBRATED:" + sha256_file(calibration)[:12]
    else:
        thr = {"motorcycle": float(review_conf), "other": float(review_conf)}
        thr_label = "REVIEW_THRESHOLD_NOT_CALIBRATED"
    for v in thr.values():
        if not (0.0 < v < 1.0):
            raise ValueError(f"threshold {v} not in (0,1)")
    # backend
    prov_path = Path(dets_provenance) if dets_provenance else (Path(str(dets) + ".provenance.json") if Path(str(dets) + ".provenance.json").is_file() else None)
    prov_backend = None
    if prov_path is not None:
        prov_backend = json.loads(prov_path.read_text(encoding="utf-8")).get("backend")
        if prov_backend is None:
            raise ValueError(f"{prov_path}: no 'backend' key")
    if prov_backend and backend and prov_backend != backend:
        raise ValueError(f"--backend {backend} contradicts dets provenance backend {prov_backend}")
    backend_used = prov_backend or backend
    if backend_used is None:
        raise ValueError("backend unknown: give --dets-provenance (or <dets>.provenance.json) or --backend")
    if backend_used not in BACKENDS:
        raise ValueError(f"backend {backend_used!r} not in {BACKENDS}")

    rois = load_roi(roi)
    cache = load_cache(dets)
    rows_idx = [r for r in read_index(index) if r["camera"] not in set(exclude)]
    missing = [r["image"] for r in rows_idx if r["image"] not in cache]
    if missing:
        raise RuntimeError(f"{len(missing)} index frame(s) missing from detection cache (not treated as empty); first: {missing[:3]}")
    rows_idx.sort(key=lambda r: (r["session"], r["camera"], r["t"], r["image"]))

    det_rows, count_rows, errors = [], [], []
    per_frame = []  # (idx row, det_rows, counts)
    n_det = 0
    for r in rows_idx:
        cam = r["camera"]
        cr = rois.get(cam)
        size = _image_size(r, root)
        fr_det = []
        counts = {lb: {c: 0 for c in CLASSES} for lb in (["FULL_FRAME"] + list(cr.labels) if cr else ["FULL_FRAME"])}
        for d in cache[r["image"]]:
            cls, xc, yb, h, w, conf = d[0], float(d[1]), float(d[2]), float(d[3]), float(d[4]), float(d[5])
            if cls not in ALL_CLASSES:
                continue
            n_det += 1
            passes = conf >= (thr["motorcycle"] if cls in TWO_WHEEL else thr["other"])
            lab = cr.label_det(xc, yb) if cr else "FULL_FRAME"
            row = {"session": r["session"], "camera": cam, "image": r["image"], "t": r["t"], "backend": backend_used,
                   "source_model": source_model(backend_used, cls), "class": cls, "conf": conf, "passes_threshold": int(passes),
                   "x_det": xc, "y_det": yb, "w_det": w, "h_det": h, "x1": "", "y1": "", "x2": "", "y2": "",
                   "foot_x": "", "foot_y": "", "roi": lab}
            if size:
                s = DET_W / size[0]
                fx, fy, bw, bh = xc / s, yb / s, w / s, h / s
                row.update(x1=round(fx - bw / 2, 2), y1=round(fy - bh, 2), x2=round(fx + bw / 2, 2), y2=round(fy, 2),
                           foot_x=round(fx, 2), foot_y=round(fy, 2))
            fr_det.append(row)
            if passes and cls in CLASSES:
                counts["FULL_FRAME"][cls] += 1
                if cr:
                    counts[lab][cls] += 1
        for lb, c in counts.items():
            c["total"] = sum(c[k] for k in CLASSES)
        if cr:
            for k in CLASSES + ["total"]:
                if sum(counts[lb][k] for lb in cr.labels) != counts["FULL_FRAME"][k]:
                    errors.append(f"conservation failed {r['image']} {k}")
        for lb, c in counts.items():
            count_rows.append({"session": r["session"], "camera": cam, "image": r["image"], "t": r["t"], "threshold_label": thr_label,
                               "roi_mode": "DIRECTIONAL_ROI" if cr else "FULL_FRAME_ONLY", "roi": lb, **c})
        det_rows.extend(fr_det)
        per_frame.append((r, fr_det, counts, cr, size))
    if errors:
        raise RuntimeError("; ".join(errors[:5]))

    dcols = ["session", "camera", "image", "t", "backend", "source_model", "class", "conf", "passes_threshold", "x_det", "y_det", "w_det",
             "h_det", "x1", "y1", "x2", "y2", "foot_x", "foot_y", "roi"]
    _write_csv(out / "detections.csv", dcols, det_rows)
    ccols = ["session", "camera", "image", "t", "threshold_label", "roi_mode", "roi"] + CLASSES + ["total"]
    _write_csv(out / "counts_per_frame.csv", ccols, count_rows)
    (out / "counts_per_frame.json").write_text(json.dumps({"measurement": MEASUREMENT, "threshold_label": thr_label, "rows": count_rows},
                                                          indent=1, ensure_ascii=False), encoding="utf-8")
    # summary
    agg = {}
    for c in count_rows:
        a = agg.setdefault((c["session"], c["camera"], c["roi"]), {"n": 0, **{k: 0 for k in CLASSES + ["total"]}})
        a["n"] += 1
        for k in CLASSES + ["total"]:
            a[k] += c[k]
    srows = []
    for (s, cam, lb), a in sorted(agg.items()):
        row = {"session": s, "camera": cam, "roi": lb, "n_frames": a["n"]}
        for k in CLASSES + ["total"]:
            row[f"sum_{k}"] = a[k]
        for k in CLASSES + ["total"]:
            row[f"mean_{k}"] = round(a[k] / a["n"], 4)
        srows.append(row)
    scols = ["session", "camera", "roi", "n_frames"] + [f"sum_{k}" for k in CLASSES + ["total"]] + [f"mean_{k}" for k in CLASSES + ["total"]]
    _write_csv(out / "summary_camera_roi.csv", scols, srows)

    # overlays
    n_ov = 0
    overlay_note = None
    chosen = set()
    if root is None:
        if overlays or overlay_images:
            overlay_note = "overlays skipped: --root not given"
            print("note: " + overlay_note)
    else:
        bycam = {}
        for i, pf in enumerate(per_frame):
            bycam.setdefault(pf[0]["camera"], []).append(i)
        for idxs in bycam.values():
            chosen.update(idxs[j] for j in _evenly(len(idxs), overlays))
        want = set(overlay_images)
        for i, pf in enumerate(per_frame):
            if pf[0]["image"] in want:
                chosen.add(i)
        unk = want - {pf[0]["image"] for pf in per_frame}
        if unk:
            raise RuntimeError(f"--overlay-images not in index: {sorted(unk)[:3]}")
        for i in sorted(chosen):
            r, fr_det, counts, cr, size = per_frame[i]
            draw_overlay(r, size, root, out / "overlays" / r["session"] / r["camera"] / (Path(r["image"]).stem + ".png"), cr, fr_det,
                         counts, thr_label, overlay_scale, backend_used)
            n_ov += 1

    import PIL
    prov = {"schema_version": SCHEMA, "measurement": MEASUREMENT,
            "inputs": {k: ({"path": str(p), "sha256": sha256_file(p)} if p is not None else None)
                       for k, p in (("index", index), ("dets", dets), ("dets_provenance", prov_path), ("roi", roi), ("calibration", calibration))},
            "backend": backend_used, "backend_source": "dets_provenance" if prov_backend else "--backend",
            "threshold": {"label": thr_label, "motorcycle": thr["motorcycle"], "other": thr["other"]},
            "n_frames": len(per_frame), "n_detections": n_det, "n_overlays": n_ov, "overlay_note": overlay_note,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "PIL": PIL.__version__}}
    (out / "roi_frames_provenance.json").write_text(json.dumps(prov, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"roi_frames: {len(per_frame)} frames, {n_det} detections, {n_ov} overlays, threshold {thr_label}, backend {backend_used} -> {out}")
    return prov


def main(argv=None):
    ap = argparse.ArgumentParser(description="Per-frame ROI vehicle counts + overlays (instantaneous count, NOT flow).")
    ap.add_argument("--index", required=True)
    ap.add_argument("--dets", required=True)
    ap.add_argument("--roi", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--calibration")
    g.add_argument("--review-conf", type=float)
    ap.add_argument("--dets-provenance")
    ap.add_argument("--backend", choices=BACKENDS)
    ap.add_argument("--root")
    ap.add_argument("--overlays", type=int, default=0, help="N evenly spaced overlay frames per camera")
    ap.add_argument("--overlay-images", nargs="*", default=[], help="index image keys to render in addition")
    ap.add_argument("--overlay-scale", type=float, default=2.0)
    ap.add_argument("--exclude", nargs="*", default=[], help="camera ids to skip")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    try:
        run(a.index, a.dets, a.roi, a.out, calibration=a.calibration, review_conf=a.review_conf, dets_provenance=a.dets_provenance,
            backend=a.backend, root=a.root, overlays=a.overlays, overlay_images=a.overlay_images, overlay_scale=a.overlay_scale,
            exclude=a.exclude)
    except (ValueError, RuntimeError, OSError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
