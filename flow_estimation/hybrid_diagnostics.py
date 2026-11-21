#!/usr/bin/env python3
"""Bounded diagnostic images of a hybrid detection cache (written by hybrid_detector.py). Diagnostics are NOT ground truth and NOT an accuracy measure.

Final images are drawn FROM THE CACHE (footpoint format, 1920-wide space -> image pixels once). Before/after and removed/flagged panels come from the audit CSVs
that the same hybrid_detector.py run wrote (--audit-dir). By default at most --max-images images are exported (evenly spaced, deterministic); there is no "all" switch.

    python -m flow_estimation.hybrid_diagnostics --cache dets.jsonl.gz --audit-dir audit/ --root <image root> --out diagnostics/ \
        [--max-images 12] [--confs 0.10 0.15 0.20 0.25 0.30 0.35] [--main-conf 0.25] [--labels all|nonmoto|none]
"""
from __future__ import annotations

import argparse
import collections
import csv
import gzip
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT_W = 1920
CANON = ["motorcycle", "car", "truck", "bus"]
COL = {"motorcycle": (255, 150, 0), "car": (60, 180, 75), "truck": (145, 30, 180), "bus": (0, 130, 200)}
RED, YELLOW = (255, 0, 0), (255, 215, 0)


def load_detections(path):
    """{image key: dets} from a (gzipped) detection jsonl; same format as common.load_detections (not imported here: common needs scipy, diagnostics must not)"""
    p = Path(path); out = {}
    with (gzip.open(p, "rt", encoding="utf-8") if p.suffix == ".gz" else open(p, encoding="utf-8")) as fh:
        for line in fh:
            r = json.loads(line); out[r["image"]] = r["dets"]
    return out


def _font(sz=11):
    for p in ("arial.ttf", "DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(p, sz)
        except OSError:
            pass
    return ImageFont.load_default()


FONT = _font()


def cache_boxes(dets, img_w):
    """[cls, x_center, y_bottom, height, width, conf] in 1920-wide space -> dicts with xyxy in IMAGE pixels (scaled once by img_w / 1920)"""
    k = img_w / OUT_W
    return [{"canon": c, "conf": p, "x0": (xc - w / 2) * k, "y0": (yb - h) * k, "x1": (xc + w / 2) * k, "y1": yb * k} for c, xc, yb, h, w, p in dets]


def draw(path, W, H, items, title, color, label, labels="all", thick_conf=0.25, hdr=18):
    im = Image.open(path).convert("RGB"); sx, sy = W / im.size[0], H / im.size[1]
    im = im.resize((W, H), Image.LANCZOS); d = ImageDraw.Draw(im)
    for r in sorted(items, key=lambda z: z["conf"]):
        col = color(r); x0, y0, x1, y1 = r["x0"] * sx, r["y0"] * sy, r["x1"] * sx, r["y1"] * sy
        d.rectangle([x0, y0, x1, y1], outline=col, width=2 if r["conf"] >= thick_conf else 1)
        if labels == "all" or (labels == "nonmoto" and r.get("canon") != "motorcycle"):
            d.text((x0 + 1, max(hdr, y0 - 12)), label(r), fill=col, font=FONT)
    d.rectangle([0, 0, W, hdr - 1], fill=(0, 0, 0)); d.text((4, 2), title, fill=(255, 255, 255), font=FONT)
    return im


def read_audit(audit_dir):
    def rd(name):
        p = Path(audit_dir) / name
        return list(csv.DictReader(open(p, encoding="utf-8"))) if p.is_file() else []
    conv = lambda r: {**r, "canon": r["class"], "conf": float(r["confidence"]), "x0": float(r["bbox_x1"]), "y0": float(r["bbox_y1"]), "x1": float(r["bbox_x2"]), "y1": float(r["bbox_y2"])}
    return [conv(r) for r in rd("hybrid_pre_cleanup_boxes.csv")], [conv(r) for r in rd("removed_motorcycle_boxes.csv")], [conv(r) for r in rd("flagged_but_kept.csv")]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, required=True); ap.add_argument("--audit-dir", type=Path, default=None)
    ap.add_argument("--root", type=Path, required=True, help="folder the cache image keys are relative to (images must exist; no image -> error)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-images", type=int, default=12); ap.add_argument("--confs", type=float, nargs="+", default=[0.10, 0.15, 0.20, 0.25, 0.30, 0.35])
    ap.add_argument("--main-conf", type=float, default=0.25); ap.add_argument("--labels", default="all", choices=["all", "nonmoto", "none"])
    a = ap.parse_args()
    if a.max_images < 1:
        ap.error("--max-images must be >= 1")
    D = load_detections(a.cache); keys = list(D)
    if not keys:
        raise RuntimeError(f"empty cache: {a.cache}")
    n = min(a.max_images, len(keys)); sel = [keys[round(i * (len(keys) - 1) / max(n - 1, 1))] for i in range(n)] if n > 1 else [keys[0]]
    sel = list(dict.fromkeys(sel))
    for k in sel:
        if not (a.root / k).is_file():
            raise FileNotFoundError(f"image missing, no diagnostic is drawn from a cache without its image: {a.root / k}")
    pre, removed, flagged = read_audit(a.audit_dir) if a.audit_dir else ([], [], [])
    (a.out / "images" / "comparison").mkdir(parents=True, exist_ok=True); (a.out / "images" / "removed_boxes").mkdir(parents=True, exist_ok=True)
    cancol = lambda r: COL[r["canon"]]; canlab = lambda r: f"{r['canon']} {r['conf']:.2f}"; rows = []; mism = []
    for n_, k in enumerate(sel):
        p = a.root / k; w = Image.open(p).size[0]; tag = f"{n_:03d}_" + k.replace("/", "__").replace("\\", "__").rsplit(".", 1)[0]
        final = cache_boxes(D[k], w)
        for t in a.confs:
            d = a.out / "images" / f"FINAL_conf{int(round(t * 100)):03d}"; d.mkdir(parents=True, exist_ok=True)
            K = [r for r in final if r["conf"] >= t - 1e-9]; c = collections.Counter(r["canon"] for r in K)
            draw(p, 1280, int(1280 * Image.open(p).size[1] / w), K, f"{k} | HYBRID (from cache) | conf>={t:.2f} | total {len(K)} (" + ", ".join(f"{x} {c[x]}" for x in CANON) + ")", cancol, canlab, a.labels).save(d / f"{tag}.png")
            rows.append({"image": k, "threshold": t, "total": len(K), **{x: c[x] for x in CANON}})
        if pre:
            P = [r for r in pre if r["image"] == k and r["conf"] >= a.main_conf - 1e-9]; Rm = [r for r in removed if r["image"] == k and r["conf"] >= a.main_conf - 1e-9]
            Fk = [r for r in flagged if r["image"] == k and r["conf"] >= a.main_conf - 1e-9]; fin = [r for r in final if r["conf"] >= a.main_conf - 1e-9]
            H = int(960 * Image.open(p).size[1] / w)
            A = draw(p, 960, H, P, f"before cleanup | conf>={a.main_conf:.2f} | {len(P)}", cancol, canlab, a.labels); B = draw(p, 960, H, fin, f"after cleanup (cache) | conf>={a.main_conf:.2f} | {len(fin)} | removed {len(Rm)}", cancol, canlab, a.labels)
            W_ = Image.new("RGB", (1920, H)); W_.paste(A, (0, 0)); W_.paste(B, (960, 0)); W_.save(a.out / "images/comparison" / f"{tag}.png")
            fk = {r["raw_id"] for r in Fk}; items = fin + Rm
            draw(p, 1280, int(1280 * Image.open(p).size[1] / w), items, f"{k} | REMOVED motorcycle boxes (red) {len(Rm)} | flagged but kept (yellow, motorcycle only) {len(Fk)}",
                 lambda r: RED if "rule" in r else cancol(r), lambda r: (f"REMOVED {r['rule'][:2]} {r['conf']:.2f}" if "rule" in r else canlab(r)), a.labels).save(a.out / "images/removed_boxes" / f"{tag}.png")
            # consistency: cache (floor) vs audit non-removed boxes, per class
            cc = collections.Counter(r["canon"] for r in final); ca = collections.Counter(r["canon"] for r in pre if r["image"] == k and r["cleanup_status"] != "REMOVE")
            if any(cc[x] != ca[x] for x in CANON):
                mism.append({"image": k, "cache": dict(cc), "audit_kept": dict(ca)})
    with open(a.out / "counts_by_image_class.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=["image", "threshold", "total"] + CANON); wr.writeheader(); wr.writerows(rows)
    summary = {"cache": str(a.cache), "n_cache_images": len(keys), "n_exported": len(sel), "max_images": a.max_images, "images": sel, "confs": a.confs, "main_conf": a.main_conf,
               "audit_dir": str(a.audit_dir) if a.audit_dir else None, "cache_vs_audit_mismatches": mism,
               "note": "diagnostic visualisation only; not ground truth, not an accuracy measure"}
    (a.out / "diagnostics_summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"diagnostics -> {a.out} | {len(sel)} images | cache/audit mismatches: {len(mism)}")


if __name__ == "__main__":
    main()
