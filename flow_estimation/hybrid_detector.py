#!/usr/bin/env python3
"""HYBRID detector: YOLO11x (car / truck / bus) + fine-tuned YOLO12-X BMD45 (Two-wheeler -> motorcycle) + motorcycle cleanup, written as a VietTSC detection cache.

Additive module: detect.py is untouched and keeps working as before. Same cache contract as detect.py: one JSON line per image,
{"image": key, "dets": [[cls, x_center, y_bottom, height, width, conf], ...]} in 1920x1080 space (x 1920/width), conf rounded to 3 decimals,
cls in {motorcycle, car, truck, bus}. Bicycles are intentionally not produced (see hybrid_cleanup docs: downstream code simply never sees that class).

    python -m flow_estimation.hybrid_detector --list frames.jsonl --root <image root> --out dets.jsonl.gz \
        --weights-yolo11x <real yolo11x.pt> --weights-bmd45 <real yolov12x_bmd45.pt> [--audit-dir DIR] [--device 0]
    python -m flow_estimation.hybrid_detector --verify-only --weights-yolo11x ... --weights-bmd45 ...

Both checkpoints are pinned by SHA-256 and size; Git LFS pointers and mismatching files are rejected (no automatic download, no fallback).
Parameters equal detect.py (conf floor 0.08, NMS iou 0.6, imgsz 1280, max_det 400, class-aware NMS); precision is explicit (fp32 on CPU, fp16 on GPU as in detect.py).
Cleanup (hybrid_motorcycle_cleanup) is decided on the full conf >= 0.08 set, then the cache keeps all remaining boxes (thresholds are applied later by calibrate/infer).
A sidecar <out>.provenance.json records weights, versions, config hash, counts and the cache SHA-256. The legacy `detector.model` label that calibrate.py
writes into its JSON ("yolo11x (COCO, Ultralytics)") says nothing about this backend: use the provenance sidecar instead.
"""
from __future__ import annotations

import argparse
import collections
import csv
import gzip
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

from . import hybrid_motorcycle_cleanup as hc

REPO = Path(__file__).resolve().parents[1]
DEFAULT_YOLO11X = REPO / "models" / "yolo11x.pt"
DEFAULT_BMD45 = REPO / "models" / "yolov12x_bmd45.pt"
BACKEND = "hybrid_cleaned"
# ---- pinned checkpoints (verified in J01 S0-A / S0-C)
YOLO11X = {"sha256": "7bc158aa95c0ebfdd87f70f01653c1131b93e92522dbe15c228bcd742e773a24", "size": 114636239, "source": "Ultralytics YOLO11x COCO release (Git LFS object of models/yolo11x.pt)"}
BMD45 = {"sha256": "4c3bc036337ea391a0bb554b94c054aa7c473fda66f6800223dd83da6769982e", "size": 356251477, "hf_repo": "iisc-aim/BMD-45",
         "hf_file": "weights/YOLOv12-X/best.pt", "hf_revision": "ec06f3dfbc5c896100265d38917cd49a996f41ec", "license_status": "LICENSE_REVIEW_PENDING"}
YOLO11X_CLASS_MAP = {2: "car", 5: "bus", 7: "truck"}          # COCO ids; names are re-checked against model.names at runtime
BMD45_MOTORCYCLE_NAME = "Two-wheeler"                          # id resolved from model.names at runtime (S0-A: 7)
# ---- detection parameters (= detect.py:45-46)
DETECTION_CONF_FLOOR = 0.08
NMS_IOU = 0.6
IMAGE_SIZE = 1280
MAX_DET = 400
OUT_W = 1920
CANON = ["motorcycle", "car", "truck", "bus"]
LEGACY_LABEL_WARNING = ("calibrate.py writes the hard-coded label 'yolo11x (COCO, Ultralytics)' into the calibration JSON; for backend hybrid_cleaned that label is "
                        "NOT the detector provenance. Use the *.provenance.json sidecars.")


# ------------------------------------------------------------------ weights verification
def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def is_lfs_pointer(path) -> bool:
    p = Path(path)
    return p.is_file() and p.stat().st_size < 4096 and p.read_bytes()[:64].startswith(b"version https://git-lfs")


def verify_weights(path, spec: dict, label: str) -> dict:
    p = Path(path)
    if not p.is_file():
        raise RuntimeError(f"{label}: file not found: {p}")
    if is_lfs_pointer(p):
        raise RuntimeError(f"{label}: {p} is a Git LFS pointer, not the real checkpoint")
    size = p.stat().st_size
    if size != spec["size"]:
        raise RuntimeError(f"{label}: size {size} != expected {spec['size']} ({p})")
    sha = sha256_file(p)
    if sha != spec["sha256"]:
        raise RuntimeError(f"{label}: SHA-256 {sha} != expected {spec['sha256']} ({p})")
    return {"path": str(p), "size": size, "sha256": sha}


# ------------------------------------------------------------------ pure fusion / cache logic (no model, no ultralytics)
def resolve_moto_id(bmd_names: dict) -> int:
    ids = [int(i) for i, n in bmd_names.items() if str(n) == BMD45_MOTORCYCLE_NAME]
    if len(ids) != 1:
        raise RuntimeError(f"BMD45 model.names has {len(ids)} classes named {BMD45_MOTORCYCLE_NAME!r}: {bmd_names}")
    return ids[0]


def check_yolo_names(yolo_names: dict):
    for i, n in YOLO11X_CLASS_MAP.items():
        if str(yolo_names.get(i)) != n:
            raise RuntimeError(f"YOLO11x model.names[{i}] = {yolo_names.get(i)!r}, expected {n!r}")


def make_row(image, canon, conf, xyxy, source_model, source_class, source_class_id, raw_id):
    """decision values follow S0-G: coordinates rounded to 0.1 px, confidence to 4 decimals; the original float32 values are kept for the cache"""
    x0, y0, x1, y1 = (float(v) for v in xyxy)
    return dict(image=image, canon=canon, conf=round(float(conf), 4), x0=round(x0, 1), y0=round(y0, 1), x1=round(x1, 1), y1=round(y1, 1),
                source_model=source_model, source_class=source_class, source_class_id=int(source_class_id), raw_id=raw_id,
                xyxy32=tuple(np.float32(v) for v in xyxy), conf32=np.float32(conf))


def fuse(yolo_raw: dict, bmd_raw: dict, yolo_names: dict, bmd_names: dict) -> list:
    """yolo_raw / bmd_raw: {image: [(class_id, conf, (x0, y0, x1, y1)), ...]} straight from the models (native pixels).
    Late fusion: YOLO11x car/bus/truck + BMD45 Two-wheeler -> motorcycle. No NMS between models. Deterministic order and row numbers."""
    check_yolo_names(yolo_names); moto_id = resolve_moto_id(bmd_names)
    rows = []
    for image, L in yolo_raw.items():
        for k, (cid, conf, xyxy) in enumerate(L):
            if cid in YOLO11X_CLASS_MAP:
                rows.append(make_row(image, YOLO11X_CLASS_MAP[cid], conf, xyxy, "yolo11x", yolo_names[cid], cid, f"{image}|yolo11x|{k}"))
    for image, L in bmd_raw.items():
        for k, (cid, conf, xyxy) in enumerate(L):
            if cid == moto_id:
                rows.append(make_row(image, "motorcycle", conf, xyxy, hc.BMD45_MODEL, bmd_names[cid], cid, f"{image}|{hc.BMD45_MODEL}|{k}"))
    rows.sort(key=lambda r: (r["image"], CANON.index(r["canon"]), -r["conf"], r["x0"], r["y0"], r["x1"], r["y1"], r["source_model"], r["raw_id"]))
    for n, r in enumerate(rows):
        r["row"] = n
    return rows


def cache_dets(rows_of_image: list, img_w: int) -> list:
    """one image -> [[cls, x_center, y_bottom, height, width, conf], ...] in 1920-wide space, rounding exactly as detect.py:48-53 (float32 arithmetic)."""
    s = OUT_W / img_w
    out = []
    for r in sorted(rows_of_image, key=lambda r: (-r["conf"], CANON.index(r["canon"]), r["x0"], r["y0"], r["x1"], r["y1"], r["raw_id"])):
        x0, y0, x1, y1 = np.array(r["xyxy32"], dtype=np.float32) * s
        out.append([r["canon"], round(float((x0 + x1) / 2), 1), round(float(y1), 1), round(float(y1 - y0), 1), round(float(x1 - x0), 1), round(float(r["conf32"]), 3)])
    return out


def write_cache(path: Path, records: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8") as fo:
        for rec in records:
            fo.write(json.dumps(rec) + "\n")


def write_audit(audit_dir: Path, rows, dec, kept, removed, sizes):
    audit_dir.mkdir(parents=True, exist_ok=True)
    base = ["image", "class", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "confidence", "source_model", "source_class", "source_class_id", "raw_id"]
    mcols = ["w", "h", "area", "aspect_w_over_h", "area_ratio_image", "size_cap_w", "size_cap_h", "n_inside_all", "n_inside_distinct", "median_inside_area", "ratio_to_inside_median",
             "n_centers_inside", "n_yolo_vehicles_inside", "yolo_vehicle_inside_class", "max_iou_yolo_vehicle", "max_ioa_covered_by_yolo_vehicle", "overlapped_vehicle_class", "nested_same_size"]

    def brow(r): return {"image": r["image"], "class": r["canon"], "bbox_x1": r["x0"], "bbox_y1": r["y0"], "bbox_x2": r["x1"], "bbox_y2": r["y1"], "confidence": r["conf"],
                         "source_model": r["source_model"], "source_class": r["source_class"], "source_class_id": r["source_class_id"], "raw_id": r["raw_id"]}

    def w(name, cols, rws):
        with open(audit_dir / name, "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=cols); wr.writeheader(); wr.writerows(rws)
    w("hybrid_pre_cleanup_boxes.csv", base + ["image_w", "image_h", "cleanup_status", "cleanup_rule", "cleanup_flags"],
      [{**brow(r), "image_w": sizes[r["image"]][0], "image_h": sizes[r["image"]][1], "cleanup_status": dec[r["raw_id"]]["status"] if r["raw_id"] in dec else "UNTOUCHED_YOLO11x",
        "cleanup_rule": (dec[r["raw_id"]]["rule"] or "") if r["raw_id"] in dec else "", "cleanup_flags": "|".join(dec[r["raw_id"]]["flags"]) if r["raw_id"] in dec else ""} for r in rows])
    w("removed_motorcycle_boxes.csv", base + mcols + ["rule", "reason"], [{**brow(r), **{k: dec[r["raw_id"]]["m"][k] for k in mcols}, "rule": dec[r["raw_id"]]["rule"],
                                                                       "reason": hc.RULE_REASON[dec[r["raw_id"]]["rule"]]} for r in removed])
    fk = [r for r in kept if r["raw_id"] in dec and dec[r["raw_id"]]["status"] == "KEEP_FLAG"]
    w("flagged_but_kept.csv", base + mcols + ["flags"], [{**brow(r), **{k: dec[r["raw_id"]]["m"][k] for k in mcols}, "flags": "|".join(dec[r["raw_id"]]["flags"])} for r in fk])
    return {"pre_cleanup_boxes": len(rows), "removed": len(removed), "flagged_but_kept": len(fk)}


def build_provenance(*, out, list_file, n_images, rows, kept, removed, dec, weights, names, versions, device, precision, cfg_path, extra=None):
    cnt = lambda L: dict(collections.Counter(r["canon"] for r in L))
    return {"backend": BACKEND, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "module": "flow_estimation/hybrid_detector.py", "module_sha256": sha256_file(__file__),
            "cleanup_module_sha256": sha256_file(hc.__file__), "cleanup_config": str(cfg_path), "cleanup_config_sha256": hc.config_sha256(cfg_path),
            "cleanup_config_status": hc.load_config(cfg_path).get("status"), "cleanup_reference": "J01 S0-G (Agent workspace scripts/hybrid_motorcycle_cleanup.py)",
            "size_cap_px_512x288": list(hc.SIZE_CAP_PX_512),
            "policy": {"yolo11x": "COCO ids 2/5/7 -> car/bus/truck, source-original class-aware NMS", "bmd45": f"class named {BMD45_MOTORCYCLE_NAME!r} -> motorcycle, checkpoint class-aware NMS",
                       "excluded": "all other YOLO11x and BMD45 classes (incl. bicycle, three-wheeler)", "fusion": "late fusion; no cross-model NMS/WBF",
                       "cleanup": "only BMD45 motorcycle boxes can be removed (rules R1/R2); everything else kept unchanged"},
            "parameters": {"conf_floor": DETECTION_CONF_FLOOR, "nms_iou": NMS_IOU, "imgsz": IMAGE_SIZE, "max_det": MAX_DET, "agnostic_nms": False, "device": device, "precision": precision},
            "weights": weights, "class_names": names, "versions": versions, "input": {"list": str(list_file), "list_sha256": sha256_file(list_file), "n_images": n_images},
            "counts": {"pre_cleanup_by_class": cnt(rows), "final_by_class": cnt(kept), "removed_motorcycle_boxes": len(removed), "final_total": len(kept),
                       "removed_by_rule": dict(collections.Counter(dec[r["raw_id"]]["rule"] for r in removed))},
            "cache": {"path": str(out), "sha256": sha256_file(out), "coordinate_space": "1920-wide (x 1920/image width), [cls, x_center, y_bottom, height, width, conf]", "conf_decimals": 3},
            "legacy_calibration_label_warning": LEGACY_LABEL_WARNING, "license_status": "LICENSE_REVIEW_PENDING", **(extra or {})}


# ------------------------------------------------------------------ CLI (needs ultralytics)
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", type=Path, help="jsonl with an 'image' key per line (as detect.py)")
    ap.add_argument("--root", type=Path, help="folder the 'image' paths are relative to")
    ap.add_argument("--out", type=Path, help="output .jsonl or .jsonl.gz (VietTSC detection cache)")
    ap.add_argument("--weights-yolo11x", type=Path, default=DEFAULT_YOLO11X, help="REAL yolo11x.pt (models/yolo11x.pt is a Git LFS pointer in the repo)")
    ap.add_argument("--weights-bmd45", type=Path, default=DEFAULT_BMD45, help="REAL yolov12x_bmd45.pt")
    ap.add_argument("--imgsz", type=int, default=IMAGE_SIZE)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None, help="e.g. 0, cuda:1, cpu (default: ultralytics auto)")
    ap.add_argument("--precision", default="auto", choices=["auto", "fp32", "fp16"], help="auto: fp16 on CUDA (as detect.py half=True), fp32 on CPU")
    ap.add_argument("--audit-dir", type=Path, default=None, help="write hybrid_pre_cleanup_boxes.csv, removed_motorcycle_boxes.csv, flagged_but_kept.csv")
    ap.add_argument("--cleanup-config", type=Path, default=hc.CONFIG_PATH)
    ap.add_argument("--verify-only", action="store_true", help="only verify the two checkpoints (existence, not LFS pointer, size, SHA-256) and exit")
    a = ap.parse_args()

    weights = {"yolo11x": {**verify_weights(a.weights_yolo11x, YOLO11X, "YOLO11x"), **{k: v for k, v in YOLO11X.items() if k not in ("sha256", "size")}},
               "bmd45": {**verify_weights(a.weights_bmd45, BMD45, "BMD45"), **{k: v for k, v in BMD45.items() if k not in ("sha256", "size")}}}
    if a.verify_only:
        print(json.dumps({"backend": BACKEND, "weights_verified": weights}, indent=1)); return
    if not (a.list and a.root and a.out):
        ap.error("--list, --root and --out are required (unless --verify-only)")
    import torch
    from ultralytics import YOLO
    import ultralytics

    cfg = hc.load_config(a.cleanup_config)
    keys = list(dict.fromkeys(json.loads(l)["image"] for l in open(a.list, encoding="utf-8")))
    cuda = torch.cuda.is_available() and str(a.device).lower() != "cpu"
    half = a.precision == "fp16" or (a.precision == "auto" and cuda)
    if half and not cuda:
        raise RuntimeError("--precision fp16 needs a CUDA device")
    device_name = (torch.cuda.get_device_name(0) if cuda else "cpu"); precision = "fp16" if half else "fp32"
    print(f"hybrid_detector: {len(keys)} images | device {device_name} | precision {precision} | cleanup config {hc.config_sha256(a.cleanup_config)[:12]}", flush=True)
    my, mb = YOLO(str(a.weights_yolo11x)), YOLO(str(a.weights_bmd45))
    yn, bn = {int(k): str(v) for k, v in my.names.items()}, {int(k): str(v) for k, v in mb.names.items()}
    check_yolo_names(yn); moto_id = resolve_moto_id(bn)
    kw = dict(imgsz=a.imgsz, conf=DETECTION_CONF_FLOOR, iou=NMS_IOU, verbose=False, half=half, max_det=MAX_DET, device=a.device)
    yolo_raw, bmd_raw, sizes = {}, {}, {}
    for i in range(0, len(keys), a.batch):
        chunk = keys[i:i + a.batch]; paths = [str(a.root / k) for k in chunk]
        ry = my.predict(paths, classes=list(YOLO11X_CLASS_MAP), **kw)
        rb = mb.predict(paths, classes=[moto_id], **kw)
        for k, y, b in zip(chunk, ry, rb):
            if tuple(y.orig_shape) != tuple(b.orig_shape):
                raise RuntimeError(f"{k}: orig_shape differs between models {y.orig_shape} vs {b.orig_shape}")
            sizes[k] = (int(y.orig_shape[1]), int(y.orig_shape[0]))
            yolo_raw[k] = [(int(c), float(p), tuple(xy)) for xy, c, p in zip(y.boxes.xyxy.cpu().numpy(), y.boxes.cls.cpu().numpy(), y.boxes.conf.cpu().numpy())]
            bmd_raw[k] = [(int(c), float(p), tuple(xy)) for xy, c, p in zip(b.boxes.xyxy.cpu().numpy(), b.boxes.cls.cpu().numpy(), b.boxes.conf.cpu().numpy())]
        print(f"{min(i + a.batch, len(keys))}/{len(keys)}", flush=True)

    rows = fuse(yolo_raw, bmd_raw, yn, bn)
    dec = hc.decide(rows, cfg, image_wh_of=lambda im: sizes[im])
    kept, removed = hc.apply(rows, dec)
    by = collections.defaultdict(list)
    for r in kept:
        by[r["image"]].append(r)
    write_cache(a.out, [{"image": k, "dets": cache_dets(by.get(k, []), sizes[k][0])} for k in keys])
    audit = write_audit(a.audit_dir, rows, dec, kept, removed, sizes) if a.audit_dir else None
    versions = {"python": platform.python_version(), "torch": torch.__version__, "ultralytics": ultralytics.__version__, "numpy": np.__version__, "platform": platform.platform()}
    prov = build_provenance(out=a.out, list_file=a.list, n_images=len(keys), rows=rows, kept=kept, removed=removed, dec=dec, weights=weights,
                            names={"yolo11x_used": {str(k): yn[k] for k in YOLO11X_CLASS_MAP}, "bmd45_all": {str(k): v for k, v in bn.items()}, "bmd45_motorcycle_class_id": moto_id},
                            versions=versions, device=device_name, precision=precision, cfg_path=a.cleanup_config, extra={"audit": audit})
    Path(str(a.out) + ".provenance.json").write_text(json.dumps(prov, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"done: {a.out} | final {len(kept)} boxes (removed motorcycle boxes {len(removed)}) | provenance {a.out}.provenance.json", flush=True)


if __name__ == "__main__":
    main()
