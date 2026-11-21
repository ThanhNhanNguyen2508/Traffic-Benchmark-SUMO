"""Motorcycle-box cleanup for the hybrid detector (pure geometry, no model, no ultralytics import).

Port of the experiment S0-G (Agent workspace: scripts/hybrid_motorcycle_cleanup.py::{distinct, metrics_for, decide}), unchanged predicates and
thresholds; the thresholds live in hybrid_cleanup_config.json (a byte-identical copy of the S0-G config, hash recorded in provenance).

Only boxes of the fine-tuned BMD45 detector classed `motorcycle` can be removed. Every other box (car / truck / bus from YOLO11x) is never touched.
Rule R1  : a box that fully encloses >= 3 DISTINCT motorcycle boxes and is >= 10x larger than their median area is an aggregate box, not one vehicle.
Rule R2  : same idea for a box at the BMD45 size cap that also encloses a YOLO11x vehicle (never fires alone in S0-G: R1 already covers it).
Flag only: size cap, contains / overlaps a YOLO11x vehicle, nested same-size box, encloses some boxes. A size cap alone NEVER removes a box.
Decisions use the full conf >= 0.08 set of an image (independent of the later confidence threshold).

Row format (dict): image, row (unique int per image set), conf, x0, y0, x1, y1 (native image pixels), canon, source_model, raw_id.
"""
from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path

import numpy as np

CONFIG_PATH = Path(__file__).with_name("hybrid_cleanup_config.json")
BMD45_MODEL = "yolo12x_bmd45"
# BMD45 boxes saturate at this size in 512x288 images with imgsz 1280 (S0-E finding, data-derived; cause unknown). Used for FLAGS and rule R2 only.
SIZE_CAP_PX_512 = (67.7, 75.3)
CAP_REFERENCE_WIDTH = 512
RULE_REASON = {"R1_ENCLOSES_CLUSTER": "box encloses >=3 distinct motorcycle boxes and is >=10x larger than their median area (aggregate box, not one vehicle)",
               "R2_CAPPED_ENCLOSES_VEHICLE_AND_MOTOS": "box at BMD45 size cap in both dimensions encloses a YOLO11x vehicle and >=2 distinct motorcycle boxes"}


def load_config(path=None) -> dict:
    return json.loads(Path(path or CONFIG_PATH).read_text(encoding="utf-8"))


def config_sha256(path=None) -> str:
    return hashlib.sha256(Path(path or CONFIG_PATH).read_bytes()).hexdigest()


def cap_for_width(img_w: int) -> tuple:
    """size cap scaled linearly with image width (identity for 512-wide images)"""
    s = img_w / CAP_REFERENCE_WIDTH
    return (SIZE_CAP_PX_512[0] * s, SIZE_CAP_PX_512[1] * s)


def box(r):
    return (r["x0"], r["y0"], r["x1"], r["y1"])


def area(r):
    return (r["x1"] - r["x0"]) * (r["y1"] - r["y0"])


def inter(a, b):
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def iou(a, b):
    i = inter(a, b); u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def ioa(a, b):
    """intersection over the area of a"""
    aa = (a[2] - a[0]) * (a[3] - a[1])
    return inter(a, b) / aa if aa > 0 else 0.0


def is_bmd_moto(r) -> bool:
    return r["source_model"] == BMD45_MODEL and r["canon"] == "motorcycle"


def distinct(boxes, max_iou):
    """greedy distinct subset (conf desc, deterministic tie-break): pairwise IoU < max_iou"""
    keep = []
    for b in sorted(boxes, key=lambda r: (-r["conf"], r["x0"], r["y0"], r["x1"], r["y1"], r["raw_id"])):
        if all(iou(box(b), box(k)) < max_iou for k in keep):
            keep.append(b)
    return keep


def metrics_for(r, motos, veh, cap, cfg, image_wh):
    E = cfg["evidence"]; b = box(r); w, h = b[2] - b[0], b[3] - b[1]
    ins_all = [o for o in motos if o["row"] != r["row"] and ioa(box(o), b) >= E["inside_ioa"]]
    ins = [o for o in ins_all if o["conf"] >= E["min_inside_conf"]]
    dist = distinct(ins, E["inside_distinct_max_iou"])
    med = float(np.median([area(o) for o in dist])) if dist else 0.0
    centers = sum(1 for o in motos if o["row"] != r["row"] and b[0] <= (o["x0"] + o["x1"]) / 2 <= b[2] and b[1] <= (o["y0"] + o["y1"]) / 2 <= b[3])
    vin = [v for v in veh if ioa(box(v), b) >= E["vehicle_inside_ioa"]]
    ov = max(((iou(b, box(v)), ioa(b, box(v)), v) for v in veh), key=lambda z: max(z[0], z[1]), default=(0.0, 0.0, None))
    tol = cfg["size_cap_tol_px"]
    same = [o for o in motos if o["row"] != r["row"] and abs((o["x1"] - o["x0"]) - w) <= 1.0 and abs((o["y1"] - o["y0"]) - h) <= 1.0 and o["conf"] > r["conf"] and iou(b, box(o)) >= 0.5]
    return dict(w=round(w, 2), h=round(h, 2), area=round(w * h, 1), aspect_w_over_h=round(w / h, 3), area_ratio_image=round(w * h / (image_wh[0] * image_wh[1]), 4),
                size_cap_w=w >= cap[0] - tol, size_cap_h=h >= cap[1] - tol, n_inside_all=len(ins_all), n_inside_distinct=len(dist), median_inside_area=round(med, 1),
                ratio_to_inside_median=round(w * h / med, 2) if med > 0 else 0.0, n_centers_inside=centers, n_yolo_vehicles_inside=len(vin),
                yolo_vehicle_inside_class="|".join(sorted({v["canon"] for v in vin})), max_iou_yolo_vehicle=round(ov[0], 3), max_ioa_covered_by_yolo_vehicle=round(ov[1], 3),
                overlapped_vehicle_class=ov[2]["canon"] if ov[2] and (ov[0] >= E["overlap_vehicle_iou"] or ov[1] >= E["covered_by_vehicle_ioa"]) else "",
                nested_same_size=bool(same), conf=r["conf"])


def decide(rows, cfg, image_wh_of, cap_of=None):
    """-> {raw_id: dict(status KEEP | KEEP_FLAG | REMOVE, rule, flags, m)} for BMD45 motorcycle rows only.
    rows: all hybrid rows of one or many images (key r["image"]); image_wh_of(image) -> (w, h); cap_of(image) -> (cap_w, cap_h) (default: scaled S0-E cap)."""
    cap_of = cap_of or (lambda image: cap_for_width(image_wh_of(image)[0]))
    out = {}; by = collections.defaultdict(list)
    for r in rows:
        by[r["image"]].append(r)
    R1, R2 = cfg["auto_suppress_rules"]["R1_ENCLOSES_CLUSTER"], cfg["auto_suppress_rules"]["R2_CAPPED_ENCLOSES_VEHICLE_AND_MOTOS"]
    for image in sorted(by):
        L = by[image]; motos = [r for r in L if is_bmd_moto(r)]; veh = [r for r in L if not is_bmd_moto(r)]; cap = cap_of(image); wh = image_wh_of(image)
        for r in sorted(motos, key=lambda z: z["row"]):
            m = metrics_for(r, motos, veh, cap, cfg, wh); flags = []
            if m["size_cap_w"]: flags.append("SIZE_CAP_W")
            if m["size_cap_h"]: flags.append("SIZE_CAP_H")
            if m["n_yolo_vehicles_inside"] >= 1: flags.append("CONTAINS_YOLO_VEHICLE")
            if m["overlapped_vehicle_class"]: flags.append("OVERLAP_YOLO_VEHICLE")
            if m["nested_same_size"]: flags.append("NESTED_SAME_SIZE")
            rule = None
            if m["n_inside_distinct"] >= R1["min_inside_distinct"] and m["ratio_to_inside_median"] >= R1["min_area_ratio_to_inside_median"]:
                rule = "R1_ENCLOSES_CLUSTER"
            elif (m["size_cap_w"] and m["size_cap_h"] and m["n_inside_distinct"] >= R2["min_inside_distinct"] and m["ratio_to_inside_median"] >= R2["min_area_ratio_to_inside_median"]
                  and m["n_yolo_vehicles_inside"] >= 1):
                rule = "R2_CAPPED_ENCLOSES_VEHICLE_AND_MOTOS"
            if not rule and m["n_inside_distinct"] >= 2:
                flags.append("ENCLOSES_SOME_NOT_SUPPRESSED")
            out[r["raw_id"]] = dict(status="REMOVE" if rule else ("KEEP_FLAG" if flags else "KEEP"), rule=rule, flags=flags, m=m)
    return out


def apply(rows, dec):
    kept, removed = [], []
    for r in rows:
        d = dec.get(r["raw_id"])
        (removed if d and d["status"] == "REMOVE" else kept).append(r)
    return kept, removed
