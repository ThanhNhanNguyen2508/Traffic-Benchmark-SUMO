"""Resolve and validate a calibration artifact for a given detector backend (J02E).

A calibration JSON (calibrate.py output) is only valid for the backend whose detections it was fitted on.
This module never falls back to another backend's calibration: any doubt -> status INVALID (exit code 2).

Usage
    python -m flow_estimation.calibration_artifact --path P --backend yolo11x|hybrid_cleaned
        [--dets-provenance CITY_CACHE.provenance.json] [--out resolved.json]

P is a calibration JSON, or a directory with one of these layouts (tried in this order)
    calibration_only run dir : calibration.json + detections.jsonl.gz
    full_recompute OUT_DIR   : calibration/<B>_recomputed.json + detections/calib_<B>.jsonl.gz
    hybrid cache dir         : hybrid_cleaned_calibration.json + calib_hybrid_cleaned.jsonl.gz

Checks: schema (classes, conf in (0,1), beta finite and > 0 for every gap and class); backend consistency via
<calibration>.provenance.json (hybrid_cleaned: required, backend + calibration sha256 + calibration-detection-cache sha256 must match;
yolo11x: optional, but rejected if it names another backend); with --dets-provenance: its backend must equal --backend and, for hybrid,
its weights sha256 must equal those recorded for the calibration.
ground_truth_kind comes from provenance["ground_truth"]["kind"], else "REFERENCE_ASSUMED" (repository baseline reference);
"SYNTHETIC_GT_CODE_TEST" => scientific_validity "NONE (synthetic ground truth, code test only)"; any other kind starting with
"SYNTHETIC" (e.g. SYNTHETIC_INTEGRATION_TEST) => "NONE (synthetic ..., code/integration test only)".
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

from .common import CLASSES

SCHEMA = "calibration-artifact-1"
BACKENDS = ("yolo11x", "hybrid_cleaned")


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _locate(p: Path, backend: str):
    """-> (layout, calibration path, calib detections cache path or None)."""
    if p.is_file():
        return "file", p, None
    if not p.is_dir():
        return None, None, None
    cands = [("calibration_only", p / "calibration.json", p / "detections.jsonl.gz"),
             ("full_recompute", p / "calibration" / f"{backend}_recomputed.json", p / "detections" / f"calib_{backend}.jsonl.gz"),
             ("hybrid_cache_dir", p / "hybrid_cleaned_calibration.json", p / "calib_hybrid_cleaned.jsonl.gz")]
    for name, cal, dets in cands:
        if cal.is_file():
            return name, cal, (dets if dets.is_file() else None)
    return None, None, None


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _check_schema(cal, problems):
    if cal.get("classes") != CLASSES:
        problems.append(f"classes {cal.get('classes')} != {CLASSES}")
    conf = cal.get("conf")
    if not isinstance(conf, dict):
        problems.append("missing conf")
    else:
        for k in ("motorcycle", "other"):
            v = conf.get(k)
            if not (_num(v) and 0.0 < v < 1.0):
                problems.append(f"conf.{k}={v!r} not a number in (0,1)")
    gaps, beta = cal.get("gaps_s"), cal.get("beta")
    if not isinstance(gaps, list) or not gaps or not isinstance(beta, dict):
        problems.append("missing gaps_s / beta")
        return
    for g in gaps:
        vals = beta.get(str(g))
        if not isinstance(vals, list) or len(vals) != len(CLASSES):
            problems.append(f"beta gap={g}: expected {len(CLASSES)} values, got {vals!r}")
            continue
        for c, v in zip(CLASSES, vals):
            if not (_num(v) and v > 0):
                problems.append(f"non-finite/invalid beta class={c} gap={g} (calibration set too small for this class)")


def resolve(path, backend, dets_provenance=None) -> dict:
    res = {"schema_version": SCHEMA, "status": "INVALID", "backend": backend, "layout": None, "calibration": None,
           "calibration_sha256": None, "provenance": None, "ground_truth_kind": None, "scientific_validity": None,
           "conf": None, "problems": []}
    pr = res["problems"]
    if backend not in BACKENDS:
        pr.append(f"backend {backend!r} not in {BACKENDS}")
        return res
    layout, cal_path, dets_path = _locate(Path(path), backend)
    res["layout"] = layout
    if cal_path is None:
        pr.append(f"no calibration found at {path} for backend {backend}")
        return res
    res["calibration"] = str(cal_path)
    res["calibration_sha256"] = sha256_file(cal_path)
    try:
        cal = json.loads(cal_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        pr.append(f"calibration unreadable: {e}")
        return res
    _check_schema(cal, pr)
    if isinstance(cal.get("conf"), dict):
        res["conf"] = cal["conf"]

    prov_path = Path(str(cal_path) + ".provenance.json")
    prov = None
    if prov_path.is_file():
        res["provenance"] = str(prov_path)
        try:
            prov = json.loads(prov_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            pr.append(f"provenance unreadable: {e}")
    if backend == "hybrid_cleaned":
        if prov is None:
            pr.append(f"hybrid_cleaned requires {prov_path.name} (not found / unreadable)")
        else:
            if prov.get("backend") != "hybrid_cleaned":
                pr.append(f"provenance backend {prov.get('backend')!r} != 'hybrid_cleaned'")
            if prov.get("calibration_json_sha256") != res["calibration_sha256"]:
                pr.append("provenance calibration_json_sha256 != sha256 of the calibration file")
            want = prov.get("detection_cache_calibration_sha256")
            if dets_path is not None and want is not None and sha256_file(dets_path) != want:
                pr.append("calibration detection cache sha256 != provenance detection_cache_calibration_sha256")
    elif prov is not None and prov.get("backend") not in (None, "yolo11x"):
        pr.append(f"provenance backend {prov.get('backend')!r} != 'yolo11x'")

    if dets_provenance is not None:
        try:
            dp = json.loads(Path(dets_provenance).read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            dp = None
            pr.append(f"dets provenance unreadable: {e}")
        if dp is not None:
            if dp.get("backend") != backend:
                pr.append(f"dets provenance backend {dp.get('backend')!r} != {backend!r}")
            if backend == "hybrid_cleaned" and prov is not None:
                for m in ("yolo11x", "bmd45"):
                    a = ((dp.get("weights") or {}).get(m) or {}).get("sha256")
                    b = ((prov.get("weights") or {}).get(m) or {}).get("sha256")
                    if a is None or b is None or a != b:
                        pr.append(f"weights.{m}.sha256 differ between dets provenance ({a}) and calibration provenance ({b})")

    gt = ((prov or {}).get("ground_truth") or {}).get("kind")
    res["ground_truth_kind"] = gt or "REFERENCE_ASSUMED"
    res["scientific_validity"] = ("NONE (synthetic ground truth, code test only)" if gt == "SYNTHETIC_GT_CODE_TEST"
                                  else f"NONE (synthetic {gt}, code/integration test only)" if isinstance(gt, str) and gt.startswith("SYNTHETIC")
                                  else "reference-based; no independent validation")
    res["status"] = "INVALID" if pr else "OK"
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description="Validate a calibration artifact for a detector backend (never falls back).")
    ap.add_argument("--path", required=True)
    ap.add_argument("--backend", required=True, choices=BACKENDS)
    ap.add_argument("--dets-provenance")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    res = resolve(a.path, a.backend, a.dets_provenance)
    txt = json.dumps(res, indent=1, ensure_ascii=False)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(txt, encoding="utf-8")
    print(txt)
    return 0 if res["status"] == "OK" else 2


if __name__ == "__main__":
    sys.exit(main())
