#!/usr/bin/env python3
"""Build and validate a city-camera frame index (schema "city-index-report-1").

Discovers snapshot folders, maps camera folder names to canonical camera IDs (never guessed:
unknown folders are errors), attaches a timestamp to every frame, checks image geometry and
timestamps, and writes an index usable by ``infer --index`` / ``directional --index``.

Layouts (session = directory that contains camera directories; camera dir = directory with
frame_*.jpg/.jpeg/.png):
    <root>/<session>/<Cam>/frame_*.jpg (+ metadata.csv)     # root holds sessions
    <root>/<Cam>/frame_*.jpg                                # root itself is a session (name = root name)
Timestamp per frame: metadata.csv column received_at_vn (matched by filename) if present
[timestamp_source=metadata_csv]; else the filename
frame_<idx>_<YYYYMMDD>_<HHMMSS>_<ms>.jpg interpreted in --tz [timestamp_source=filename].
Frames with neither are reported as errors (timestamps are never fabricated).

    python -m flow_estimation.city_index --root data/city_cam --exclude Cam_To_Hien_Thanh_Su_Van_Hanh \
        --out index.jsonl --report report.json
    python -m flow_estimation.city_index --root data/5cam_14p_extracted --require Cam_To_Hien_Thanh_1 \
        --window 5cam_14p 13:06 13:21 --out index.jsonl --report report.json

Index line: {"session","camera","t","image" (posix, relative to --root),"width","height","timestamp_source"}.
Errors (exit code 2): unknown camera folder, missing --require camera, unreadable image, aspect != 512:288
(+-1%; ROI cameras use the roi.py reference size), duplicate timestamps in a camera, frames without timestamp.
Gaps > 35 s are only counted (info). --window SESSION HH:MM HH:MM keeps t in [from - margin, to] on the date
of the session's first frame; sessions without a --window are dropped when any --window is given.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt_
import hashlib
import json
import re
import sys
from pathlib import Path

from PIL import Image

try:
    from . import roi as roi_mod
except ImportError:  # direct script execution
    import roi as roi_mod  # type: ignore

SCHEMA = "city-index-report-1"
IMG_EXT = (".jpg", ".jpeg", ".png")
FNAME_RE = re.compile(r"^frame_(\d+)_(\d{8})_(\d{6})_(\d{3})\.(jpe?g|png)$", re.IGNORECASE)
MAX_GAP_S = 35.0
DEFAULT_ALIASES = Path(__file__).resolve().parent / "camera_aliases.json"
DEFAULT_ROI = Path(__file__).resolve().parent / "directional_roi.json"


def parse_tz(s: str) -> dt_.timezone:
    m = re.fullmatch(r"([+-])(\d{2}):?(\d{2})", s.strip())
    if not m:
        raise ValueError(f"bad --tz {s!r} (expected +HH:MM)")
    d = dt_.timedelta(hours=int(m.group(2)), minutes=int(m.group(3)))
    return dt_.timezone(-d if m.group(1) == "-" else d)


def _is_img(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXT and p.name.lower().startswith("frame_")


def is_camera_dir(d: Path) -> bool:
    try:
        return any(_is_img(p) for p in d.iterdir() if p.is_file())
    except OSError:
        return False


def camera_dirs(d: Path):
    return sorted(c for c in d.iterdir() if c.is_dir() and is_camera_dir(c))


def find_sessions(root: Path):
    """-> {session_name: [camera dirs]}"""
    if camera_dirs(root):
        return {root.name: camera_dirs(root)}
    out = {}
    for s in sorted(c for c in root.iterdir() if c.is_dir()):
        cams = camera_dirs(s)
        if cams:
            out[s.name] = cams
    return out


def load_aliases(path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        a = json.load(f)
    if a.get("schema_version") != "camera-aliases-1":
        raise ValueError(f"{path}: schema_version {a.get('schema_version')!r} != 'camera-aliases-1'")
    return a


def canonical_camera(folder: str, aliases: dict):
    """folder name -> canonical camera ID or None (never guessed)."""
    cams = aliases["cameras"]
    if folder in cams:
        return folder
    base = re.sub(aliases.get("strip_suffix_regex", "$^"), "", folder)
    if base in cams:
        return base
    for canon, al in cams.items():
        if base in al:
            return canon
    return None


def _frames_of(cam_dir: Path, tz, warnings, errors, label):
    """-> list of (datetime, t_string, Path, source)"""
    files = sorted(p for p in cam_dir.iterdir() if p.is_file() and _is_img(p))
    out, meta = [], None
    mp = cam_dir / "metadata.csv"
    if mp.is_file():
        meta = {}
        with mp.open("r", encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                if r.get("filename") and r.get("received_at_vn"):
                    meta[r["filename"]] = r["received_at_vn"].strip()
    bad = []
    for p in files:
        if meta is not None:
            ts = meta.get(p.name)
            if ts:
                try:
                    out.append((dt_.datetime.fromisoformat(ts), ts, p, "metadata_csv"))
                    continue
                except ValueError:
                    pass
            bad.append(p.name)
            continue
        m = FNAME_RE.match(p.name)
        if not m:
            bad.append(p.name)
            continue
        try:
            d = dt_.datetime.strptime(m.group(2) + m.group(3), "%Y%m%d%H%M%S").replace(
                microsecond=int(m.group(4)) * 1000, tzinfo=tz)
        except ValueError:
            bad.append(p.name)
            continue
        out.append((d, d.isoformat(timespec="milliseconds"), p, "filename"))
    if bad:
        errors.append(f"{label}: {len(bad)} frame(s) without a usable timestamp (invalid, not indexed), "
                      f"e.g. {bad[:3]}")
    if meta is not None:
        miss = len(set(meta) - {p.name for p in files})
        if miss:
            warnings.append(f"{label}: {miss} metadata.csv row(s) without an image file (ignored)")
    return out


def serialize(rows) -> str:
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


def discover(root, sessions=None, exclude=(), require=(), aliases=DEFAULT_ALIASES, roi=DEFAULT_ROI,
             tz="+07:00", windows=None, window_margin_s=60.0):
    """-> (rows, report). rows sorted by session, camera, t. See module docstring."""
    root = Path(root)
    tzi = parse_tz(tz)
    al = load_aliases(aliases)
    rois = roi_mod.load_roi(roi) if roi and Path(roi).is_file() else {}
    windows = {w[0]: (w[1], w[2]) for w in (windows or [])}
    exclude = set(exclude)
    errors, warnings, rows = [], [], []
    rep = {"schema_version": SCHEMA, "root": root.as_posix(), "sessions": {}, "excluded": [],
           "errors": errors, "warnings": warnings}
    if not root.is_dir():
        errors.append(f"root {root} is not a directory")
        rep.update(index_sha256=None, n_frames=0)
        return rows, rep
    allsess = find_sessions(root)
    if not allsess:
        errors.append(f"no session/camera folders with frame_*.jpg found under {root}")
    for sname in windows:
        if sname not in allsess:
            errors.append(f"--window session {sname!r} not found (have {sorted(allsess)})")
    for sname, cams in allsess.items():
        if sessions and sname not in sessions:
            continue
        if windows and sname not in windows:
            continue
        scam, seen = {}, {}
        for cd in cams:
            canon = canonical_camera(cd.name, al)
            if canon is None:
                if cd.name in exclude:
                    rep["excluded"].append({"session": sname, "folder": cd.name, "camera": None})
                    continue
                errors.append(f"{sname}: unknown camera folder {cd.name!r} (add it to the aliases file; never guessed)")
                continue
            if canon in exclude or cd.name in exclude:
                rep["excluded"].append({"session": sname, "folder": cd.name, "camera": canon})
                continue
            if canon in seen:
                errors.append(f"{sname}: folders {seen[canon]!r} and {cd.name!r} both map to {canon}")
                continue
            seen[canon] = cd.name
            scam[canon] = cd
        for r in require:
            if r not in scam:
                errors.append(f"{sname}: required camera {r} missing")
        fr = {c: _frames_of(d, tzi, warnings, errors, f"{sname}/{c}") for c, d in scam.items()}
        if sname in windows:
            firsts = [min(f[0] for f in v) for v in fr.values() if v]
            if firsts:
                first = min(firsts)
                try:
                    a, b = (dt_.datetime.strptime(x, "%H:%M").time() for x in windows[sname])
                except ValueError:
                    errors.append(f"--window {sname}: times must be HH:MM")
                    continue
                lo = dt_.datetime.combine(first.date(), a, tzinfo=first.tzinfo) - dt_.timedelta(seconds=window_margin_s)
                hi = dt_.datetime.combine(first.date(), b, tzinfo=first.tzinfo)
                fr = {c: [f for f in v if lo <= f[0] <= hi] for c, v in fr.items()}
        cams_rep = {}
        for canon in sorted(fr):
            v = sorted(fr[canon], key=lambda f: (f[0], f[2].name))
            label = f"{sname}/{canon}"
            if not v:
                warnings.append(f"{label}: no frames after filtering")
                continue
            ts = [f[0].timestamp() for f in v]
            dups = len(ts) - len(set(ts))
            if dups:
                errors.append(f"{label}: {dups} duplicate timestamp(s)")
            gaps = sum(1 for i in range(1, len(ts)) if ts[i] - ts[i - 1] > MAX_GAP_S)
            sizes, unreadable, geom = {}, [], []
            for d, t, p, src in v:
                try:
                    with Image.open(p) as im:
                        w, h = im.size
                except Exception as e:  # noqa: BLE001
                    unreadable.append(f"{p.name}: {type(e).__name__}")
                    continue
                sizes[(w, h)] = sizes.get((w, h), 0) + 1
                try:
                    if canon in rois and rois[canon].reference_size is not None:
                        rois[canon].check_geometry(w, h)
                    elif not roi_mod.check_aspect(w, h):
                        raise ValueError(f"{canon}: image {w}x{h} aspect != reference 512x288 aspect (tol 0.01)")
                except ValueError as e:
                    geom.append(f"{p.name}: {e}")
                rows.append({"session": sname, "camera": canon, "t": t,
                             "image": p.relative_to(root).as_posix(), "width": w, "height": h,
                             "timestamp_source": src})
            if unreadable:
                errors.append(f"{label}: {len(unreadable)} unreadable image(s), e.g. {unreadable[:3]}")
            if geom:
                errors.append(f"geometry {label}: {len(geom)} frame(s) not 16:9 (512:288 +-1%), "
                              f"sizes {sorted(sizes)}; e.g. {geom[0]}")
            if len(sizes) > 1:
                warnings.append(f"{label}: mixed image sizes {sorted(sizes)}")
            w0, h0 = max(sizes, key=sizes.get) if sizes else (None, None)
            srcs = sorted({f[3] for f in v})
            cams_rep[canon] = {"source_folder": scam[canon].name, "n_frames": len(v),
                               "first_t": v[0][1], "last_t": v[-1][1], "width": w0, "height": h0,
                               "n_gaps_gt_35s": gaps,
                               "timestamp_source": srcs[0] if len(srcs) == 1 else "mixed"}
        rep["sessions"][sname] = {"cameras": cams_rep}
    rows.sort(key=lambda r: (r["session"], r["camera"], dt_.datetime.fromisoformat(r["t"]).timestamp(), r["image"]))
    rep["n_frames"] = len(rows)
    rep["index_sha256"] = hashlib.sha256(serialize(rows).encode("utf-8")).hexdigest()
    return rows, rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--sessions", nargs="+", default=None)
    ap.add_argument("--exclude", nargs="*", default=[], help="canonical camera IDs (or folder names) to skip")
    ap.add_argument("--require", nargs="*", default=[], help="canonical camera IDs that must be present")
    ap.add_argument("--aliases", type=Path, default=DEFAULT_ALIASES)
    ap.add_argument("--roi", type=Path, default=DEFAULT_ROI, help="directional_roi.json (reference size for geometry)")
    ap.add_argument("--tz", default="+07:00", help="timezone of filename timestamps")
    ap.add_argument("--window", nargs=3, action="append", metavar=("SESSION", "FROM", "TO"), default=None)
    ap.add_argument("--window-margin-s", type=float, default=60.0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    a = ap.parse_args(argv)
    try:
        rows, rep = discover(a.root, a.sessions, a.exclude, a.require, a.aliases, a.roi, a.tz, a.window,
                             a.window_margin_s)
    except (ValueError, OSError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.report.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", encoding="utf-8", newline="") as f:
        f.write(serialize(rows))
    with a.report.open("w", encoding="utf-8") as f:
        json.dump(rep, f, indent=1, ensure_ascii=False)
    for s, sv in rep["sessions"].items():
        for c, cv in sv["cameras"].items():
            print(f"{s} {c}: {cv['n_frames']} frames {cv['width']}x{cv['height']} {cv['timestamp_source']} "
                  f"{cv['first_t']} .. {cv['last_t']} gaps>35s={cv['n_gaps_gt_35s']}")
    print(f"frames={rep['n_frames']} excluded={len(rep['excluded'])} warnings={len(rep['warnings'])} "
          f"errors={len(rep['errors'])} index_sha256={rep['index_sha256']}")
    for e in rep["errors"]:
        print("ERROR:", e)
    for w in rep["warnings"]:
        print("WARN:", w)
    return 2 if rep["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
