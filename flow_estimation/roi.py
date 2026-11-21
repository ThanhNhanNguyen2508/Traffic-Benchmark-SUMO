"""Directional ROI labelling of detections (J02, schema "j02-roi-1").

Each camera has a set of named polygons (in normalized image coordinates) that
partition the frame into direction labels:

* "FORWARD" / "REVERSE"  - the two traffic directions that are counted;
* "EXCLUDE"              - region that is never counted (e.g. the far side of
                           the outer boundary line).

Semantics
---------
* Anchor of a detection is its footpoint ``(x_center, y_bottom)``.
* Detection cache entries are ``[cls, x_center, y_bottom, h, w, conf]`` in the
  1920x1080 detection space (512x288 frames x 3.75).
* Normalized coordinates are ``xn = x_px / width``, ``yn = y_px / height`` of the
  reference image (continuous pixel-edge convention, NOT /(width-1)); hence
  ``xn = x_det / 1920`` and ``yn = y_det / 1080``.  Both axes are clamped to
  ``[0, HI]`` with ``HI = 1.0 - 1e-9`` (``CLAMP_HI``) before testing: values
  below 0 become 0 and values ``>= HI`` (on/below the bottom border, on/after the
  right border) become HI, i.e. the point just inside the image.  This is needed
  because the half-open edge rule would otherwise reject points exactly on
  ``y = 1.0`` / ``x = 1.0`` (where polygon bottom edges lie) and give them
  ``default_label``.  Values in ``[0, HI)`` are unchanged.
* Assignment: the polygons are tested in the order given by ``order``; the first
  polygon containing the point wins (even-odd ray casting with half-open edges,
  so the result is deterministic on boundaries).  A point in no polygon gets
  ``default_label`` - every point therefore receives exactly one label and
  EXCLUDE is never forced into FORWARD/REVERSE.

Geometry: the normalized polygons are only valid for frames with the aspect ratio of
the reference image (512x288, 16:9).  ``CameraROI.reference_size`` holds the reference
(width, height) read from ``reference_image`` in the JSON; ``check_geometry(w, h)`` /
``check_aspect(w, h)`` raise / return False when a frame's aspect differs by more than
1% (relative), e.g. a 4:3 or portrait frame.  This is a guard only; labelling is unchanged.

Only the standard library and pathlib are used.
"""

from __future__ import annotations

import json
from pathlib import Path

SCHEMA_VERSION = "j02-roi-1"
LABELS_ALL = ("FORWARD", "REVERSE", "EXCLUDE")
DET_W = 1920.0
DET_H = 1080.0
CLAMP_HI = 1.0 - 1e-9  # upper clamp of normalized coords (see module docstring)


def point_in_polygon(x, y, poly) -> bool:
    """Even-odd ray casting, half-open edges (deterministic on edges/vertices)."""
    inside = False
    n = len(poly)
    if n < 3:
        return False
    x1, y1 = poly[n - 1]
    for i in range(n):
        x2, y2 = poly[i]
        if (y1 > y) != (y2 > y):
            xc = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < xc:
                inside = not inside
        x1, y1 = x2, y2
    return inside


class CameraROI:
    """Direction labelling for one camera."""

    def __init__(self, name, labels, polygons, order, default_label, reference_size=None):
        self.name = str(name)
        self.reference_size = None if reference_size is None else (int(reference_size[0]), int(reference_size[1]))
        self.labels = tuple(labels)
        self.directions = tuple(lb for lb in self.labels if lb != "EXCLUDE")
        self.order = tuple(order)
        self.default_label = default_label
        self.polygons = {k: [(float(p[0]), float(p[1])) for p in v] for k, v in polygons.items()}
        for k in self.order:
            if k not in self.polygons:
                raise ValueError(f"{self.name}: order label {k!r} has no polygon")
        for lb in list(self.order) + [default_label]:
            if lb not in self.labels:
                raise ValueError(f"{self.name}: label {lb!r} not in labels {self.labels}")
        self._bbox = {
            k: (min(p[0] for p in v), min(p[1] for p in v), max(p[0] for p in v), max(p[1] for p in v))
            for k, v in self.polygons.items()
        }

    def check_geometry(self, width, height, tol=0.01):
        """Raise ValueError if the image aspect differs from the reference aspect by > tol (relative)."""
        if self.reference_size is None:
            return
        if not check_aspect(width, height, self.reference_size, tol):
            rw, rh = self.reference_size
            raise ValueError(f"{self.name}: image {width}x{height} aspect "
                             f"{float(width) / float(height) if height else float('nan'):.4f} != reference "
                             f"{rw}x{rh} aspect {rw / rh:.4f} (tol {tol})")

    def label_norm(self, xn: float, yn: float) -> str:
        xn = 0.0 if xn < 0.0 else (CLAMP_HI if xn >= CLAMP_HI else float(xn))
        yn = 0.0 if yn < 0.0 else (CLAMP_HI if yn >= CLAMP_HI else float(yn))
        for k in self.order:
            x0, y0, x1, y1 = self._bbox[k]
            if x0 <= xn <= x1 and y0 <= yn <= y1 and point_in_polygon(xn, yn, self.polygons[k]):
                return k
        return self.default_label

    def label_det(self, x: float, y: float) -> str:
        return self.label_norm(x / DET_W, y / DET_H)

    def split(self, dets) -> dict:
        """dets [[cls, x, y_bottom, h, w, conf], ...] -> {label: [det, ...]} (det order kept)."""
        out = {lb: [] for lb in self.labels}
        for d in dets:
            out[self.label_det(d[1], d[2])].append(d)
        return out


def check_aspect(width, height, ref=(512, 288), tol=0.01) -> bool:
    """True if width/height matches ref aspect within relative tolerance tol."""
    if not width or not height or width <= 0 or height <= 0:
        return False
    a, r = float(width) / float(height), float(ref[0]) / float(ref[1])
    return abs(a - r) / r <= tol


def load_roi(path) -> dict:
    """Load a j02-roi-1 JSON file -> {camera: CameraROI}."""
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{path}: schema_version {data.get('schema_version')!r} != {SCHEMA_VERSION!r}")
    out = {}
    for cam, c in data["cameras"].items():
        ri = c.get("reference_image") or {}
        ref = (ri["width"], ri["height"]) if "width" in ri and "height" in ri else None
        out[cam] = CameraROI(cam, c["labels"], c["polygons"], c["order"], c["default_label"], reference_size=ref)
    return out
