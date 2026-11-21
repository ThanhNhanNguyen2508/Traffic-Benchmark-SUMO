#!/usr/bin/env python3
"""Build the simplified study network sumo/map/hcmc.net.xml.gz.

The OSM import (sumo/map/osm.net.xml.gz) is kept only as a geometry source. The
study network keeps the real junction positions and street shapes, and applies
the team's confirmed specification (2026-11-08):

* one node and one traffic light per junction (J0-J5); J4 keeps a default
  fixed-time light and is not an agent (edge of the study area);
* Lý Thường Kiệt and Bắc Hải: hard median, 2 lanes x 3.5 m per direction
  (~3 cars / 2 trucks abreast); other streets: painted centre line,
  2 lanes x 3.0 m per direction; Hẻm 319A: 1 lane x 3.0 m per direction;
* alleys and small roads removed (Hẻm 319A kept: it is an arm of J0);
* no pedestrians, no U-turns;
* turn rules and signal groups per junction as described by the team, see SPEC.

Signal timings are placeholders (30 s green + 3 s yellow per phase) until the
real plans are measured.

Usage:  python build_network.py
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
import sumolib  # noqa: E402

ROOT = Path(__file__).resolve().parent
OSM_NET = ROOT / "sumo" / "map" / "osm.net.xml.gz"
OUT_NET = ROOT / "sumo" / "map" / "hcmc.net.xml.gz"

GREEN_TIME, YELLOW_TIME = 30, 3
ARM_CAP = 400.0      # m: boundary arms are cut at this length
ARM_MIN = 150.0      # m: short boundary arms are extended straight to this length

# junction id -> (name, approximate centre in the OSM network)
JUNCTIONS = {
    "J0": ("Lý Thường Kiệt / Bắc Hải", (344.2, 1555.7)),
    "J1": ("Lý Thường Kiệt / Đường Số 2", (533.1, 907.2)),
    "J2": ("Lý Thường Kiệt / Tô Hiến Thành / Lữ Gia", (579.7, 739.9)),
    "J3": ("Thành Thái / Tô Hiến Thành", (1167.5, 1379.6)),
    "J4": ("Tô Hiến Thành / Đồng Nai", (1276.9, 1478.0)),
    "J5": ("Bắc Hải / Thành Thái", (657.4, 1872.5)),
}

# street -> (lanes per direction, lane width, speed m/s)
STREETS = {
    "Lý Thường Kiệt": (2, 3.5, 13.89),
    "Bắc Hải": (2, 3.5, 13.89),
    "Thành Thái": (2, 3.0, 13.89),
    "Tô Hiến Thành": (2, 3.0, 13.89),
    "Lữ Gia": (2, 3.0, 13.89),
    "Đường Số 2": (2, 3.0, 11.11),
    "Đồng Nai": (2, 3.0, 11.11),
    "Hẻm 319A Lý Thường Kiệt": (1, 3.0, 8.33),
}
ABBR = {"Lý Thường Kiệt": "LTK", "Bắc Hải": "BH", "Thành Thái": "TT", "Tô Hiến Thành": "THT",
        "Lữ Gia": "LG", "Đường Số 2": "DS2", "Đồng Nai": "DN", "Hẻm 319A Lý Thường Kiệt": "H319A"}

# road segments: (street, from node, to node). Boundary nodes start with "B".
# boundary target: "maxy" / "miny" / "minx" / "maxx" = farthest point of that street in OSM,
# or ("extend", junction, street) = continue the street straight through the junction.
SEGMENTS = [
    ("Lý Thường Kiệt", "B_LTK_N", "J0", "maxy"),
    ("Lý Thường Kiệt", "J0", "J1", None),
    ("Lý Thường Kiệt", "J1", "J2", None),
    ("Lý Thường Kiệt", "J2", "B_LTK_S", "miny"),
    ("Hẻm 319A Lý Thường Kiệt", "B_H319A", "J0", "minx"),
    ("Bắc Hải", "J0", "J5", None),
    ("Bắc Hải", "J5", "B_BH_E", ("extend", "J5", "Bắc Hải")),   # not in OSM, 250 m
    ("Đường Số 2", "B_DS2", "J1", "minx"),
    ("Lữ Gia", "B_LG", "J2", "minx"),
    ("Tô Hiến Thành", "J2", "J3", None),
    ("Tô Hiến Thành", "J3", "J4", None),
    ("Tô Hiến Thành", "J4", "B_THT_E", "maxx"),
    ("Thành Thái", "J5", "J3", None),
    ("Thành Thái", "J3", "B_TT_S", "miny"),
    ("Đồng Nai", "J4", "B_DN", "maxy"),
]

# Turn rules, as confirmed by the team. Arms are named by the neighbour node.
#   "free"    always allowed (no signal; yields to conflicting green traffic)
#   "ban"     not allowed
#   other     name of the signal group (one signal head) controlling the movement
# "strict": conflicting groups never share a phase (lights protect turns).
# "permissive": approach lights; a left turn may run with the opposite straight and yields.
SPEC = {
    "J0": {"mode": "strict", "rules": {
        ("J5", "J1"): "BH_left", ("J5", "B_H319A"): "ban", ("J5", "B_LTK_N"): "free",
        ("B_LTK_N", "J5"): "N_left", ("B_LTK_N", "J1"): "free", ("B_LTK_N", "B_H319A"): "free",
        ("J1", "B_LTK_N"): "S_straight", ("J1", "J5"): "free", ("J1", "B_H319A"): "ban",
        ("B_H319A", "*"): "free"}},
    "J1": {"mode": "permissive", "rules": {
        ("J0", "*"): "N", ("J2", "*"): "S", ("B_DS2", "*"): "W"}},
    "J2": {"mode": "permissive", "rules": {
        ("J1", "B_LTK_S"): "N", ("J1", "J3"): "N", ("J1", "B_LG"): "free",
        ("B_LG", "J3"): "W", ("B_LG", "J1"): "W", ("B_LG", "B_LTK_S"): "free",
        ("B_LTK_S", "*"): "S",
        ("J3", "B_LG"): "E", ("J3", "B_LTK_S"): "E", ("J3", "J1"): "free"}},
    "J3": {"mode": "permissive", "rules": {
        ("J5", "*"): "N", ("B_TT_S", "*"): "S", ("J4", "*"): "E",
        ("J2", "B_TT_S"): "free", ("J2", "*"): "W"}},
    "J5": {"mode": "strict", "rules": {
        ("B_BH_E", "J0"): "E", ("B_BH_E", "J3"): "E",
        ("J0", "*"): "W",
        ("J3", "B_BH_E"): "S", ("J3", "J0"): "S"}},
}
# J5: motorcycles always go straight from Bắc Hải east and turn right from Thành Thái.
# Modelled with class-specific connections from the right lane (lane 0).
MOTO_FREE = {("J5", "B_BH_E", "J0"), ("J5", "J3", "B_BH_E")}
CONTROLLED = ["J0", "J1", "J2", "J3", "J5"]


# --------------------------------------------------------------------------- geometry

def street_graph(net, street):
    adj = {}
    for e in net.getEdges():
        if e.getName() != street:
            continue
        a, b = e.getFromNode().getID(), e.getToNode().getID()
        shape = e.getShape()
        adj.setdefault(a, []).append((b, e.getLength(), shape))
        adj.setdefault(b, []).append((a, e.getLength(), shape[::-1]))
    return adj


def nearest_nodes(net, adj, point, radius=90.0):
    out = []
    for n in adj:
        d = math.dist(net.getNode(n).getCoord(), point)
        if d <= radius:
            out.append((d, n))
    return [n for _, n in sorted(out)]


def _dijkstra(adj, sources):
    import heapq
    best = {s: 0.0 for s in sources}
    prev = {}
    heap = [(0.0, s) for s in sources]
    while heap:
        d, u = heapq.heappop(heap)
        if d > best.get(u, math.inf):
            continue
        for v, w, seg in adj.get(u, []):
            nd = d + w
            if nd < best.get(v, math.inf):
                best[v] = nd
                prev[v] = (u, seg)
                heapq.heappush(heap, (nd, v))
    return best, prev


def _shape_to(prev, u):
    shape = []
    while u in prev:
        u, seg = prev[u]
        shape = seg[1:] + shape if shape else list(seg)
    return shape


def dijkstra_path(adj, sources, targets):
    best, prev = _dijkstra(adj, sources)
    reached = [t for t in targets if t in best]
    if not reached:
        return None
    return _shape_to(prev, min(reached, key=lambda t: best[t]))


def farthest_path(adj, sources, key):
    """Path from the junction to the reachable street node that minimises `key`."""
    best, prev = _dijkstra(adj, sources)
    if not best:
        return []
    return _shape_to(prev, min(best, key=key))


def polyline_len(pts):
    return sum(math.dist(a, b) for a, b in zip(pts, pts[1:]))


def cut(pts, length):
    out, acc = [pts[0]], 0.0
    for a, b in zip(pts, pts[1:]):
        seg = math.dist(a, b)
        if acc + seg >= length:
            t = (length - acc) / seg if seg else 0
            out.append((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])))
            return out
        out.append(b)
        acc += seg
    return out


def extend(pts, length):
    total = polyline_len(pts)
    if total >= length or len(pts) < 2:
        return pts
    (x1, y1), (x2, y2) = pts[-2], pts[-1]
    d = math.dist(pts[-2], pts[-1]) or 1.0
    k = (length - total) / d
    return pts + [(x2 + k * (x2 - x1), y2 + k * (y2 - y1))]


def simplify(pts, tol=1.0):
    """Drop near-duplicate points."""
    out = [pts[0]]
    for p in pts[1:]:
        if math.dist(p, out[-1]) > tol:
            out.append(p)
    return out


def segment_shapes(net):
    centres = {j: c for j, (_n, c) in JUNCTIONS.items()}
    shapes, boundary_nodes = {}, {}
    for street, a, b, target in SEGMENTS:
        adj = street_graph(net, street)
        if a.startswith("J") and b.startswith("J"):
            path = dijkstra_path(adj, nearest_nodes(net, adj, centres[a]), nearest_nodes(net, adj, centres[b]))
            pts = [centres[a]] + (path or []) + [centres[b]]
        else:
            j, bnode = (b, a) if a.startswith("B") else (a, b)
            if isinstance(target, tuple):                      # synthetic straight arm
                _, jj, st = target
                other = next(s for s in shapes.values() if s["street"] == st and jj in (s["a"], s["b"]))
                p = other["pts"] if other["b"] == jj else other["pts"][::-1]
                (x1, y1), (x2, y2) = p[-2], p[-1]
                d = math.dist(p[-2], p[-1])
                pts = [centres[j], (x2 + 250 * (x2 - x1) / d, y2 + 250 * (y2 - y1) / d)]
            else:
                key = {"maxy": lambda n: -net.getNode(n).getCoord()[1],
                       "miny": lambda n: net.getNode(n).getCoord()[1],
                       "minx": lambda n: net.getNode(n).getCoord()[0],
                       "maxx": lambda n: -net.getNode(n).getCoord()[0]}[target]
                path = farthest_path(adj, nearest_nodes(net, adj, centres[j])[:1], key)
                pts = extend(cut([centres[j]] + path, ARM_CAP), ARM_MIN)
            if a.startswith("B"):
                pts = pts[::-1]
            boundary_nodes[bnode] = pts[0] if a.startswith("B") else pts[-1]
        shapes[(street, a, b)] = {"street": street, "a": a, "b": b, "pts": simplify(pts)}
    return shapes, boundary_nodes


# --------------------------------------------------------------------------- plain XML

def edge_id(street, a, b):
    return f"{ABBR[street]}_{a}_{b}"


def write_plain(tmp: Path, shapes, boundary_nodes, location: str):
    nod = ['<nodes>', f'    {location}']
    for j, (_name, (x, y)) in JUNCTIONS.items():
        nod.append(f'    <node id="{j}" x="{x:.2f}" y="{y:.2f}" type="traffic_light" tlType="static"/>')
    for b, (x, y) in boundary_nodes.items():
        nod.append(f'    <node id="{b}" x="{x:.2f}" y="{y:.2f}" type="priority"/>')
    nod.append('</nodes>')
    (tmp / "net.nod.xml").write_text("\n".join(nod), encoding="utf-8")

    edg = ['<edges>']
    for s in shapes.values():
        lanes, width, speed = STREETS[s["street"]]
        for a, b, pts in ((s["a"], s["b"], s["pts"]), (s["b"], s["a"], s["pts"][::-1])):
            shape = " ".join(f"{x:.2f},{y:.2f}" for x, y in pts)
            edg.append(f'    <edge id="{edge_id(s["street"], a, b)}" from="{a}" to="{b}" name="{s["street"]}" '
                       f'numLanes="{lanes}" width="{width}" speed="{speed}" disallow="pedestrian" '
                       f'shape="{shape}"/>')
    edg.append('</edges>')
    (tmp / "net.edg.xml").write_text("\n".join(edg), encoding="utf-8")


def arm_edges(shapes):
    """(junction, neighbour) -> (incoming edge id, outgoing edge id)."""
    out = {}
    for s in shapes.values():
        a, b, st = s["a"], s["b"], s["street"]
        out[(b, a)] = (edge_id(st, a, b), edge_id(st, b, a))
        out[(a, b)] = (edge_id(st, b, a), edge_id(st, a, b))
    return out


def write_connections(tmp: Path, shapes):
    arms = arm_edges(shapes)
    con = ['<connections>']
    for j, spec in SPEC.items():
        for (frm, to), rule in spec["rules"].items():
            if rule == "ban":
                con.append(f'    <delete from="{arms[(j, frm)][0]}" to="{arms[(j, to)][1]}"/>')
    # J5 class-specific connections: define every connection of the two approaches
    in_e, bhw_out, tts_out, bhe_out = (arms[("J5", "B_BH_E")][0], arms[("J5", "J0")][1],
                                       arms[("J5", "J3")][1], arms[("J5", "B_BH_E")][1])
    con += [
        f'    <connection from="{in_e}" to="{bhw_out}" fromLane="0" toLane="0" allow="motorcycle"/>',
        f'    <connection from="{in_e}" to="{bhw_out}" fromLane="1" toLane="1" disallow="motorcycle pedestrian"/>',
        f'    <connection from="{in_e}" to="{tts_out}" fromLane="1" toLane="1"/>',
    ]
    tt_in = arms[("J5", "J3")][0]
    con += [
        f'    <connection from="{tt_in}" to="{bhe_out}" fromLane="0" toLane="0" allow="motorcycle"/>',
        f'    <connection from="{tt_in}" to="{bhe_out}" fromLane="0" toLane="1" disallow="motorcycle pedestrian"/>',
        f'    <connection from="{tt_in}" to="{bhw_out}" fromLane="1" toLane="1"/>',
    ]
    con.append('</connections>')
    (tmp / "net.con.xml").write_text("\n".join(con), encoding="utf-8")


# --------------------------------------------------------------------------- signals

def edge_dir(net, edge, at_node):
    """Unit direction of an edge pointing away from the junction, from its first 30 m."""
    shape = edge.getShape()
    if edge.getToNode().getID() == at_node:
        shape = shape[::-1]
    p = cut(shape, 30.0)
    dx, dy = p[-1][0] - p[0][0], p[-1][1] - p[0][1]
    d = math.hypot(dx, dy) or 1.0
    return dx / d, dy / d


def build_programs(net, shapes):
    arms = arm_edges(shapes)
    in_arm, out_arm = {}, {}
    for (j, nb), (ein, eout) in arms.items():
        in_arm[ein] = nb
        out_arm[eout] = nb
    programs = {}
    for j in CONTROLLED:
        spec = SPEC[j]
        tls = net.getTLS(j)
        node = net.getNode(j)
        links = {}                    # li -> dict
        for in_lane, out_lane, li in tls.getConnections():
            conn = next(c for c in in_lane.getOutgoing() if c.getToLane() is out_lane)
            frm = in_arm[in_lane.getEdge().getID()]
            to = out_arm[out_lane.getEdge().getID()]
            rule = spec["rules"].get((frm, to), spec["rules"].get((frm, "*")))
            if (j, frm, to) in MOTO_FREE and in_lane.getIndex() == 0 and out_lane.getIndex() == 0:
                rule = "free"
            if rule is None or rule == "ban":
                rule = "free"
            links[li] = {"rule": rule, "from": frm, "to": to, "dir": conn.getDirection(),
                         "jidx": conn.getJunctionIndex()}
        n = max(links) + 1

        def foes(a, b):
            return node.areFoes(links[a]["jidx"], links[b]["jidx"])

        arm_vec = {nb: edge_dir(net, net.getEdge(arms[(j, nb)][1]), j)
                   for (jj, nb) in arms if jj == j}

        def opposite(a, b):
            va, vb = arm_vec[a], arm_vec[b]
            return va[0] * vb[0] + va[1] * vb[1] < -0.7          # > ~135 degrees apart

        def yield_ok(a, b):
            """Conflict allowed in permissive mode: left turn vs opposite approach."""
            la, lb = links[a], links[b]
            for x, y in ((la, lb), (lb, la)):
                if x["dir"] in ("l", "L") and opposite(x["from"], y["from"]):
                    return True
            return False

        groups = sorted({l["rule"] for l in links.values() if l["rule"] != "free"})
        members = {g: [li for li, l in links.items() if l["rule"] == g] for g in groups}

        def compatible(g1, g2):
            for a in members[g1]:
                for b in members[g2]:
                    if foes(a, b) and not (spec["mode"] == "permissive" and yield_ok(a, b)):
                        return False
            return True

        # maximal sets of mutually compatible signal groups
        cliques = []

        def bk(r, p, x):
            if not p and not x:
                cliques.append(sorted(r))
                return
            for v in list(p):
                bk(r | {v}, {u for u in p if u != v and compatible(u, v)},
                   {u for u in x if compatible(u, v)})
                p = p - {v}
                x = x | {v}

        bk(set(), set(groups), set())
        cliques.sort()

        phases = []
        for clique in cliques:
            green = [li for g in clique for li in members[g]]
            state = []
            for li in range(n):
                l = links[li]
                if l["rule"] == "free":
                    state.append("g" if any(foes(li, o) for o in green) else "G")
                elif li in green:
                    yields = any(foes(li, o) for o in green if o != li)
                    turning = l["dir"] in ("l", "L", "r", "R")
                    state.append("g" if yields and turning else "G")
                else:
                    state.append("r")
            phases.append("".join(state))
        programs[j] = (phases, cliques)
    return programs


def write_tll(tmp: Path, programs):
    out = ['<additional>']
    for j, (phases, _cliques) in programs.items():
        out.append(f'    <tlLogic id="{j}" type="static" programID="0" offset="0">')
        for k, st in enumerate(phases):
            nxt = phases[(k + 1) % len(phases)]
            yellow = "".join("y" if c in "Gg" and d not in "Gg" else c for c, d in zip(st, nxt))
            out.append(f'        <phase duration="{GREEN_TIME}" state="{st}"/>')
            out.append(f'        <phase duration="{YELLOW_TIME}" state="{yellow}"/>')
        out.append('    </tlLogic>')
    out.append('</additional>')
    (tmp / "net.tll.xml").write_text("\n".join(out), encoding="utf-8")


# --------------------------------------------------------------------------- main

def run(cmd):
    print(">", " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], check=True)


def main():
    osm = sumolib.net.readNet(str(OSM_NET))
    location = next(line.strip() for line in sumolib.openz(str(OSM_NET)) if "<location" in line)
    shapes, boundary_nodes = segment_shapes(osm)
    netconvert = sumolib.checkBinary("netconvert")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        write_plain(tmp, shapes, boundary_nodes, location)
        write_connections(tmp, shapes)
        stage1 = tmp / "stage1.net.xml"
        common = ["--offset.disable-normalization", "true", "--no-turnarounds", "true", "--tls.default-type", "static",
                  "--tls.yellow.time", str(YELLOW_TIME), "--junctions.corner-detail", "5",
                  "--output.street-names", "true", "--no-warnings", "true"]
        run([netconvert, "-n", tmp / "net.nod.xml", "-e", tmp / "net.edg.xml", "-x", tmp / "net.con.xml",
             "-o", stage1, *common])
        net = sumolib.net.readNet(str(stage1), withPrograms=True)
        programs = build_programs(net, shapes)
        write_tll(tmp, programs)
        run([netconvert, "-s", stage1, "-i", tmp / "net.tll.xml", "-o", OUT_NET, *common])
    print(f"\nwrote {OUT_NET}")
    for j, (phases, cliques) in programs.items():
        print(f"  {j} {JUNCTIONS[j][0]}: {len(phases)} phases {cliques}")


if __name__ == "__main__":
    main()
