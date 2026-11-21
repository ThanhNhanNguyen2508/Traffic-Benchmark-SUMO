"""Turn the SUMO network into junction-level agents.

OSM imports often split one physical intersection into several traffic lights
(tlLogic) a few metres apart. Here every group of nearby traffic lights is one
junction, and one junction is one agent. For each junction we derive:

* movements -- (entry edge, exit edge) pairs through the whole junction, the SUMO
               counterpart of MPLight's WL/WT/EL/... movements;
* phases    -- coordinated joint signal states over all lights of the junction.
               The original per-light plans in this map were generated
               independently (all cycle 90 s, offset 0), so some movements are
               never green along their whole path. Phases are therefore built
               as maximal sets of mutually compatible movements, using SUMO's
               conflict matrix (foes) plus the green combinations the original
               plans already allow (permissive turns);
* zones     -- upstream/downstream edge chains used to count vehicles, because
               many boundary edges in this map are only 0.2-5 m long.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sumolib

GREEN = set("Gg")
VEHICLE_CLASSES = ("passenger", "motorcycle", "bus", "truck")


def is_vehicle_edge(edge) -> bool:
    return any(edge.allows(c) for c in VEHICLE_CLASSES)


def _is_reverse(a, b) -> bool:
    return a.getFromNode() is b.getToNode() and a.getToNode() is b.getFromNode()


@dataclass
class Movement:
    entry: str            # edge where the movement starts (junction entry, or an internal
    exit: str             # edge when a path must cross the same junction twice)
    links: tuple          # TLS links (tls_id, link_index) that must be green


@dataclass
class Junction:
    id: str
    name: str
    tls: list
    nodes: set
    entries: list
    exits: list
    internal_edges: set
    movements: list                   # controllable movements only (state of the agent)
    phases: list                      # list of {tls_id: state string}
    phase_movement: np.ndarray        # (P, M) 0/1
    relation: np.ndarray              # (P, P) 1 if two phases share a movement
    entry_zone: dict                  # entry edge -> [edges upstream, entry first]
    exit_zone: dict                   # exit edge  -> [edges downstream, exit first]
    always_green: list                # movements green in every phase (not in the state)
    starts: set = None                # movement start edges (entries + internal split edges)

    @property
    def num_phases(self) -> int:
        return len(self.phases)

    @property
    def num_movements(self) -> int:
        return len(self.movements)

    def summary(self) -> str:
        lines = [f"{self.id} [{self.name}] lights={len(self.tls)} nodes={len(self.nodes)} "
                 f"movements={self.num_movements} (+{len(self.always_green)} always green) "
                 f"phases={self.num_phases}"]
        for p in range(self.num_phases):
            greens = [f"{m.entry}->{m.exit}" for m, g in zip(self.movements, self.phase_movement[p]) if g]
            lines.append(f"  phase {p}: {', '.join(greens)}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- clustering

def _tls_nodes(net, tls_id):
    nodes = set()
    for in_lane, _out_lane, _li in net.getTLS(tls_id).getConnections():
        nodes.add(in_lane.getEdge().getToNode().getID())
    return nodes


def cluster_traffic_lights(net, max_dist: float):
    """Single-linkage clustering of traffic lights by the centroid of their nodes."""
    ids = [t.getID() for t in net.getTrafficLights()]
    cent = {}
    for t in ids:
        pts = [net.getNode(n).getCoord() for n in _tls_nodes(net, t)]
        cent[t] = (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))
    parent = {t: t for t in ids}

    def root(t):
        while parent[t] != t:
            parent[t] = parent[parent[t]]
            t = parent[t]
        return t

    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            if math.dist(cent[a], cent[b]) < max_dist:
                parent[root(a)] = root(b)
    groups = defaultdict(list)
    for t in ids:
        groups[root(t)].append(t)
    out = []
    for members in groups.values():
        cx = sum(cent[t][0] for t in members) / len(members)
        cy = sum(cent[t][1] for t in members) / len(members)
        out.append((cx, cy, sorted(members)))
    out.sort(key=lambda g: (round(g[0], -1), g[1]))
    return [g[2] for g in out]


def _junction_name(net, tls_ids):
    nodes = set().union(*(_tls_nodes(net, t) for t in tls_ids))
    names = Counter()
    for n in nodes:
        for e in net.getNode(n).getIncoming():
            if is_vehicle_edge(e) and e.getName():
                names[e.getName()] += 1
    top = [n for n, _ in names.most_common(2)]
    return " / ".join(top) if top else "unnamed"


def load_or_create_junction_file(path: Path, net, max_dist: float):
    """Junction grouping lives in an editable JSON file (created on first run)."""
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        return data["junctions"], False
    groups = cluster_traffic_lights(net, max_dist)
    junctions = [
        {"id": f"J{i}", "name": _junction_name(net, g), "enabled": True, "tls": g}
        for i, g in enumerate(groups)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "_comment": "One entry = one agent. Edit 'tls' to regroup lights; set enabled=false to "
                    "leave a junction on its original fixed-time plan.",
        "cluster_dist_m": max_dist,
        "junctions": junctions,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return junctions, True


def _add_glue_nodes(net, nodes: set, other_nodes: set, glue_len: float) -> set:
    """Add unsignalised nodes that sit between two parts of the same junction.

    A node m outside the junction is glue if a short edge leaves the junction to m and
    another short edge (not the reverse one) goes from m back into the junction.
    """
    nodes = set(nodes)
    changed = True
    while changed:
        changed = False
        for n in list(nodes):
            for e in net.getNode(n).getOutgoing():
                m = e.getToNode()
                mid = m.getID()
                if (mid in nodes or mid in other_nodes or not is_vehicle_edge(e)
                        or e.getLength() > glue_len):
                    continue
                back = [o for o in m.getOutgoing()
                        if is_vehicle_edge(o) and o.getToNode().getID() in nodes
                        and o.getLength() <= glue_len and not _is_reverse(o, e)]
                if back:
                    nodes.add(mid)
                    changed = True
    return nodes


# --------------------------------------------------------------------------- zones

def _extend_zone(start_edge, stop_nodes: set, length: float, upstream: bool):
    zone = [start_edge]
    total = start_edge.getLength()
    cur = start_edge
    while total < length:
        node = cur.getFromNode() if upstream else cur.getToNode()
        if node.getID() in stop_nodes:
            break
        pool = node.getIncoming() if upstream else node.getOutgoing()
        cands = [e for e in pool if is_vehicle_edge(e) and not _is_reverse(e, cur) and e not in zone]
        if len(cands) != 1:
            same = [e for e in cands if e.getName() and e.getName() == cur.getName()]
            if len(same) != 1:
                break
            cands = same
        cur = cands[0]
        zone.append(cur)
        total += cur.getLength()
    return [e.getID() for e in zone]


# --------------------------------------------------------------------------- movements

def _trace_paths(junction_tls: set, junction_nodes: set, entry_edge, max_depth: int = 10):
    """All lane paths from an entry edge to the junction's exit edges.

    Returns {exit_edge_id: set of tuples of (from_edge_id, tls_id, link_index)}.
    """
    found = defaultdict(set)
    stack = [(entry_edge, (), (entry_edge.getID(),))]
    while stack:
        edge, links, visited = stack.pop()
        for to_edge, conns in edge.getOutgoing().items():
            if not is_vehicle_edge(to_edge) or to_edge.getID() in visited:
                continue
            for c in conns:
                lane = c.getFromLane()
                if not any(lane.allows(v) for v in VEHICLE_CLASSES):
                    continue
                tls = c.getTLSID()
                step = links
                if tls:
                    if tls not in junction_tls:
                        continue
                    step = links + ((edge.getID(), tls, c.getTLLinkIndex()),)
                inside = to_edge.getToNode().getID() in junction_nodes
                if inside and len(visited) < max_depth:
                    stack.append((to_edge, step, visited + (to_edge.getID(),)))
                elif not inside and not _is_reverse(to_edge, entry_edge):
                    found[to_edge.getID()].add(step)
    return found


# --------------------------------------------------------------------------- conflicts

class LinkConflicts:
    """Which pairs of TLS links may be green together inside one junction."""

    def __init__(self, net, tls_ids):
        self.conns = defaultdict(list)       # (tls, li) -> [(node, junction_index)]
        self.states = {}                     # tls -> list of original state strings
        self.num_links = {}
        for t in tls_ids:
            tls = net.getTLS(t)
            prog = tls.getPrograms()[sorted(tls.getPrograms())[0]]
            self.states[t] = [ph.state for ph in prog._phases]
            self.num_links[t] = len(self.states[t][0])
            for in_lane, out_lane, li in tls.getConnections():
                for c in in_lane.getOutgoing():
                    if c.getToLane() is out_lane and c.getTLLinkIndex() == li:
                        self.conns[(t, li)].append((c.getJunction(), c.getJunctionIndex()))
        self._cache = {}

    def foes(self, a, b) -> bool:
        for na, ia in self.conns.get(a, []):
            for nb, ib in self.conns.get(b, []):
                if na is nb and ia >= 0 and ib >= 0 and na.areFoes(ia, ib):
                    return True
        return False

    def cooccur_char(self, a, b):
        """Signal char of link a in an original state where a and b are both green."""
        for s in self.states[a[0]]:
            if s[a[1]] in GREEN and s[b[1]] in GREEN:
                return s[a[1]]
        return None

    def compatible(self, a, b) -> bool:
        if a == b or a[0] != b[0]:
            return True               # different lights control different nodes
        key = (a, b) if a <= b else (b, a)
        if key not in self._cache:
            if not self.conns.get(a) or not self.conns.get(b):
                # e.g. pedestrian crossings: no conflict data, trust only the original plan
                self._cache[key] = self.cooccur_char(a, b) is not None
            else:
                self._cache[key] = (not self.foes(a, b)) or self.cooccur_char(a, b) is not None
        return self._cache[key]

    def all_compatible(self, links_a, links_b) -> bool:
        return all(self.compatible(a, b) for a in links_a for b in links_b)


def _split_path(entry, exit_edge, path, conflicts: LinkConflicts):
    """Cut a lane path where a link conflicts with an earlier link of the same path.

    Returns [(start_edge, end_edge, links), ...]; the cut edge is the end of one
    segment and the start of the next (vehicles wait there between the two stages).
    """
    out, start, cur = [], entry, []
    for frm, t, li in path:
        link = (t, li)
        if cur and not all(conflicts.compatible(link, c) for c in cur):
            out.append((start, frm, tuple(cur)))
            start, cur = frm, []
        cur.append(link)
    out.append((start, exit_edge, tuple(cur)))
    return out


def _maximal_cliques(n, adj):
    """Bron-Kerbosch with pivoting; adj is a list of sets."""
    out = []

    def bk(r, p, x):
        if not p and not x:
            out.append(sorted(r))
            return
        pivot = max(p | x, key=lambda u: len(adj[u] & p))
        for v in list(p - adj[pivot]):
            bk(r | {v}, p & adj[v], x & adj[v])
            p = p - {v}
            x = x | {v}

    bk(set(), set(range(n)), set())
    return sorted(out)


def _phase_states(conflicts: LinkConflicts, tls_ids, green_links: set):
    """Signal strings for a set of green links; other links get green if compatible."""
    green = set(green_links)
    for t in tls_ids:                       # fill: e.g. pedestrian crossings, U-turn links
        for li in range(conflicts.num_links[t]):
            link = (t, li)
            if link not in green and all(conflicts.compatible(link, g) for g in green if g[0] == t):
                green.add(link)
    states = {}
    for t in tls_ids:
        chars = []
        for li in range(conflicts.num_links[t]):
            link = (t, li)
            if link not in green:
                chars.append("r")
                continue
            char = "G"
            for g in green:
                if g[0] == t and g != link and conflicts.foes(link, g):
                    if conflicts.cooccur_char(link, g) == "g":
                        char = "g"
            chars.append(char)
        states[t] = "".join(chars)
    return states


# --------------------------------------------------------------------------- plan stages

def _plan_stages(net, tls_ids, min_run: int):
    """Joint states of the original plans, sampled every second over one common cycle.

    Keeps states without yellow that last at least `min_run` seconds and are not
    dominated (green links a strict subset of another state's green links).
    """
    progs = {}
    for t in tls_ids:
        tls = net.getTLS(t)
        progs[t] = tls.getPrograms()[sorted(tls.getPrograms())[0]]
    cycles = {t: max(1, int(round(sum(ph.duration for ph in p._phases)))) for t, p in progs.items()}
    horizon = 1
    for c in cycles.values():
        horizon = min(3600, horizon * c // math.gcd(horizon, c))
    seq = []
    for sec in range(horizon):
        joint = []
        for t in tls_ids:
            pos, acc = (sec - float(progs[t]._offset)) % cycles[t], 0.0
            state = progs[t]._phases[-1].state
            for ph in progs[t]._phases:
                acc += ph.duration
                if pos < acc:
                    state = ph.state
                    break
            joint.append(state)
        seq.append(tuple(joint))
    run, prev, best = 0, None, Counter()
    for key in seq + seq:
        run = run + 1 if key == prev else 1
        prev = key
        best[key] = max(best[key], min(run, len(seq)))
    keys = [k for k, r in best.items() if r >= min_run and not any(c in "yYu" for st in k for c in st)]
    green = {k: {(t, i) for t, st in zip(tls_ids, k) for i, c in enumerate(st) if c in GREEN} for k in keys}
    keys = [k for k in keys if not any(green[k] < green[o] for o in keys if o != k)]
    order = {k: seq.index(k) for k in keys}
    keys.sort(key=lambda k: order[k])
    return [dict(zip(tls_ids, k)) for k in keys], [best[k] for k in keys]


# --------------------------------------------------------------------------- build

def build_junction(net, spec: dict, nodes: set, all_cluster_nodes: set, conf: dict) -> Junction | None:
    tls_ids = list(spec["tls"])
    tls_set = set(tls_ids)
    conflicts = LinkConflicts(net, tls_ids)

    entries, exits, internal = [], [], set()
    for n in nodes:
        node = net.getNode(n)
        for e in node.getIncoming():
            if is_vehicle_edge(e) and e.getFromNode().getID() not in nodes:
                entries.append(e)
        for e in node.getOutgoing():
            if not is_vehicle_edge(e):
                continue
            if e.getToNode().getID() in nodes:
                internal.add(e.getID())
            else:
                exits.append(e)

    # movements: split paths that cross conflicting links of the same junction twice,
    # then keep, per (start, end), the lane paths that can be green together
    segments = defaultdict(set)
    for entry in sorted(entries, key=lambda e: e.getID()):
        for exit_id, paths in _trace_paths(tls_set, nodes, entry).items():
            for path in paths:
                for seg in _split_path(entry.getID(), exit_id, path, conflicts):
                    segments[seg[:2]].add(seg[2])
    all_movs = []
    for (start, end), paths in sorted(segments.items()):
        chosen = set()
        for path in sorted(paths, key=lambda p: (len(p), p)):
            if conflicts.all_compatible(path, chosen):
                chosen |= set(path)
        if chosen:
            all_movs.append(Movement(start, end, tuple(sorted(chosen))))

    if conf.get("PHASE_MODE", "plan") == "plan":
        return _build_from_plan(net, spec, nodes, all_cluster_nodes, conf, conflicts,
                                tls_ids, entries, exits, internal)

    n = len(all_movs)
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if conflicts.all_compatible(all_movs[i].links, all_movs[j].links):
                adj[i].add(j)
                adj[j].add(i)
    cliques = _maximal_cliques(n, adj) if n else []
    if len(cliques) < 2:
        return None

    phases, masks = [], []
    for clique in cliques:
        green_links = set().union(*(all_movs[k].links for k in clique))
        phases.append(_phase_states(conflicts, tls_ids, green_links))
        masks.append([1 if k in clique else 0 for k in range(n)])
    full = np.array(masks, dtype=np.float32)
    controllable = [k for k in range(n) if 0 < full[:, k].sum() < full.shape[0]]
    always = [all_movs[k] for k in range(n) if full[:, k].sum() == full.shape[0]]
    movements = [all_movs[k] for k in controllable]
    phase_movement = full[:, controllable]
    relation = ((phase_movement @ phase_movement.T) > 0).astype(np.int64)

    starts = {m.entry for m in movements} | {m.entry for m in always}
    used_entries = sorted({m.entry for m in movements} - internal)
    used_exits = sorted({m.exit for m in movements})
    by_id = {e.getID(): e for e in entries + exits}
    zone_len = conf["ZONE_LENGTH"]
    entry_zone = {e: _extend_zone(by_id[e], all_cluster_nodes, zone_len, upstream=True) for e in used_entries}
    exit_zone = {e: ([e] if e in internal else
                     _extend_zone(by_id[e], all_cluster_nodes, zone_len, upstream=False))
                 for e in used_exits}

    return Junction(
        id=spec["id"], name=spec.get("name", spec["id"]), tls=tls_ids, nodes=nodes,
        entries=used_entries, exits=used_exits, internal_edges=internal,
        movements=movements, phases=phases, phase_movement=phase_movement,
        relation=relation, entry_zone=entry_zone, exit_zone=exit_zone, always_green=always,
        starts=starts,
    )


def _build_from_plan(net, spec, nodes, all_cluster_nodes, conf, conflicts, tls_ids,
                     entries, exits, internal):
    """Phases = the stages of the original plans (what the fixed-time baseline runs).

    A movement (entry -> exit) belongs to a phase when the first signal on its path
    whose colour changes between stages is green: that is where the movement is
    released. Later signals inside the junction keep their original timing.
    """
    stages, runs = _plan_stages(net, tls_ids, conf["MIN_PHASE_RUN"])
    if len(stages) < 2:
        return None
    varying = {(t, i) for t in tls_ids for i in range(conflicts.num_links[t])
               if len({st[t][i] in GREEN for st in stages}) > 1}
    tls_set = set(tls_ids)
    movs, masks = [], []
    for entry in sorted(entries, key=lambda e: e.getID()):
        for exit_id, paths in sorted(_trace_paths(tls_set, nodes, entry).items()):
            row = []
            for st in stages:
                served = False
                for path in paths:
                    first = next(((t, i) for _f, t, i in path if (t, i) in varying), None)
                    if first is not None and st[first[0]][first[1]] in GREEN:
                        served = True
                row.append(1 if served else 0)
            links = tuple(sorted({(t, i) for path in paths for _f, t, i in path}))
            movs.append(Movement(entry.getID(), exit_id, links))
            masks.append(row)
    full = np.array(masks, dtype=np.float32).T                     # (P, M_all)
    controllable = [k for k in range(full.shape[1]) if 0 < full[:, k].sum() < full.shape[0]]
    if not controllable:
        return None
    always = [movs[k] for k in range(full.shape[1]) if full[:, k].sum() == full.shape[0]]
    movements = [movs[k] for k in controllable]
    phase_movement = full[:, controllable]
    # drop stages that serve exactly the same movements (keep the longest one)
    keep, seen = [], {}
    for p in range(len(stages)):
        key = tuple(phase_movement[p])
        if key in seen:
            if runs[p] > runs[seen[key]]:
                keep[keep.index(seen[key])] = p
                seen[key] = p
            continue
        seen[key] = p
        keep.append(p)
    if len(keep) < 2:
        return None
    stages = [stages[p] for p in keep]
    phase_movement = phase_movement[keep]
    relation = ((phase_movement @ phase_movement.T) > 0).astype(np.int64)
    used_entries = sorted({m.entry for m in movements})
    used_exits = sorted({m.exit for m in movements})
    by_id = {e.getID(): e for e in entries + exits}
    zone_len = conf["ZONE_LENGTH"]
    return Junction(
        id=spec["id"], name=spec.get("name", spec["id"]), tls=tls_ids, nodes=nodes,
        entries=used_entries, exits=used_exits, internal_edges=internal,
        movements=movements, phases=stages, phase_movement=phase_movement, relation=relation,
        entry_zone={e: _extend_zone(by_id[e], all_cluster_nodes, zone_len, True) for e in used_entries},
        exit_zone={e: _extend_zone(by_id[e], all_cluster_nodes, zone_len, False) for e in used_exits},
        always_green=always, starts=set(used_entries) | {m.entry for m in always},
    )


def build_junctions(net_file: Path, junction_file: Path, conf: dict, log=print):
    net = sumolib.net.readNet(str(net_file), withPrograms=True, withInternal=False)
    specs, created = load_or_create_junction_file(junction_file, net, conf["CLUSTER_DIST"])
    if created:
        log(f"Created junction grouping: {junction_file}")
    tls_nodes = {s["id"]: set().union(*(_tls_nodes(net, t) for t in s["tls"])) for s in specs}
    all_tls_nodes = set().union(*tls_nodes.values())
    full_nodes = {
        sid: _add_glue_nodes(net, n, all_tls_nodes - n, conf["GLUE_LENGTH"])
        for sid, n in tls_nodes.items()
    }
    all_nodes = set().union(*full_nodes.values())
    junctions, skipped = [], []
    for s in specs:
        if not s.get("enabled", True):
            skipped.append((s["id"], "disabled"))
            continue
        j = build_junction(net, s, full_nodes[s["id"]], all_nodes, conf)
        if j is None:
            skipped.append((s["id"], "fewer than 2 usable phases"))
        else:
            junctions.append(j)
    return junctions, skipped
