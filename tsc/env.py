"""Multi-junction SUMO environment with MPLight's interaction protocol.

* One decision every MIN_ACTION_TIME seconds per junction ("set" action pattern:
  the action is the index of the phase to show).
* Changing phase inserts YELLOW_TIME seconds of yellow (then ALL_RED_TIME of red)
  inside that interval, exactly as in MPLight's anon_env.
* Reward = REWARD_PRESSURE_WEIGHT * |sum of halting vehicles on entering zones -
  sum on exiting zones|, averaged over the MEASURE_TIME seconds after the action
  (MPLight: DIC_REWARD_INFO["pressure"] = -0.25 on waiting-vehicle counts).
* Observation per junction: current phase + pressure of every movement
  (vehicles heading into the movement minus vehicles on its exit zone).
  CityFlow has one lane per movement; in SUMO lanes are shared, so vehicles on
  the approach are assigned to a movement by their route.
"""

from __future__ import annotations

import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

# Allow `import traci` / `import libsumo` from a SUMO installation without pip packages.
if "SUMO_HOME" in os.environ:
    _tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    if _tools not in sys.path:
        sys.path.append(_tools)

import sumolib  # noqa: E402

SKIP_SECTIONS = {"output", "report", "gui_only", "input"}


def _sumocfg_args(cfg: Path):
    """Read net/route files and processing options from a .sumocfg (outputs are dropped)."""
    root = ET.parse(cfg).getroot()
    base = cfg.parent
    net = routes = None
    extra = []
    for section in root:
        tag = section.tag.split("}")[-1]
        for opt in section:
            key = opt.tag.split("}")[-1]
            val = opt.get("value")
            if tag == "input":
                if key == "net-file":
                    net = str((base / val).resolve())
                elif key == "route-files":
                    routes = ",".join(str((base / v.strip()).resolve()) for v in val.split(","))
            elif tag not in SKIP_SECTIONS and val is not None:
                extra += [f"--{key}", val]
    if net is None or routes is None:
        raise ValueError(f"{cfg}: net-file and route-files are required")
    return net, routes, extra


class SumoTSCEnv:
    def __init__(self, sumocfg: Path, junctions, env_conf: dict, seed: int = 0,
                 gui: bool = False, use_libsumo: bool = True, control: bool = True,
                 sumo_warnings: bool = False, extra_sumo_args=()):
        self.cfg = Path(sumocfg)
        self.junctions = junctions
        self.conf = env_conf
        self.seed = seed
        self.gui = gui
        self.control = control
        self.sumo_warnings = sumo_warnings
        self.extra_sumo_args = list(extra_sumo_args)
        if env_conf["YELLOW_TIME"] + env_conf["ALL_RED_TIME"] >= env_conf["MIN_ACTION_TIME"]:
            raise ValueError("MIN_ACTION_TIME must be longer than YELLOW_TIME + ALL_RED_TIME")

        self.tc = None
        if use_libsumo and not gui:
            try:
                import libsumo as tc  # noqa: F401
                self.tc = tc
            except ImportError:
                self.tc = None
        if self.tc is None:
            import traci as tc
            self.tc = tc
        self.backend = self.tc.__name__

        self.net_file, self.route_files, self.cfg_args = _sumocfg_args(self.cfg)
        self._prepare_junctions()
        self._running = False

    # ------------------------------------------------------------------ setup
    def _prepare_junctions(self):
        self.edges_in, self.edges_out = {}, {}
        self.all_count_edges = set()
        for j in self.junctions:
            j.mov_index = {(m.entry, m.exit): k for k, m in enumerate(j.movements)}
            j.mov_exit = [m.exit for m in j.movements]
            j.lookahead = max((len(z) for z in j.entry_zone.values()), default=0) + 1
            ins = sorted({e for z in j.entry_zone.values() for e in z})
            outs = sorted({e for z in j.exit_zone.values() for e in z})
            j.inside = sorted(j.internal_edges)
            # vehicles waiting between the lights of one junction count as entering traffic
            self.edges_in[j.id], self.edges_out[j.id] = sorted(set(ins) | set(j.inside)), outs
            self.all_count_edges |= set(ins) | set(outs) | set(j.inside)
        self.all_count_edges = sorted(self.all_count_edges)

    def _sumo_cmd(self, seed):
        binary = sumolib.checkBinary("sumo-gui" if self.gui else "sumo")
        cmd = [binary, "-n", self.net_file, "-r", self.route_files,
               "--seed", str(seed), "--no-step-log", "true", "--duration-log.disable", "true",
               "--no-warnings", "false" if self.sumo_warnings else "true"]
        cmd += self.cfg_args
        if self.gui:
            cmd += ["--start", "true", "--quit-on-end", "true"]
        return cmd + self.extra_sumo_args

    # ------------------------------------------------------------------ episode
    def reset(self, seed: int | None = None):
        self.close()
        self.tc.start(self._sumo_cmd(self.seed if seed is None else seed))
        self._running = True
        self.time = 0
        self.depart, self.arrive = {}, {}
        self.teleports = 0
        self.queue_sum = 0.0
        self.route_cache = {}
        self.cur_phase = {j.id: 0 for j in self.junctions}
        self.cur_state = {}
        if self.control:
            for j in self.junctions:
                self._apply(j, j.phases[0])
        return self._observe()

    def close(self):
        if self._running:
            try:
                self.tc.close()
            except Exception:  # noqa: BLE001
                pass
            self._running = False

    def _apply(self, j, joint_state: dict):
        for t, s in joint_state.items():
            self.tc.trafficlight.setRedYellowGreenState(t, s)
            self.cur_state[t] = s

    @staticmethod
    def _transition(cur: str, nxt: str, char: str) -> str:
        """Links green now but not next turn `char` (y or r); others keep their state."""
        return "".join(char if c in "Gg" and n not in "Gg" else c for c, n in zip(cur, nxt))

    def step(self, actions: dict | None):
        """Advance MIN_ACTION_TIME seconds. `actions` maps junction id -> phase index."""
        c = self.conf
        switching = []
        if self.control and actions is not None:
            for j in self.junctions:
                a = int(actions[j.id])
                if a != self.cur_phase[j.id]:
                    target = j.phases[a]
                    yellow = {t: self._transition(self.cur_state[t], target[t], "y") for t in j.tls}
                    self._apply(j, yellow)
                    switching.append((j, a))
        reward_acc = {j.id: 0.0 for j in self.junctions}
        for sec in range(c["MIN_ACTION_TIME"]):
            if switching and sec == c["YELLOW_TIME"]:
                for j, a in switching:
                    if c["ALL_RED_TIME"] > 0:
                        red = {t: self._transition(self.cur_state[t], j.phases[a][t], "r") for t in j.tls}
                        self._apply(j, red)
                    else:
                        self._apply(j, j.phases[a])
                        self.cur_phase[j.id] = a
            if switching and c["ALL_RED_TIME"] > 0 and sec == c["YELLOW_TIME"] + c["ALL_RED_TIME"]:
                for j, a in switching:
                    self._apply(j, j.phases[a])
                    self.cur_phase[j.id] = a
            self.tc.simulationStep()
            self.time += 1
            self._record_step()
            halt = {e: self.tc.edge.getLastStepHaltingNumber(e) for e in self.all_count_edges}
            for j in self.junctions:
                q_in = sum(halt[e] for e in self.edges_in[j.id])
                q_out = sum(halt[e] for e in self.edges_out[j.id])
                reward_acc[j.id] += c["REWARD_PRESSURE_WEIGHT"] * abs(q_in - q_out)
                self.queue_sum += q_in
        n = c["MIN_ACTION_TIME"]
        rewards = {k: v / n for k, v in reward_acc.items()}
        done = self.time >= c["RUN_COUNTS"]
        return self._observe(), rewards, done

    # ------------------------------------------------------------------ measurements
    def _record_step(self):
        sim = self.tc.simulation
        for vid in sim.getDepartedIDList():
            self.depart[vid] = self.time
        for vid in sim.getArrivedIDList():
            self.arrive[vid] = self.time
            self.route_cache.pop(vid, None)
        self.teleports += sim.getStartingTeleportNumber()

    def _route(self, vid, edge):
        veh = self.tc.vehicle
        idx = veh.getRouteIndex(vid)
        route = self.route_cache.get(vid)
        if route is None or idx >= len(route) or route[idx] != edge:
            route = tuple(veh.getRoute(vid))
            self.route_cache[vid] = route
        return route, idx

    def _movement_of(self, j, vid, edge, inside: bool):
        """Movement index of a vehicle, from its route.

        Outside the junction: the first movement start ahead (a junction entry).
        Inside (between the junction's lights): the closest start at or behind it.
        """
        route, idx = self._route(vid, edge)
        if inside:
            starts = range(idx, max(-1, idx - 12), -1)
        else:
            starts = range(idx, min(len(route), idx + j.lookahead))
        for k in starts:
            if route[k] not in j.starts:
                continue
            for k2 in range(max(k, idx) + 1, min(len(route), k + 16)):
                hit = j.mov_index.get((route[k], route[k2]))
                if hit is not None:
                    return hit
                if route[k2] not in j.internal_edges:
                    break
            if not inside:
                return None
        return None

    def _observe(self):
        edge = self.tc.edge
        counts = {e: edge.getLastStepVehicleNumber(e) for e in self.all_count_edges}
        obs = {}
        for j in self.junctions:
            x_in = np.zeros(j.num_movements, dtype=np.float32)
            for e in self.edges_in[j.id]:
                if counts[e] == 0:
                    continue
                inside = e in j.internal_edges
                for vid in edge.getLastStepVehicleIDs(e):
                    k = self._movement_of(j, vid, e, inside)
                    if k is not None:
                        x_in[k] += 1
            x_out = {ex: sum(counts[e] for e in zone) for ex, zone in j.exit_zone.items()}
            pressure = x_in - np.array([x_out[ex] for ex in j.mov_exit], dtype=np.float32)
            obs[j.id] = {"pressure": pressure, "phase": self.cur_phase[j.id]}
        return obs

    def metrics(self) -> dict:
        end = self.time
        tts = [self.arrive.get(v, end) - d for v, d in self.depart.items()]
        done_tts = [self.arrive[v] - d for v, d in self.depart.items() if v in self.arrive]
        try:
            pending = len(self.tc.simulation.getPendingVehicles())
        except Exception:  # noqa: BLE001
            pending = -1
        return {
            "travel_time": float(np.mean(tts)) if tts else 0.0,          # CityFlow-style ATT
            "travel_time_arrived": float(np.mean(done_tts)) if done_tts else 0.0,
            "departed": len(self.depart),
            "arrived": len(self.arrive),
            "in_network_at_end": len(self.depart) - len(self.arrive),
            "pending_insertion": pending,
            "teleports": self.teleports,
            "avg_queue": self.queue_sum / max(end, 1),
        }
