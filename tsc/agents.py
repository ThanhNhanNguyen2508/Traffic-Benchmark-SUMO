"""Controllers: MPLight (shared FRAP + DQN), MaxPressure, and the original fixed-time plan."""

from __future__ import annotations

import copy
import random
from pathlib import Path

import numpy as np
import torch

from .frap import FRAP


class FixedTimeAgent:
    """Runs the network's own signal programs (no control). Use with env control=False."""

    name = "fixedtime"

    def act(self, obs, explore=False):
        return None


class MaxPressureAgent:
    """Max-pressure control (Varaiya, 2013), as the MPLight baseline: every
    MIN_ACTION_TIME seconds choose the phase with the largest sum of movement
    pressures. Ties keep the current phase."""

    name = "maxpressure"

    def __init__(self, junctions):
        self.junctions = junctions

    def act(self, obs, explore=False):
        actions = {}
        for j in self.junctions:
            o = obs[j.id]
            scores = j.phase_movement @ o["pressure"]
            best = np.flatnonzero(scores == scores.max())
            actions[j.id] = int(o["phase"]) if o["phase"] in best else int(best[0])
        return actions


class MPLightAgent:
    """MPLight = FRAP Q-network shared by all junctions, pressure state and reward.

    Training follows the official round-based pipeline (pipeline.py / updater.py /
    network_agent.py):
      * one model and one memory for all junctions (TransferDQNPressOne);
      * per round: forget beyond MAX_MEMORY_LEN, sample SAMPLE_SIZE, build targets
        y[a] = r / NORMAL_FACTOR + GAMMA * max_a' Q_bar(s', a') with Q_bar = the model
        saved UPDATE_Q_BAR_FREQ rounds earlier, then fit up to EPOCHS epochs
        (batch BATCH_SIZE, validation_split 0.3, no shuffling, early stopping
        PATIENCE on val_loss);
      * epsilon-greedy with epsilon = max(EPSILON * EPSILON_DECAY**round, MIN_EPSILON).
    """

    name = "mplight"

    def __init__(self, junctions, agent_conf: dict, seed: int = 0, device: str = "cpu"):
        self.junctions = junctions
        self.conf = dict(agent_conf)
        self.device = torch.device(device)
        self.rng = random.Random(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.jidx = {j.id: i for i, j in enumerate(junctions)}
        self.P = max(j.num_phases for j in junctions)
        self.M = max(j.num_movements for j in junctions)
        nj = len(junctions)
        pm = np.zeros((nj, self.P, self.M), np.float32)
        pmask = np.zeros((nj, self.P), np.float32)
        mmask = np.zeros((nj, self.M), np.float32)
        rel = np.zeros((nj, self.P, self.P), np.int64)
        for i, j in enumerate(junctions):
            pm[i, :j.num_phases, :j.num_movements] = j.phase_movement
            pmask[i, :j.num_phases] = 1
            mmask[i, :j.num_movements] = 1
            rel[i, :j.num_phases, :j.num_phases] = j.relation
        t = lambda a: torch.as_tensor(a, device=self.device)  # noqa: E731
        self.t_pm, self.t_pmask, self.t_mmask, self.t_rel = t(pm), t(pmask), t(mmask), t(rel)
        self.num_phases = [j.num_phases for j in junctions]

        self.model = FRAP(self.conf["D_DENSE"]).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.conf["LEARNING_RATE"], eps=1e-8)
        self.memory = {j.id: [] for j in junctions}
        self.snapshots = {}            # round -> state_dict, used as Q_bar later
        self.epsilon = self.conf["EPSILON"]

    # ------------------------------------------------------------------ helpers
    def set_round(self, cnt_round: int):
        c = self.conf
        self.epsilon = max(c["EPSILON"] * c["EPSILON_DECAY"] ** cnt_round, c["MIN_EPSILON"])

    def _pad(self, pressure):
        out = np.zeros(self.M, np.float32)
        out[:len(pressure)] = pressure
        return out

    def _forward(self, net, jids, pressure, phases):
        j = torch.as_tensor(jids, device=self.device)
        ph = torch.as_tensor(phases, device=self.device)
        pm = self.t_pm[j]
        cur_bits = pm[torch.arange(len(jids), device=self.device), ph].long()
        x = torch.as_tensor(pressure, device=self.device)
        return net(x, cur_bits, pm, self.t_mmask[j], self.t_pmask[j], self.t_rel[j])

    # ------------------------------------------------------------------ acting
    def act(self, obs, explore=True):
        jids = list(range(len(self.junctions)))
        # official choose_action: one coin flip for all intersections at this step
        if explore and self.rng.random() <= self.epsilon:
            return {j.id: self.rng.randrange(j.num_phases) for j in self.junctions}
        pressure = np.stack([self._pad(obs[j.id]["pressure"]) for j in self.junctions])
        phases = [obs[j.id]["phase"] for j in self.junctions]
        with torch.no_grad():
            q = self._forward(self.model, jids, pressure, phases)
        best = q.argmax(dim=1).cpu().numpy()
        return {j.id: int(best[i]) for i, j in enumerate(self.junctions)}

    def remember(self, obs, actions, rewards, next_obs):
        for j in self.junctions:
            self.memory[j.id].append((
                self._pad(obs[j.id]["pressure"]), int(obs[j.id]["phase"]), int(actions[j.id]),
                float(rewards[j.id]),
                self._pad(next_obs[j.id]["pressure"]), int(next_obs[j.id]["phase"]),
            ))

    # ------------------------------------------------------------------ learning
    def _sample(self):
        c = self.conf
        for jid in self.memory:                               # updater.load_sample_with_forget
            self.memory[jid] = self.memory[jid][-c["MAX_MEMORY_LEN"]:]
        first = self.memory[self.junctions[0].id]
        size = min(c["SAMPLE_SIZE"], len(first))
        idx = self.rng.sample(range(len(first)), size)      # same indexes for every junction
        combined = []
        for j in self.junctions:
            mem = self.memory[j.id]
            combined += [(self.jidx[j.id],) + mem[k] for k in idx if k < len(mem)]
        combined = combined[-c["MAX_MEMORY_LEN"]:]           # network_agent.prepare_Xs_Y
        return self.rng.sample(combined, min(c["SAMPLE_SIZE"], len(combined)))

    def _masked_mse(self, q, y, mask):
        err = ((q.masked_fill(mask == 0, 0) - y) ** 2) * mask
        return (err.sum(1) / mask.sum(1)).mean()             # keras MSE: mean over outputs

    def update(self, cnt_round: int) -> dict:
        c = self.conf
        batch = self._sample()
        jids = np.array([b[0] for b in batch])
        s = np.stack([b[1] for b in batch])
        ph = np.array([b[2] for b in batch])
        a = np.array([b[3] for b in batch])
        r = np.array([b[4] for b in batch], np.float32)
        s2 = np.stack([b[5] for b in batch])
        ph2 = np.array([b[6] for b in batch])

        q_bar = FRAP(c["D_DENSE"]).to(self.device)
        if cnt_round == 0 or not self.snapshots:
            q_bar.load_state_dict(self.model.state_dict())
        else:
            src = max(cnt_round - c["UPDATE_Q_BAR_FREQ"], 0)
            src = src if src in self.snapshots else min(self.snapshots)
            q_bar.load_state_dict(self.snapshots[src])
        q_bar.eval()

        self.model.eval()
        with torch.no_grad():
            target = self._forward(self.model, jids, s, ph)
            next_q = self._forward(q_bar, jids, s2, ph2).max(dim=1).values
        mask = self.t_pmask[torch.as_tensor(jids, device=self.device)]
        y = target.masked_fill(mask == 0, 0).clone()
        rows = torch.arange(len(batch), device=self.device)
        y[rows, torch.as_tensor(a, device=self.device)] = (
            torch.as_tensor(r, device=self.device) / c["NORMAL_FACTOR"] + c["GAMMA"] * next_q)

        n = len(batch)
        split = int(n * (1 - c["VALIDATION_SPLIT"]))          # keras takes the last 30% as val
        bs = min(c["BATCH_SIZE"], n)
        best, wait, epochs, loss_v, val_v = float("inf"), 0, 0, 0.0, 0.0
        for epoch in range(c["EPOCHS"]):
            self.model.train()
            losses = []
            for start in range(0, split, bs):
                sl = slice(start, min(start + bs, split))
                q = self._forward(self.model, jids[sl], s[sl], ph[sl])
                loss = self._masked_mse(q, y[sl], mask[sl])
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
            self.model.eval()
            with torch.no_grad():
                qv = self._forward(self.model, jids[split:], s[split:], ph[split:])
                val_v = self._masked_mse(qv, y[split:], mask[split:]).item() if split < n else 0.0
            loss_v = float(np.mean(losses)) if losses else 0.0
            epochs = epoch + 1
            if val_v < best:                                  # keras EarlyStopping(min_delta=0)
                best, wait = val_v, 0
            else:
                wait += 1
                if wait >= c["PATIENCE"]:
                    break
        self.snapshots[cnt_round] = copy.deepcopy(self.model.state_dict())
        keep = cnt_round - c["UPDATE_Q_BAR_FREQ"]
        for k in [k for k in self.snapshots if k < keep]:
            del self.snapshots[k]
        return {"loss": loss_v, "val_loss": val_v, "epochs": epochs, "samples": n}

    # ------------------------------------------------------------------ persistence
    def save(self, path: Path, cnt_round: int):
        torch.save({
            "round": cnt_round,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "conf": self.conf,
            "junctions": [j.id for j in self.junctions],
        }, path)

    def load(self, path: Path, with_optimizer: bool = True) -> int:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        if ckpt.get("junctions") and ckpt["junctions"] != [j.id for j in self.junctions]:
            print(f"warning: checkpoint junctions {ckpt['junctions']} differ from current setup")
        self.model.load_state_dict(ckpt["model"])
        if with_optimizer and "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        return int(ckpt.get("round", -1))
