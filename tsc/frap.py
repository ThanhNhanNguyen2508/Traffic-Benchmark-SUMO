"""FRAP phase-competition network used by MPLight, generalised to irregular junctions.

Layer sizes, activations and initialisation follow the official Keras model
(MPLight repo, transfer_dqn_agent_pressure_one.py: TransferDQNPressOneAgent):

    movement feature   d = Dense(4, sigmoid)(pressure)
    phase bit          p = sigmoid(Embedding(2, 4)(movement is green in current phase))
    lane embedding     Dense(16, relu)(concat[d, p])
    phase demand       sum of the lane embeddings of the movements in the phase
    pair demand        Conv1x1(D_DENSE, relu)(concat[demand_i, demand_j]),   i != j
    relation           Conv1x1(D_DENSE, relu)(Embedding(2, 4)(phases share a movement))
    combine            Conv1x1(D_DENSE, relu)(pair demand * relation)
    Q(i)               sum_j Conv1x1(1)(combine)[i, j]

The official model is hard-coded for 4-arm intersections (8 movements, 4 or 8
phases of exactly two movements). Here the number of movements and phases varies
per junction, so inputs are padded and masked; with 8 movements and the 8 standard
two-movement phases this computes exactly the original network.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _keras_init(module: nn.Module):
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)      # keras glorot_uniform
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.uniform_(m.weight, -0.05, 0.05)  # keras Embedding default


class FRAP(nn.Module):
    def __init__(self, d_dense: int = 20):
        super().__init__()
        self.phase_embedding = nn.Embedding(2, 4)
        self.num_vec_mapping = nn.Linear(1, 4)
        self.lane_embedding = nn.Linear(8, 16)
        self.lane_conv = nn.Conv2d(32, d_dense, kernel_size=1)
        self.relation_embedding = nn.Embedding(2, 4)
        self.relation_conv = nn.Conv2d(4, d_dense, kernel_size=1)
        self.combine_conv = nn.Conv2d(d_dense, d_dense, kernel_size=1)
        self.before_merge = nn.Conv2d(d_dense, 1, kernel_size=1)
        _keras_init(self)

    def forward(self, pressure, cur_bits, phase_mov, mov_mask, phase_mask, relation):
        """
        pressure   (B, M) float   movement pressures (padded with 0)
        cur_bits   (B, M) long    1 if the movement is green in the current phase
        phase_mov  (B, P, M)      1 if movement m belongs to phase p
        mov_mask   (B, M)         1 for real movements
        phase_mask (B, P)         1 for real phases
        relation   (B, P, P) long 1 if phases i and j share a movement
        returns    (B, P) Q-values (invalid phases set to -inf)
        """
        b, p_max = phase_mask.shape
        d = torch.sigmoid(self.num_vec_mapping(pressure.unsqueeze(-1)))      # B,M,4
        p = torch.sigmoid(self.phase_embedding(cur_bits))                     # B,M,4
        lane = F.relu(self.lane_embedding(torch.cat([d, p], dim=-1)))         # B,M,16
        lane = lane * mov_mask.unsqueeze(-1)
        demand = torch.bmm(phase_mov, lane)                                   # B,P,16

        di = demand.unsqueeze(2).expand(b, p_max, p_max, demand.size(-1))
        dj = demand.unsqueeze(1).expand(b, p_max, p_max, demand.size(-1))
        pair = torch.cat([di, dj], dim=-1).permute(0, 3, 1, 2)                # B,32,P,P
        pair = F.relu(self.lane_conv(pair))

        rel = self.relation_embedding(relation).permute(0, 3, 1, 2)           # B,4,P,P
        rel = F.relu(self.relation_conv(rel))

        comb = F.relu(self.combine_conv(pair * rel))
        score = self.before_merge(comb).squeeze(1)                            # B,P,P

        eye = torch.eye(p_max, device=score.device, dtype=score.dtype).unsqueeze(0)
        pair_mask = phase_mask.unsqueeze(2) * phase_mask.unsqueeze(1) * (1 - eye)
        q = (score * pair_mask).sum(-1)
        return q.masked_fill(phase_mask == 0, float("-inf"))
