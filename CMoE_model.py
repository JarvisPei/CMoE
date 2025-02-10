import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from typing import Optional, Tuple, List


class LlamaMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str = "silu"
    ):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = F.silu if hidden_act == "silu" else getattr(F, hidden_act)

    def forward(self, x):
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)

        intermediate = gate * up
        output = self.down_proj(intermediate)
        return output

class Router(nn.Module):
    def __init__(self, hidden_size, n_experts, n_activated, bias_speed = 0.001):
        super().__init__()
        self.dim = hidden_size
        self.topk = n_activated

        self.act_fn = F.silu
        self.gate = nn.Linear(hidden_size, n_experts, bias=False)
        self.classifier = nn.Linear(hidden_size, n_experts, bias=False)

        self.extra_scale = nn.Parameter(torch.zeros(n_experts, device='cuda', dtype=torch.bfloat16))
        self.extra_bias = torch.zeros(n_experts, device='cuda', dtype=torch.float32)
        self.bias_update_speed = bias_speed
    
    def update_bias(self, counts):
        mean_load = counts.mean()
        # Decrease bias for overloaded experts, increase for underloaded
        overloaded = counts > mean_load
        underloaded = counts < mean_load

        self.extra_bias.data[overloaded] -= self.bias_update_speed
        self.extra_bias.data[underloaded] += self.bias_update_speed

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        scores = (self.classifier(x) * self.act_fn(self.gate(x))).abs() 

        scores = scores.softmax(dim=-1, dtype=torch.float32)
        original_scores = scores
        scores = scores + self.extra_bias[None, :]

        indices = torch.topk(scores, self.topk, dim=-1)[1]

        original_scores = 1 + original_scores*self.extra_scale
        weights = original_scores.gather(1, indices)

        return weights.type_as(x), indices

class MoE(nn.Module):
    def __init__(self, hidden_size, moe_inter_dim, n_experts, n_shared, n_activated):
        super().__init__()
        self.dim = hidden_size
        n_routed_experts = n_experts - n_shared
        self.n_routed_experts = n_routed_experts
        self.n_activated_experts = n_activated
        self.experts_start_idx = 0
        self.experts_end_idx = n_routed_experts
        self.gate = Router(hidden_size, n_routed_experts, self.n_activated_experts)
        self.n_shared_experts = n_shared
        self.experts = nn.ModuleList([LlamaMLP(self.dim, moe_inter_dim) if self.experts_start_idx <= i < self.experts_end_idx else None
                                      for i in range(self.n_routed_experts)])
        self.shared_experts = LlamaMLP(self.dim, self.n_shared_experts * moe_inter_dim)


        self.cus_training = False
        self.enable_scale = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.size()
        x = x.view(-1, self.dim)
        weights, indices = self.gate(x)
        y = torch.zeros_like(x)
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts)
        if self.cus_training:
            self.gate.update_bias(counts.to(dtype = torch.bfloat16))

        counts = counts.tolist()
        for i in range(self.experts_start_idx, self.experts_end_idx):
            if counts[i] == 0:
                continue
            expert = self.experts[i]
            idx, top = torch.where(indices == i)
            if self.enable_scale:
                y[idx] += expert(x[idx]) * weights[idx, top, None]
            else:
                y[idx] += expert(x[idx]) 
        z = self.shared_experts(x)
        return (y + z).view(shape)

