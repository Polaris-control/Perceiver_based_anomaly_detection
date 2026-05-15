from __future__ import annotations

import math
from typing import Iterable, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from perceiver.model.core.modules import MultiHeadAttention


class LinearWithLoRA(nn.Module):
    """Wraps an existing nn.Linear: y = base(x) + dropout(x @ A @ B) * (alpha / r)."""

    def __init__(
        self,
        linear: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive for LinearWithLoRA")
        self.base = linear
        for p in self.base.parameters():
            p.requires_grad = False # 冻结base线性层

        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank

        self.lora_a = nn.Parameter(torch.empty(linear.in_features, rank))
        self.lora_b = nn.Parameter(torch.empty(rank, linear.out_features))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b) # lora_b初始化为0
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        z = self.dropout(x) @ self.lora_a @ self.lora_b
        return y + z * self.scaling


def inject_lora_into_encoder(
    encoder: nn.Module,
    rank: int,
    alpha: float,
    dropout: float = 0.0,
    target_proj_names: Sequence[str] = ("q_proj", "v_proj"),
) -> Tuple[int, int]:
    """
    Wrap selected projections on every MultiHeadAttention under ``encoder``.

    Returns (num_wrapped_linears, num_skipped_already_wrapped).
    """
    if rank <= 0:
        return 0, 0

    wrapped = 0
    skipped = 0
    # Snapshot to avoid mutating tree while iterating
    for module in list(encoder.modules()):
        if not isinstance(module, MultiHeadAttention):
            continue
        for attr in target_proj_names:
            if not hasattr(module, attr):
                continue
            linear = getattr(module, attr)
            if isinstance(linear, LinearWithLoRA):
                skipped += 1
                continue
            if not isinstance(linear, nn.Linear):
                continue
            setattr(module, attr, LinearWithLoRA(linear, rank, alpha, dropout))
            wrapped += 1
    return wrapped, skipped


def lora_parameter_count(encoder: nn.Module) -> int:
    n = 0
    for m in encoder.modules():
        if isinstance(m, LinearWithLoRA):
            n += m.lora_a.numel() + m.lora_b.numel()
    return n


def iter_lora_parameters(encoder: nn.Module) -> Iterable[torch.nn.Parameter]:
    for m in encoder.modules():
        if isinstance(m, LinearWithLoRA):
            yield m.lora_a
            yield m.lora_b
