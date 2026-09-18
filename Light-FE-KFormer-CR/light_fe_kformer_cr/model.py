from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class KFormerEncoder(nn.Module):
    """Learnable-keypoint attention encoder derived from the uploaded FE-KFormer.

    Input shape:  [batch, tokens, token_dim]
    Output shape: [batch, num_factors * factor_dim]

    Unlike the uploaded script, the optional input projection is actually used
    whenever token_dim differs from factor_dim, and similarity is computed in
    the projected space.
    """

    def __init__(
        self,
        token_dim: int = 8,
        num_factors: int = 4,
        factor_dim: int = 8,
        num_heads: int = 2,
    ) -> None:
        super().__init__()
        if factor_dim % num_heads != 0:
            raise ValueError("factor_dim must be divisible by num_heads")
        self.token_dim = token_dim
        self.num_factors = num_factors
        self.factor_dim = factor_dim

        self.Q = nn.Parameter(torch.empty(num_factors, factor_dim))
        self.input_proj = (
            nn.Identity() if token_dim == factor_dim else nn.Linear(token_dim, factor_dim)
        )
        self.self_attn = nn.MultiheadAttention(
            embed_dim=factor_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=factor_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self._init_weights()

    @property
    def output_dim(self) -> int:
        return self.num_factors * self.factor_dim

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.Q)
        if isinstance(self.input_proj, nn.Linear):
            nn.init.xavier_uniform_(self.input_proj.weight)
            nn.init.zeros_(self.input_proj.bias)
        for attn in (self.self_attn, self.cross_attn):
            nn.init.xavier_uniform_(attn.out_proj.weight)
            nn.init.zeros_(attn.out_proj.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(f"Expected [B,T,D] input, got {tuple(x.shape)}")
        batch_size = x.size(0)
        x_seq = self.input_proj(x)

        q = self.Q.unsqueeze(0).expand(batch_size, -1, -1)
        q_updated, _ = self.self_attn(q, q, q, need_weights=False)
        q_cross, _ = self.cross_attn(q_updated, x_seq, x_seq, need_weights=False)

        similarity = torch.bmm(q_cross, x_seq.transpose(1, 2)) / math.sqrt(
            self.factor_dim
        )
        attn_weights = torch.softmax(similarity, dim=-1)
        final_factors = torch.bmm(attn_weights, x_seq)
        fused = final_factors.reshape(batch_size, -1)
        return fused, similarity


class _PositiveRegressionHead(nn.Module):
    """Small positive-output regressor for CI or CR."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CR may exceed 1, so Softplus is preferable to sigmoid.
        return F.softplus(self.net(x))


class LightFEKFormerCR(nn.Module):
    """Paired normal/heatwave FE-KFormer for cooling resilience prediction."""

    def __init__(
        self,
        token_dim: int = 8,
        num_factors: int = 4,
        factor_dim: int = 8,
        num_heads: int = 2,
        head_hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        # One encoder object is deliberately shared by both weather states.
        self.encoder = KFormerEncoder(
            token_dim=token_dim,
            num_factors=num_factors,
            factor_dim=factor_dim,
            num_heads=num_heads,
        )
        state_dim = self.encoder.output_dim
        self.ci_normal_head = _PositiveRegressionHead(state_dim, head_hidden_dim)
        self.ci_heat_head = _PositiveRegressionHead(state_dim, head_hidden_dim)
        self.cr_head = _PositiveRegressionHead(state_dim * 4, head_hidden_dim)

    def forward(
        self, normal_tokens: torch.Tensor, heat_tokens: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        if normal_tokens.shape != heat_tokens.shape:
            raise ValueError(
                "Normal and heatwave token tensors must have the same shape; "
                f"got {tuple(normal_tokens.shape)} and {tuple(heat_tokens.shape)}"
            )

        f_normal, sim_normal = self.encoder(normal_tokens)
        f_heat, sim_heat = self.encoder(heat_tokens)

        diff = f_heat - f_normal
        abs_diff = diff.abs()
        pair_feature = torch.cat([f_normal, f_heat, diff, abs_diff], dim=-1)

        return {
            "ci_normal": self.ci_normal_head(f_normal),
            "ci_heat": self.ci_heat_head(f_heat),
            "cr": self.cr_head(pair_feature),
            "similarity_normal": sim_normal,
            "similarity_heat": sim_heat,
        }
