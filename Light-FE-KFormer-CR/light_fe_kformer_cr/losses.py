from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn


class CoolingResilienceLoss(nn.Module):
    """Multi-task CI/CR loss with physical consistency regularization."""

    def __init__(
        self,
        *,
        lambda_cr: float = 1.0,
        lambda_consistency: float = 0.2,
        beta: float = 1.0,
        eps: float = 1e-3,
    ) -> None:
        super().__init__()
        self.lambda_cr = lambda_cr
        self.lambda_consistency = lambda_consistency
        self.eps = eps
        self.regression = nn.SmoothL1Loss(beta=beta)

    def forward(
        self,
        pred: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        required = {"ci_normal", "ci_heat", "cr"}
        missing_pred = required.difference(pred)
        missing_target = required.difference(target)
        if missing_pred or missing_target:
            raise KeyError(
                f"Missing prediction keys={sorted(missing_pred)}, "
                f"target keys={sorted(missing_target)}"
            )

        l_normal = self.regression(pred["ci_normal"], target["ci_normal"])
        l_heat = self.regression(pred["ci_heat"], target["ci_heat"])
        l_cr = self.regression(pred["cr"], target["cr"])

        implied_cr = pred["ci_heat"] / pred["ci_normal"].clamp_min(self.eps)
        l_consistency = self.regression(pred["cr"], implied_cr)

        total = (
            l_normal
            + l_heat
            + self.lambda_cr * l_cr
            + self.lambda_consistency * l_consistency
        )
        parts = {
            "ci_normal": l_normal.detach(),
            "ci_heat": l_heat.detach(),
            "cr": l_cr.detach(),
            "consistency": l_consistency.detach(),
        }
        return total, parts
