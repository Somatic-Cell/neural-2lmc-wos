from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


@dataclass
class BiasVarianceResult:
    estimates: torch.Tensor
    truth: torch.Tensor
    mean: torch.Tensor
    bias: torch.Tensor
    variance: torch.Tensor
    mse_per_point: torch.Tensor

    def metrics(self) -> dict[str, float]:
        sem = torch.sqrt(self.variance / max(self.estimates.shape[0], 1))
        significant = torch.abs(self.bias) > 3.0 * sem.clamp_min(1.0e-30)
        return {
            "bias2_mean": float(torch.mean(self.bias**2).cpu()),
            "bias_l2": float(torch.sqrt(torch.mean(self.bias**2)).cpu()),
            "var_mean": float(torch.mean(self.variance).cpu()),
            "mse_mean": float(torch.mean(self.mse_per_point).cpu()),
            "max_abs_bias": float(torch.max(torch.abs(self.bias)).cpu()),
            "mean_abs_bias": float(torch.mean(torch.abs(self.bias)).cpu()),
            "frac_points_abs_bias_gt_3sem": float(torch.mean(significant.float()).cpu()),
        }


def estimate_repeated(
    *,
    estimator: Callable[[int], torch.Tensor],
    truth: torch.Tensor,
    repetitions: int,
) -> BiasVarianceResult:
    """Run estimator(rep_index) repeatedly and decompose error.

    estimator(rep_index) must return one estimate per evaluation point, shape [M].
    """
    outs = []
    for rep in range(int(repetitions)):
        outs.append(estimator(rep).detach())
    estimates = torch.stack(outs, dim=0)
    mean = estimates.mean(dim=0)
    variance = estimates.var(dim=0, unbiased=True) if repetitions > 1 else torch.zeros_like(mean)
    bias = mean - truth
    mse_per_point = ((estimates - truth[None, :]) ** 2).mean(dim=0)
    return BiasVarianceResult(
        estimates=estimates,
        truth=truth,
        mean=mean,
        bias=bias,
        variance=variance,
        mse_per_point=mse_per_point,
    )
