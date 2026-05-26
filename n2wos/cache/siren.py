from __future__ import annotations

import math

import torch
from torch import nn


class SineLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, *, bias: bool = True, is_first: bool = False, w0: float = 30.0):
        super().__init__()
        self.in_features = int(in_features)
        self.is_first = bool(is_first)
        self.w0 = float(w0)
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            if self.is_first:
                bound = 1.0 / self.in_features
            else:
                bound = math.sqrt(6.0 / self.in_features) / self.w0
            self.linear.weight.uniform_(-bound, bound)
            if self.linear.bias is not None:
                self.linear.bias.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w0 * self.linear(x))


class Siren(nn.Module):
    def __init__(
        self,
        *,
        in_dim: int,
        hidden_dim: int = 128,
        hidden_layers: int = 3,
        out_dim: int = 1,
        w0: float = 30.0,
        outermost_linear: bool = True,
    ):
        super().__init__()
        layers: list[nn.Module] = [
            SineLayer(in_dim, hidden_dim, is_first=True, w0=w0),
        ]
        for _ in range(hidden_layers):
            layers.append(SineLayer(hidden_dim, hidden_dim, is_first=False, w0=w0))
        if outermost_linear:
            final = nn.Linear(hidden_dim, out_dim)
            with torch.no_grad():
                bound = math.sqrt(6.0 / hidden_dim) / w0
                final.weight.uniform_(-bound, bound)
                final.bias.uniform_(-bound, bound)
            layers.append(final)
        else:
            layers.append(SineLayer(hidden_dim, out_dim, is_first=False, w0=w0))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def build_siren_from_config(dim: int, cfg: dict) -> Siren:
    return Siren(
        in_dim=dim,
        hidden_dim=int(cfg.get("hidden_dim", 128)),
        hidden_layers=int(cfg.get("hidden_layers", 3)),
        w0=float(cfg.get("w0", 30.0)),
        outermost_linear=bool(cfg.get("outermost_linear", True)),
    )
