from __future__ import annotations

from dataclasses import dataclass

import torch

from n2wos.wos.directions import sample_unit_directions


@dataclass(frozen=True)
class UnitBall:
    """Analytic unit ball geometry in R^dim.

    This backend is the first validation geometry because it gives exact distance
    and boundary projection queries. It therefore keeps geometry bias out of the
    initial Neural-Caches-style bias test.
    """

    dim: int = 3

    def distance(self, x: torch.Tensor) -> torch.Tensor:
        """Distance from x to the unit-sphere boundary, positive inside."""
        return 1.0 - torch.linalg.norm(x, dim=-1)

    def project_to_boundary(self, x: torch.Tensor) -> torch.Tensor:
        """Radial projection to the unit-sphere boundary."""
        n = torch.linalg.norm(x, dim=-1, keepdim=True).clamp_min(1.0e-30)
        return x / n

    def inside(self, x: torch.Tensor) -> torch.Tensor:
        return torch.linalg.norm(x, dim=-1) < 1.0

    def sample_interior(
        self,
        n: int,
        *,
        radius: float = 1.0,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Uniformly sample inside a ball of the given radius."""
        device = torch.device(device)
        dirs = sample_unit_directions(n, self.dim, device=device, dtype=dtype, generator=generator)
        u = torch.rand(n, device=device, dtype=dtype, generator=generator)
        r = radius * u.pow(1.0 / self.dim)
        return dirs * r[:, None]

    def sample_boundary(
        self,
        n: int,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        device = torch.device(device)
        return sample_unit_directions(n, self.dim, device=device, dtype=dtype, generator=generator)
