from __future__ import annotations

import torch


def sample_unit_directions(
    n: int,
    dim: int,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample n directions uniformly on S^(dim-1)."""
    z = torch.randn(n, dim, device=device, dtype=dtype, generator=generator)
    return z / torch.linalg.norm(z, dim=-1, keepdim=True).clamp_min(1.0e-30)
