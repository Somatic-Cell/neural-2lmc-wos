from __future__ import annotations

import torch

from n2wos.utils.rng import make_generator
from n2wos.wos.directions import sample_unit_directions


def _flatten_walks(points: torch.Tensor, n_walks: int) -> torch.Tensor:
    if points.ndim != 2:
        raise ValueError(f"points must have shape [M, dim], got {tuple(points.shape)}")
    return points[None, :, :].expand(n_walks, -1, -1).reshape(n_walks * points.shape[0], points.shape[1]).clone()


@torch.no_grad()
def full_wos_samples(
    points: torch.Tensor,
    *,
    geometry,
    pde,
    n_walks: int,
    eps: float = 1.0e-4,
    max_steps: int = 512,
    seed: int | None = None,
) -> torch.Tensor:
    """Return independent full WoS samples with shape [n_walks, n_points]."""
    if n_walks <= 0:
        raise ValueError("n_walks must be positive")
    device = points.device
    dtype = points.dtype
    dim = points.shape[-1]
    gen = make_generator(device, seed)

    n_points = points.shape[0]
    x = _flatten_walks(points, n_walks)
    total = x.shape[0]
    values = torch.empty(total, device=device, dtype=dtype)
    active = torch.ones(total, device=device, dtype=torch.bool)

    for _ in range(max_steps):
        active_idx = torch.nonzero(active, as_tuple=False).flatten()
        if active_idx.numel() == 0:
            break
        xa = x.index_select(0, active_idx)
        r = geometry.distance(xa)
        hit = r <= eps

        if hit.any():
            hit_idx = active_idx[hit]
            y = geometry.project_to_boundary(xa[hit])
            values[hit_idx] = pde.boundary_value(y)
            active[hit_idx] = False

        keep = ~hit
        if keep.any():
            keep_idx = active_idx[keep]
            rk = r[keep].clamp_min(0.0)
            dirs = sample_unit_directions(
                keep_idx.numel(), dim, device=device, dtype=dtype, generator=gen
            )
            x[keep_idx] = x[keep_idx] + rk[:, None] * dirs

    active_idx = torch.nonzero(active, as_tuple=False).flatten()
    if active_idx.numel() > 0:
        y = geometry.project_to_boundary(x.index_select(0, active_idx))
        values[active_idx] = pde.boundary_value(y)
        active[active_idx] = False

    return values.reshape(n_walks, n_points)


@torch.no_grad()
def cached_wos_samples(
    points: torch.Tensor,
    *,
    geometry,
    pde,
    cache: torch.nn.Module,
    n_walks: int,
    cache_depth: int,
    eps: float = 1.0e-4,
    seed: int | None = None,
) -> torch.Tensor:
    """Return finite-depth cached WoS samples with shape [n_walks, n_points].

    The walk is advanced for at most cache_depth WoS steps. Particles that have
    reached the boundary return boundary data. Remaining particles return the
    frozen neural cache value at their current state.
    """
    if n_walks <= 0:
        raise ValueError("n_walks must be positive")
    if cache_depth < 0:
        raise ValueError("cache_depth must be non-negative")

    device = points.device
    dtype = points.dtype
    dim = points.shape[-1]
    gen = make_generator(device, seed)

    n_points = points.shape[0]
    x = _flatten_walks(points, n_walks)
    total = x.shape[0]
    values = torch.empty(total, device=device, dtype=dtype)
    active = torch.ones(total, device=device, dtype=torch.bool)

    for _ in range(cache_depth):
        active_idx = torch.nonzero(active, as_tuple=False).flatten()
        if active_idx.numel() == 0:
            break
        xa = x.index_select(0, active_idx)
        r = geometry.distance(xa)
        hit = r <= eps

        if hit.any():
            hit_idx = active_idx[hit]
            y = geometry.project_to_boundary(xa[hit])
            values[hit_idx] = pde.boundary_value(y)
            active[hit_idx] = False

        keep = ~hit
        if keep.any():
            keep_idx = active_idx[keep]
            rk = r[keep].clamp_min(0.0)
            dirs = sample_unit_directions(
                keep_idx.numel(), dim, device=device, dtype=dtype, generator=gen
            )
            x[keep_idx] = x[keep_idx] + rk[:, None] * dirs

    active_idx = torch.nonzero(active, as_tuple=False).flatten()
    if active_idx.numel() > 0:
        xa = x.index_select(0, active_idx)
        r = geometry.distance(xa)
        hit = r <= eps
        if hit.any():
            hit_idx = active_idx[hit]
            y = geometry.project_to_boundary(xa[hit])
            values[hit_idx] = pde.boundary_value(y)
            active[hit_idx] = False
        keep = ~hit
        if keep.any():
            keep_idx = active_idx[keep]
            pred = cache(xa[keep]).reshape(-1).to(dtype=dtype)
            values[keep_idx] = pred
            active[keep_idx] = False

    return values.reshape(n_walks, n_points)
