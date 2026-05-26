import torch
from torch import nn

from n2wos.geometry import UnitBall
from n2wos.pde import make_external_point_charges
from n2wos.wos.wavefront import cached_wos_samples, full_wos_samples


class ConstantCache(nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = float(value)

    def forward(self, x):
        return torch.full((x.shape[0],), self.value, device=x.device, dtype=x.dtype)


def test_cached_depth_zero_returns_cache_for_interior_points():
    geom = UnitBall(dim=3)
    pde = make_external_point_charges(dim=3, n_sources=3, seed=0)
    cache = ConstantCache(1.25)
    points = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.2, 0.0]], dtype=torch.float32)
    samples = cached_wos_samples(
        points,
        geometry=geom,
        pde=pde,
        cache=cache,
        n_walks=4,
        cache_depth=0,
        eps=1.0e-5,
        seed=9,
    )
    assert samples.shape == (4, 2)
    assert torch.allclose(samples, torch.full_like(samples, 1.25))


def test_full_wos_returns_finite_samples():
    geom = UnitBall(dim=3)
    pde = make_external_point_charges(dim=3, n_sources=3, seed=0)
    points = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.2, 0.0]], dtype=torch.float32)
    samples = full_wos_samples(
        points,
        geometry=geom,
        pde=pde,
        n_walks=8,
        eps=1.0e-4,
        max_steps=128,
        seed=10,
    )
    assert samples.shape == (8, 2)
    assert torch.isfinite(samples).all()
