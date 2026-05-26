import torch

from n2wos.geometry import UnitBall


def test_unit_ball_distance_and_projection():
    geom = UnitBall(dim=3)
    x = torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.3, 0.4]])
    d = geom.distance(x)
    assert torch.allclose(d, torch.tensor([1.0, 0.5, 0.5]), atol=1.0e-6)

    y = geom.project_to_boundary(x[1:])
    assert torch.allclose(torch.linalg.norm(y, dim=-1), torch.ones(2), atol=1.0e-6)


def test_unit_ball_samples_are_inside():
    geom = UnitBall(dim=3)
    gen = torch.Generator().manual_seed(1)
    x = geom.sample_interior(256, radius=0.9, generator=gen)
    assert torch.all(torch.linalg.norm(x, dim=-1) <= 0.900001)
