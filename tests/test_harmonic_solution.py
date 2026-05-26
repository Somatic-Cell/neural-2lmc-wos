import torch

from n2wos.pde import make_external_point_charges


def test_external_sources_are_outside_unit_ball():
    pde = make_external_point_charges(dim=3, n_sources=5, source_radius=1.2, seed=3)
    assert torch.all(torch.linalg.norm(pde.centers, dim=-1) > 1.0)


def test_boundary_value_matches_true_solution():
    pde = make_external_point_charges(dim=3, n_sources=5, source_radius=1.2, seed=3)
    y = torch.tensor([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    assert torch.allclose(pde.boundary_value(y), pde.true_solution(y))
