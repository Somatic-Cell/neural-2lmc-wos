from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from tqdm import trange

from n2wos.geometry import UnitBall
from n2wos.pde import make_external_point_charges
from n2wos.utils.rng import make_generator


def train_exact_cache(
    *,
    model: torch.nn.Module,
    geometry: UnitBall,
    pde,
    steps: int,
    batch_size: int,
    lr: float,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
    log_every: int = 250,
) -> dict[str, Any]:
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=float(lr))
    gen = make_generator(device, seed)
    history: list[dict[str, float | int]] = []

    pbar = trange(1, int(steps) + 1, desc="train exact-label cache")
    for step in pbar:
        x = geometry.sample_interior(
            int(batch_size), radius=0.999, device=device, dtype=dtype, generator=gen
        )
        y = pde.true_solution(x)
        pred = model(x)
        loss = torch.mean((pred - y) ** 2)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step == 1 or step % int(log_every) == 0 or step == int(steps):
            item = {"step": step, "loss": float(loss.detach().cpu())}
            history.append(item)
            pbar.set_postfix(loss=f"{item['loss']:.3e}")

    return {"history": history}


def save_checkpoint(
    *,
    path: str | Path,
    model: torch.nn.Module,
    model_config: dict,
    problem_config: dict,
    metadata: dict[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model_config,
            "problem_config": problem_config,
            "metadata": metadata,
        },
        path,
    )


def make_problem_from_config(problem_cfg: dict, *, device: torch.device, dtype: torch.dtype):
    geometry = UnitBall(dim=int(problem_cfg.get("dim", 3)))
    pde = make_external_point_charges(
        dim=geometry.dim,
        n_sources=int(problem_cfg.get("n_sources", 8)),
        source_radius=float(problem_cfg.get("source_radius", 1.18)),
        seed=int(problem_cfg.get("source_seed", 7)),
        weight_scale=float(problem_cfg.get("weight_scale", 0.25)),
        device=device,
        dtype=dtype,
    )
    return geometry, pde
