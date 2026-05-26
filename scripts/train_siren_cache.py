#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from n2wos.cache.siren import build_siren_from_config
from n2wos.training.train_exact import make_problem_from_config, save_checkpoint, train_exact_cache
from n2wos.utils.config import get_device, get_dtype, load_yaml, save_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    seed = int(cfg.get("seed", 1234))
    torch.manual_seed(seed)

    device = get_device(str(cfg.get("device", "auto")))
    dtype = get_dtype(str(cfg.get("dtype", "float32")))
    output_dir = Path(cfg.get("output_dir", "runs/ball3_siren_bias"))
    output_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(cfg, output_dir / "config_resolved.yaml")

    geometry, pde = make_problem_from_config(cfg.get("problem", {}), device=device, dtype=dtype)
    model = build_siren_from_config(geometry.dim, cfg.get("cache", {})).to(device=device, dtype=dtype)

    training_cfg = cfg.get("training", {})
    result = train_exact_cache(
        model=model,
        geometry=geometry,
        pde=pde,
        steps=int(training_cfg.get("steps", 3000)),
        batch_size=int(training_cfg.get("batch_size", 8192)),
        lr=float(training_cfg.get("lr", 1.0e-4)),
        seed=seed + 1000,
        device=device,
        dtype=dtype,
        log_every=int(training_cfg.get("log_every", 250)),
    )

    ckpt = training_cfg.get("checkpoint", str(output_dir / "siren.pt"))
    save_checkpoint(
        path=ckpt,
        model=model,
        model_config=cfg.get("cache", {}),
        problem_config=cfg.get("problem", {}),
        metadata={"training": training_cfg, "result": result},
    )
    print(f"saved checkpoint: {ckpt}")


if __name__ == "__main__":
    main()
