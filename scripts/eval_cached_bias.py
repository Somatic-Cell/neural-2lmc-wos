#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from n2wos.cache.siren import build_siren_from_config
from n2wos.eval.bias_variance import estimate_repeated
from n2wos.training.train_exact import make_problem_from_config
from n2wos.utils.config import get_device, get_dtype, load_yaml
from n2wos.utils.rng import make_generator
from n2wos.wos.wavefront import cached_wos_samples, full_wos_samples


def _load_cache(path: str | Path, dim: int, cache_cfg: dict, device, dtype):
    model = build_siren_from_config(dim, cache_cfg).to(device=device, dtype=dtype)
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def _make_eval_points(geometry, n_points: int, radius: float, seed: int, device, dtype):
    gen = make_generator(device, seed)
    return geometry.sample_interior(
        n_points, radius=radius, device=device, dtype=dtype, generator=gen
    )


def _plot_results(results: dict, output_dir: Path) -> None:
    sample_counts = sorted({int(k.split("N")[-1]) for k in results if "_N" in k})

    plt.figure()
    for prefix in sorted({k.split("_N")[0] for k in results if "_N" in k}):
        xs = []
        ys = []
        for n in sample_counts:
            key = f"{prefix}_N{n}"
            if key in results:
                xs.append(n)
                ys.append(results[key]["mse_mean"])
        if xs:
            plt.loglog(xs, ys, marker="o", label=prefix)
    plt.xlabel("samples per evaluation point")
    plt.ylabel("mean MSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "mse_vs_samples.png", dpi=180)
    plt.close()

    plt.figure()
    for prefix in sorted({k.split("_N")[0] for k in results if "_N" in k}):
        xs = []
        bias2 = []
        var = []
        for n in sample_counts:
            key = f"{prefix}_N{n}"
            if key in results:
                xs.append(n)
                bias2.append(results[key]["bias2_mean"])
                var.append(results[key]["var_mean"])
        if xs:
            plt.loglog(xs, bias2, marker="o", linestyle="-", label=f"{prefix} bias^2")
            plt.loglog(xs, var, marker="x", linestyle="--", label=f"{prefix} var")
    plt.xlabel("samples per evaluation point")
    plt.ylabel("mean component")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(output_dir / "decomp_vs_samples.png", dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    seed = int(cfg.get("seed", 1234))
    device = get_device(str(cfg.get("device", "auto")))
    dtype = get_dtype(str(cfg.get("dtype", "float32")))
    output_dir = Path(cfg.get("output_dir", "runs/ball3_siren_bias"))
    output_dir.mkdir(parents=True, exist_ok=True)

    geometry, pde = make_problem_from_config(cfg.get("problem", {}), device=device, dtype=dtype)
    eval_cfg = cfg.get("eval", {})
    cache = _load_cache(eval_cfg.get("checkpoint"), geometry.dim, cfg.get("cache", {}), device, dtype)

    points = _make_eval_points(
        geometry,
        n_points=int(eval_cfg.get("n_points", 128)),
        radius=float(eval_cfg.get("point_radius", 0.85)),
        seed=seed + 2000,
        device=device,
        dtype=dtype,
    )
    truth = pde.true_solution(points).detach()

    repetitions = int(eval_cfg.get("repetitions", 64))
    sample_counts = [int(n) for n in eval_cfg.get("sample_counts", [1, 2, 4, 8, 16, 32, 64])]
    cache_depths = [int(m) for m in eval_cfg.get("cache_depths", [0, 1, 2, 4, 8])]
    eps = float(eval_cfg.get("eps", 1.0e-4))
    max_steps = int(eval_cfg.get("max_steps", 512))

    results: dict[str, dict[str, float]] = {}

    if bool(eval_cfg.get("full_wos", True)):
        for n in tqdm(sample_counts, desc="full WoS eval"):
            def estimator(rep: int, n=n):
                samples = full_wos_samples(
                    points,
                    geometry=geometry,
                    pde=pde,
                    n_walks=n,
                    eps=eps,
                    max_steps=max_steps,
                    seed=seed + 10_000 + 1_000_000 * rep + n,
                )
                return samples.mean(dim=0)

            res = estimate_repeated(estimator=estimator, truth=truth, repetitions=repetitions)
            results[f"full_wos_N{n}"] = res.metrics()

    for depth in cache_depths:
        for n in tqdm(sample_counts, desc=f"cached depth {depth}"):
            def estimator(rep: int, n=n, depth=depth):
                samples = cached_wos_samples(
                    points,
                    geometry=geometry,
                    pde=pde,
                    cache=cache,
                    n_walks=n,
                    cache_depth=depth,
                    eps=eps,
                    seed=seed + 20_000 + 1_000_000 * rep + 10_000 * depth + n,
                )
                return samples.mean(dim=0)

            res = estimate_repeated(estimator=estimator, truth=truth, repetitions=repetitions)
            results[f"cached_m{depth}_N{n}"] = res.metrics()

    out_json = output_dir / "cached_bias_results.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    _plot_results(results, output_dir)

    print(f"saved metrics: {out_json}")
    print(f"saved plots: {output_dir / 'mse_vs_samples.png'}")
    print(f"saved plots: {output_dir / 'decomp_vs_samples.png'}")

    # Print a compact summary for terminal inspection.
    for k in sorted(results):
        m = results[k]
        print(
            f"{k:18s} mse={m['mse_mean']:.3e} bias2={m['bias2_mean']:.3e} "
            f"var={m['var_mean']:.3e} frac|bias|>3sem={m['frac_points_abs_bias_gt_3sem']:.2f}"
        )


if __name__ == "__main__":
    main()
