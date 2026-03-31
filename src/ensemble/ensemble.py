"""Parallel ensemble simulation with checkpointed Parquet output."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numpy.random import Generator

from ensemble.lyapunov import compute_mle
from ensemble.sampler import sample_parameters
from physics.energy import compute_energy_timeseries
from physics.integrator import integrate
from physics.pendulum import dstate_dt


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _results_dir() -> Path:
    d = _project_root() / "data" / "results"
    d.mkdir(parents=True, exist_ok=True)
    return d


def ensemble_results_path(config: Mapping[str, Any] | None = None) -> Path:
    """Default path for the merged ensemble table."""
    return _results_dir() / "ensemble_results.parquet"


def ensemble_checkpoint_path() -> Path:
    return _results_dir() / "ensemble_checkpoint.parquet"


def _simulate_one_slot(
    run_id: int,
    params_candidates: list[dict[str, float]],
    config: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Simulate one slot, resampling candidate params until energy check passes."""
    last_exc: Exception | None = None
    for params in params_candidates:
        try:
            state0 = np.array(
                [
                    params["theta1"],
                    params["omega1"],
                    params["theta2"],
                    params["omega2"],
                ],
                dtype=np.float64,
            )
            t, y = integrate(dstate_dt, state0, params, config)
            energies = compute_energy_timeseries(t, y, params, config)
            er = energies["energy_ratio"]
            mask = np.isfinite(er)
            er_clean = er[mask]
            var_er = float(np.var(er_clean)) if er_clean.size > 1 else 0.0
            mean_er = float(np.mean(er_clean)) if er_clean.size else float("nan")

            mle = float(compute_mle(t, y, params, config))
            if not np.isfinite(mle):
                raise ValueError("Non-finite MLE.")

            eps = float(config["lyapunov"]["epsilon"])
            chaotic = bool(mle > eps)

            return {
                "run_id": run_id,
                "m1": params["m1"],
                "m2": params["m2"],
                "L1": params["L1"],
                "L2": params["L2"],
                "theta1": params["theta1"],
                "theta2": params["theta2"],
                "omega1": params["omega1"],
                "omega2": params["omega2"],
                "mle": mle,
                "energy_ratio_mean": mean_er,
                "energy_ratio_variance": var_er,
                "is_chaotic": chaotic,
            }
        except (AssertionError, RuntimeError, ValueError) as exc:
            # Known/expected numerical issues:
            # - energy conservation drift (AssertionError)
            # - solve_ivp failure (RuntimeError)
            # - pathological Lyapunov config / underflow edge cases (ValueError)
            last_exc = exc
            continue

    # Slot failed all attempts; let the caller decide whether to abort.
    if last_exc is None:
        last_exc = RuntimeError("Unknown slot failure.")
    return None


def _checkpoint_write(rows: list[dict[str, Any]]) -> None:
    path = ensemble_checkpoint_path()
    pd.DataFrame(rows).to_parquet(path, index=False)


def run_ensemble(
    config: Mapping[str, Any],
    rng: Generator | None = None,
) -> pd.DataFrame:
    """Run N pendulums in parallel; checkpoint to Parquet; write final merged file.

    Parameters
    ----------
    config
        Full loaded YAML config.
    rng
        Optional ``numpy.random.Generator``; if omitted, uses
        ``default_rng(config['ensemble']['seed'])``.
    """
    ens = config["ensemble"]
    n = int(ens["n_pendulums"])
    seed = int(ens["seed"])
    if rng is None:
        rng = np.random.default_rng(seed)

    checkpoint_every = max(1, int(ens.get("checkpoint_every", 25)))
    n_jobs = int(ens.get("n_jobs", -1))

    max_energy_attempts_per_slot = int(ens.get("max_energy_attempts_per_slot", 10))
    if max_energy_attempts_per_slot < 1:
        raise ValueError("ensemble.max_energy_attempts_per_slot must be >= 1.")

    n_candidates = n * max_energy_attempts_per_slot
    params_list = sample_parameters(n_candidates, config, rng)

    min_valid_fraction = float(ens.get("min_valid_fraction", 0.98))
    if not (0.0 < min_valid_fraction <= 1.0):
        raise ValueError("ensemble.min_valid_fraction must be in (0, 1].")

    rows_accum: list[dict[str, Any]] = []

    for batch_start in range(0, n, checkpoint_every):
        batch_end = min(batch_start + checkpoint_every, n)
        batch_slots = []
        for i in range(batch_start, batch_end):
            start = i * max_energy_attempts_per_slot
            end = (i + 1) * max_energy_attempts_per_slot
            batch_slots.append((i, params_list[start:end]))

        batch_rows = Parallel(n_jobs=n_jobs)(
            delayed(_simulate_one_slot)(i, cands, config) for i, cands in batch_slots
        )
        rows_accum.extend([r for r in batch_rows if r is not None])
        _checkpoint_write(rows_accum)

    df = pd.DataFrame(rows_accum)
    if len(df) < int(np.ceil(n * min_valid_fraction)):
        raise RuntimeError(
            "Too many ensemble slots failed validation (energy conservation / integration). "
            f"Got {len(df)} valid rows out of {n}."
        )
    df.to_parquet(ensemble_results_path(config), index=False)
    return df


def load_ensemble_results(path: Path | None = None) -> pd.DataFrame:
    """Load ensemble table from Parquet."""
    p = path or ensemble_results_path()
    if not p.is_file():
        raise FileNotFoundError(f"No ensemble results at {p}")
    return pd.read_parquet(p)
