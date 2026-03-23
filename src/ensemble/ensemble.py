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


def _simulate_one(
    run_id: int,
    params: dict[str, float],
    config: Mapping[str, Any],
) -> dict[str, Any]:
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
    eps = float(config["lyapunov"]["epsilon"])
    chaotic = bool(mle > eps)

    row: dict[str, Any] = {
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
    return row


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

    params_list = sample_parameters(n, config, rng)

    rows_accum: list[dict[str, Any]] = []

    for batch_start in range(0, n, checkpoint_every):
        batch_end = min(batch_start + checkpoint_every, n)
        batch_params = [
            (i, params_list[i]) for i in range(batch_start, batch_end)
        ]
        batch_rows = Parallel(n_jobs=n_jobs)(
            delayed(_simulate_one)(i, p, config) for i, p in batch_params
        )
        rows_accum.extend(batch_rows)
        _checkpoint_write(rows_accum)

    df = pd.DataFrame(rows_accum)
    df.to_parquet(ensemble_results_path(config), index=False)
    return df


def load_ensemble_results(path: Path | None = None) -> pd.DataFrame:
    """Load ensemble table from Parquet."""
    p = path or ensemble_results_path()
    if not p.is_file():
        raise FileNotFoundError(f"No ensemble results at {p}")
    return pd.read_parquet(p)
