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


def ensemble_energy_ratio_timeseries_path() -> Path:
    """NPZ with downsampled ``KE₂/KE_tot`` time series per run (see ``prediction`` config)."""
    return _results_dir() / "ensemble_energy_ratio_timeseries.npz"


def _time_sample_indices(n_steps: int, n_samples: int) -> np.ndarray:
    """``n_samples`` indices in ``[0, n_steps]`` along the integration grid."""
    k = max(2, int(n_samples))
    return np.linspace(0, int(n_steps), k, dtype=int)


def _simulate_one_slot(
    run_id: int,
    params_candidates: list[dict[str, float]],
    config: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, np.ndarray | None]:
    """Simulate one slot, resampling candidate params until energy check passes.

    Returns
    -------
    row, energy_ratio_samples
        ``energy_ratio_samples`` is shape ``(K,)`` at indices from ``prediction.n_time_samples``
        (aligned with ``t_sample`` written to the NPZ). ``None`` if the slot failed.
    """
    pred_cfg = config.get("prediction") or {}
    save_ts = bool(pred_cfg.get("enabled", True))
    n_ts = int(pred_cfg.get("n_time_samples", 32))
    integ = config["integration"]
    n_steps = int(integ["n_steps"])
    ts_idx = _time_sample_indices(n_steps, n_ts) if save_ts else None

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

            row = {
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
            er_samp = None
            if save_ts and ts_idx is not None:
                er_samp = np.asarray(er[ts_idx], dtype=np.float64)
            return row, er_samp
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
    return None, None


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
    er_by_run: dict[int, np.ndarray] = {}

    for batch_start in range(0, n, checkpoint_every):
        batch_end = min(batch_start + checkpoint_every, n)
        batch_slots = []
        for i in range(batch_start, batch_end):
            start = i * max_energy_attempts_per_slot
            end = (i + 1) * max_energy_attempts_per_slot
            batch_slots.append((i, params_list[start:end]))

        batch_out = Parallel(n_jobs=n_jobs)(
            delayed(_simulate_one_slot)(i, cands, config) for i, cands in batch_slots
        )
        for row, er_samp in batch_out:
            if row is not None:
                rows_accum.append(row)
                if er_samp is not None:
                    er_by_run[int(row["run_id"])] = np.asarray(er_samp, dtype=np.float64)
        _checkpoint_write(rows_accum)

    df = pd.DataFrame(rows_accum)
    if len(df) < int(np.ceil(n * min_valid_fraction)):
        raise RuntimeError(
            "Too many ensemble slots failed validation (energy conservation / integration). "
            f"Got {len(df)} valid rows out of {n}."
        )
    df = df.sort_values("run_id").reset_index(drop=True)
    df.to_parquet(ensemble_results_path(config), index=False)

    pred_cfg = config.get("prediction") or {}
    if bool(pred_cfg.get("enabled", True)) and er_by_run:
        n_steps = int(config["integration"]["n_steps"])
        k_ts = int(pred_cfg.get("n_time_samples", 32))
        ts_idx = _time_sample_indices(n_steps, k_ts)
        t_span = tuple(float(x) for x in config["integration"]["t_span"])
        t_full = np.linspace(t_span[0], t_span[1], n_steps + 1, dtype=np.float64)
        t_sample = t_full[ts_idx]
        run_ids = df["run_id"].to_numpy(dtype=np.int64)
        er_stack = np.stack([er_by_run[int(r)] for r in run_ids], axis=0)
        np.savez_compressed(
            ensemble_energy_ratio_timeseries_path(),
            run_id=run_ids,
            t_sample=t_sample,
            time_index=ts_idx,
            energy_ratio=er_stack,
        )

    return df


def load_ensemble_results(path: Path | None = None) -> pd.DataFrame:
    """Load ensemble table from Parquet."""
    p = path or ensemble_results_path()
    if not p.is_file():
        raise FileNotFoundError(f"No ensemble results at {p}")
    return pd.read_parquet(p)
