"""Out-of-fold prediction of instantaneous KE₂/KE_tot: sharpness vs time and breakdown time."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import norm
from sklearn.model_selection import KFold

from ensemble.ensemble import ensemble_energy_ratio_timeseries_path
from stats.stats import FEATURE_ORDER, fit_gp_regressor, predict_with_ci


def load_energy_ratio_timeseries(
    path: Path | None = None,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.integer]]:
    """Load ``run_id``, ``t_sample``, and ``energy_ratio`` (n × K) from NPZ."""
    p = path or ensemble_energy_ratio_timeseries_path()
    if not p.is_file():
        raise FileNotFoundError(f"No ensemble energy-ratio timeseries at {p}")
    z = np.load(p)
    run_id = z["run_id"]
    t_sample = z["t_sample"].astype(np.float64)
    er = z["energy_ratio"].astype(np.float64)
    return run_id, t_sample, er


def _analytic_pi(
    model: Mapping[str, Any],
    X: NDArray[np.floating],
    confidence_level: float,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """GPR predictive mean and normal-approx equal-tailed interval; returns mean, lo, hi, width."""
    gpr = model["gpr"]
    mean, std = gpr.predict(np.asarray(X, dtype=np.float64), return_std=True)
    std = np.maximum(np.asarray(std, dtype=np.float64), 1e-15)
    z = float(norm.ppf(0.5 + 0.5 * float(confidence_level)))
    lo = mean - z * std
    hi = mean + z * std
    width = hi - lo
    return mean.astype(np.float64), lo.astype(np.float64), hi.astype(np.float64), width.astype(np.float64)


def run_prediction_breakdown_oof(
    df: pd.DataFrame,
    t_sample: NDArray[np.floating],
    energy_ratio: NDArray[np.floating],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """K-fold OOF: at each time slice, GPR(initial → energy_ratio(t)).

    Sharpness: interval width (analytic GP or bootstrap). Breakdown time for run ``i``:
    first ``t`` where the true ratio lies **outside** the OOF predictive interval.

    Parameters
    ----------
    df
        Ensemble table sorted consistently with ``energy_ratio`` rows (same order as ``run_id`` in NPZ).
    t_sample
        Shape ``(K,)`` physical times.
    energy_ratio
        Shape ``(n, K)`` with ``n == len(df)``.
    """
    pred = config.get("prediction") or {}
    n_splits = max(2, int(pred.get("cv_folds", 5)))
    interval_method = str(pred.get("interval_method", "analytic")).lower()
    if interval_method not in ("analytic", "bootstrap"):
        raise ValueError("prediction.interval_method must be 'analytic' or 'bootstrap'.")

    n, k = energy_ratio.shape
    if len(df) != n:
        raise ValueError("df length must match energy_ratio rows.")
    if t_sample.shape[0] != k:
        raise ValueError("t_sample length must match energy_ratio columns.")

    X = df[FEATURE_ORDER].to_numpy(dtype=np.float64)
    conf = float(config["statistics"]["confidence_level"])
    n_boot_pred = int(pred.get("bootstrap_iterations", config["statistics"].get("figure_bootstrap_iterations", 80)))

    n_splits = min(n_splits, n)
    if n < 2 * n_splits:
        n_splits = max(2, n // 2)
    if n_splits < 2 or n < 4:
        raise ValueError("Need at least 4 ensemble rows for breakdown CV.")

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=int(pred.get("cv_random_state", 0)))

    oof_lo = np.full((n, k), np.nan, dtype=np.float64)
    oof_hi = np.full((n, k), np.nan, dtype=np.float64)
    oof_mean = np.full((n, k), np.nan, dtype=np.float64)
    oof_width = np.full((n, k), np.nan, dtype=np.float64)

    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        for j in range(k):
            y_tr = energy_ratio[train_idx, j]
            if not np.all(np.isfinite(y_tr)):
                mask = np.isfinite(y_tr)
                if np.sum(mask) < 3:
                    continue
                X_tr_j = X_tr[mask]
                y_tr_j = y_tr[mask]
            else:
                X_tr_j, y_tr_j = X_tr, y_tr

            model = fit_gp_regressor(
                X_tr_j,
                y_tr_j,
                config,
                optimize_hyperparameters=False,
            )
            if interval_method == "analytic":
                m, lo, hi, w = _analytic_pi(model, X_val, conf)
            else:
                m, lo, hi = predict_with_ci(model, X_val, config, n_bootstrap=n_boot_pred)
                w = hi - lo
            oof_mean[val_idx, j] = m
            oof_lo[val_idx, j] = lo
            oof_hi[val_idx, j] = hi
            oof_width[val_idx, j] = w

    y = energy_ratio
    valid = np.isfinite(y) & np.isfinite(oof_lo) & np.isfinite(oof_hi)
    outside = valid & ((y < oof_lo) | (y > oof_hi))
    denom = np.maximum(np.sum(valid, axis=0), 1)
    frac_outside = np.sum(outside & valid, axis=0).astype(np.float64) / denom.astype(np.float64)

    median_width = np.nanmedian(oof_width, axis=0)
    mean_width = np.nanmean(oof_width, axis=0)

    t_break = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        hit = np.where(outside[i, :])[0]
        if hit.size:
            j0 = int(hit[0])
            t_break[i] = float(t_sample[j0])

    return {
        "t_sample": t_sample,
        "oof_lower": oof_lo,
        "oof_upper": oof_hi,
        "oof_mean": oof_mean,
        "oof_width": oof_width,
        "median_interval_width": median_width,
        "mean_interval_width": mean_width,
        "fraction_outside_ci": frac_outside,
        "t_breakdown": t_break,
        "interval_method": interval_method,
        "cv_folds": n_splits,
    }


def align_timeseries_to_dataframe(
    df: pd.DataFrame,
    run_id: NDArray[np.integer],
    energy_ratio: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Reorder ``energy_ratio`` rows to match ``df`` order (by ``run_id``)."""
    order = {int(r): i for i, r in enumerate(run_id)}
    rows = []
    for r in df["run_id"].to_numpy():
        rows.append(order[int(r)])
    return energy_ratio[np.array(rows, dtype=int), :]
