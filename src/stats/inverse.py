"""Inverse map: maximum θ₁ such that GPR upper-CI variance stays below a target."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq

from stats.stats import FEATURE_ORDER, predict_with_ci


def _reference_feature_matrix(
    model: Mapping[str, Any],
    theta1_values: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Rows are training medians with ``theta1`` replaced by each value in ``theta1_values``."""
    med = np.median(model["X_train"], axis=0)
    X = np.tile(med, (len(theta1_values), 1))
    idx = FEATURE_ORDER.index("theta1")
    X[:, idx] = theta1_values
    return X


def upper_ci_theta1(
    model: Mapping[str, Any],
    theta1: float,
    config: Mapping[str, Any],
    *,
    n_bootstrap: int | None = None,
) -> float:
    """GPR bootstrap upper confidence bound on variance at fixed ``theta1`` (other dims median)."""
    X = _reference_feature_matrix(model, np.array([theta1], dtype=np.float64))
    _, _, upper = predict_with_ci(model, X, config, n_bootstrap=n_bootstrap)
    return float(upper[0])


def angle_from_variance_target(
    model: Mapping[str, Any],
    target_variance: float,
    config: Mapping[str, Any],
) -> float:
    """Largest θ₁ whose GPR upper-CI (marginal in θ₁, other features at medians) hits the target.

    Builds a θ₁ grid, evaluates the upper CI with ``inverse.bootstrap_iterations`` (to avoid
    thousands of nested bootstrap fits inside a root finder), then refines the crossing with
    ``brentq`` on a linear interpolant.
    """
    inv = config.get("inverse", {})
    lo_b = float(config["parameters"]["theta1"][0])
    hi_b = float(config["parameters"]["theta1"][1])
    n_grid = int(inv.get("theta_grid_points", 96))
    n_boot = int(inv.get("bootstrap_iterations", 150))

    th_grid = np.linspace(lo_b, hi_b, n_grid, dtype=np.float64)
    X = _reference_feature_matrix(model, th_grid)
    _, _, upper = predict_with_ci(model, X, config, n_bootstrap=n_boot)

    g = upper - float(target_variance)

    if float(g[-1]) <= 0.0:
        return float(hi_b)
    if float(g[0]) >= 0.0:
        return float(lo_b)

    for i in range(len(g) - 1):
        if float(g[i] * g[i + 1]) <= 0.0:

            def upper_interp(t: float) -> float:
                return float(np.interp(t, th_grid, upper))

            def gap(t: float) -> float:
                return upper_interp(t) - float(target_variance)

            return float(brentq(gap, float(th_grid[i]), float(th_grid[i + 1])))

    j = int(np.argmin(np.abs(g)))
    return float(th_grid[j])
