"""GPR sanity, logistic monotonicity, inverse consistency."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from stats.inverse import angle_from_variance_target, upper_ci_theta1
from stats.stats import FEATURE_ORDER, fit_model, predict_with_ci
from stats.threshold import find_chaos_threshold


@pytest.fixture(scope="module")
def base_config() -> dict:
    cfg_path = Path(__file__).resolve().parents[1] / "config.yaml"
    with cfg_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture
def stats_config(base_config: dict) -> dict:
    c = copy.deepcopy(base_config)
    c["statistics"]["bootstrap_iterations"] = 25
    c["statistics"]["figure_bootstrap_iterations"] = 25
    c["inverse"]["bootstrap_iterations"] = 25
    c["inverse"]["theta_grid_points"] = 48
    return c


def test_gpr_synthetic_theta_relation(stats_config: dict) -> None:
    """GPR recovers a smooth variance-like function of θ₁ within 10 % on holdout."""
    rng = np.random.default_rng(7)
    n = 120
    theta1 = rng.uniform(0.2, 2.5, size=n)
    rows = []
    for t in theta1:
        rows.append(
            {
                "theta1": float(t),
                "theta2": 0.5,
                "m1": 1.0,
                "m2": 1.0,
                "L1": 1.0,
                "L2": 1.0,
                "omega1": 0.0,
                "omega2": 0.0,
                "energy_ratio_variance": float((0.35 * t) ** 2 + 0.002 * rng.standard_normal()),
                "mle": 0.0,
                "energy_ratio_mean": 0.5,
                "is_chaotic": False,
                "run_id": 0,
            }
        )
    df = pd.DataFrame(rows)
    model = fit_model(df, stats_config)
    hold = df.iloc[::6].copy()
    pred, _, _ = predict_with_ci(
        model,
        hold[FEATURE_ORDER],
        stats_config,
        n_bootstrap=20,
    )
    true_y = hold["energy_ratio_variance"].to_numpy()
    rel = np.abs(pred - true_y) / np.maximum(np.abs(true_y), 1e-6)
    assert float(np.mean(rel)) < 0.10


def test_logistic_monotonic_in_theta1(base_config: dict) -> None:
    """Marginal P(chaotic | θ₁) from logistic fit increases with θ₁ on a monotone synthetic set."""
    c = copy.deepcopy(base_config)
    rng = np.random.default_rng(3)
    n = 400
    theta1 = np.sort(rng.uniform(0.05, 3.0, size=n))
    is_chaotic = theta1 > 1.35
    df = pd.DataFrame(
        {
            "theta1": theta1,
            "theta2": 0.5,
            "m1": 1.0,
            "m2": 1.0,
            "L1": 1.0,
            "L2": 1.0,
            "omega1": 0.0,
            "omega2": 0.0,
            "energy_ratio_variance": 0.01 * np.ones(n),
            "mle": np.where(is_chaotic, 1.0, -1.0),
            "energy_ratio_mean": 0.5,
            "is_chaotic": is_chaotic,
            "run_id": np.arange(n),
        }
    )
    out = find_chaos_threshold(df, c)
    clf = out["logistic_model"]
    grid = np.linspace(float(c["parameters"]["theta1"][0]), float(c["parameters"]["theta1"][1]), 200)
    p = clf.predict_proba(grid.reshape(-1, 1))[:, 1]
    assert np.all(np.diff(p) >= -1e-5), "P(chaotic | θ₁) should be monotone non-decreasing"


def test_inverse_consistency_near_threshold(stats_config: dict) -> None:
    """Using upper-CI variance at the logistic threshold as target recovers ~that angle."""
    rng = np.random.default_rng(11)
    n = 200
    theta1 = rng.uniform(0.1, 3.0, size=n)
    var = 0.02 + 0.08 * (theta1 / np.pi) ** 2 + 0.01 * rng.standard_normal(size=n)
    var = np.clip(var, 0.001, None)
    chaotic = theta1 > 1.55 + 0.05 * rng.standard_normal(size=n)
    df = pd.DataFrame(
        {
            "theta1": theta1,
            "theta2": rng.uniform(0.1, 1.0, size=n),
            "m1": rng.uniform(0.8, 1.2, size=n),
            "m2": rng.uniform(0.8, 1.2, size=n),
            "L1": rng.uniform(0.8, 1.2, size=n),
            "L2": rng.uniform(0.8, 1.2, size=n),
            "omega1": rng.uniform(0.0, 0.3, size=n),
            "omega2": rng.uniform(0.0, 0.3, size=n),
            "energy_ratio_variance": var,
            "mle": np.where(chaotic, 0.5, -0.5),
            "energy_ratio_mean": 0.5,
            "is_chaotic": chaotic,
            "run_id": np.arange(n),
        }
    )
    model = fit_model(df, stats_config)
    th_info = find_chaos_threshold(df, stats_config)
    th = float(th_info["threshold_angle"])
    target = upper_ci_theta1(
        model,
        th,
        stats_config,
        n_bootstrap=int(stats_config["inverse"]["bootstrap_iterations"]),
    )
    recovered = angle_from_variance_target(model, target, stats_config)
    assert abs(recovered - th) < 0.18, (recovered, th)
