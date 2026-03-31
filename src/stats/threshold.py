"""Logistic regression for marginal chaos probability vs initial θ₁."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import brentq
from sklearn.linear_model import LogisticRegression


def _degenerate_threshold_info(df: pd.DataFrame, config: Mapping[str, Any]) -> dict[str, Any]:
    """When every run has the same chaos label, logistic regression is undefined."""
    st = config["statistics"]
    conf = float(st["confidence_level"])
    p_star = 1.0 - conf
    y = df["is_chaotic"].astype(bool)
    theta1_min = float(config["parameters"]["theta1"][0])
    theta1_max = float(config["parameters"]["theta1"][1])
    theta_grid = np.linspace(theta1_min, theta1_max, 512, dtype=np.float64)
    all_chaotic = bool(y.all())
    p_const = 1.0 if all_chaotic else 0.0
    p_grid = np.full_like(theta_grid, p_const, dtype=np.float64)
    if p_const >= p_star:
        threshold_angle = theta1_min
    else:
        threshold_angle = theta1_max
    return {
        "threshold_angle": float(threshold_angle),
        "threshold_angle_deg": float(np.degrees(threshold_angle)),
        "confidence_level": conf,
        "p_target": p_star,
        "logistic_model": None,
        "theta_grid": theta_grid,
        "p_chaotic_grid": p_grid,
    }


def find_chaos_threshold(df: pd.DataFrame, config: Mapping[str, Any]) -> dict[str, Any]:
    """Infer θ₁ where ``P(chaotic | θ₁) = 1 - confidence_level`` from a logistic fit.

    Only ``theta1`` is used as a covariate (other parameters are marginalized in the
    sense that their influence is folded into the binary labels).
    """
    st = config["statistics"]
    conf = float(st["confidence_level"])
    p_star = 1.0 - conf

    y = df["is_chaotic"].astype(int).to_numpy()
    if np.unique(y).size < 2:
        return _degenerate_threshold_info(df, config)

    X = df[["theta1"]].to_numpy(dtype=np.float64)

    clf = LogisticRegression(solver="lbfgs", max_iter=2000, random_state=0)
    clf.fit(X, y)

    theta_grid = np.linspace(
        float(config["parameters"]["theta1"][0]),
        float(config["parameters"]["theta1"][1]),
        512,
    )
    probs = clf.predict_proba(theta_grid.reshape(-1, 1))[:, 1]

    def p_chaotic(theta: float) -> float:
        return float(clf.predict_proba(np.array([[theta]], dtype=np.float64))[0, 1])

    lo_b = float(config["parameters"]["theta1"][0])
    hi_b = float(config["parameters"]["theta1"][1])
    f_lo = p_chaotic(lo_b) - p_star
    f_hi = p_chaotic(hi_b) - p_star

    threshold_angle: float
    if f_lo == 0.0:
        threshold_angle = lo_b
    elif f_hi == 0.0:
        threshold_angle = hi_b
    elif f_lo * f_hi < 0:
        threshold_angle = float(brentq(lambda t: p_chaotic(t) - p_star, lo_b, hi_b))
    elif f_lo > 0 and f_hi > 0:
        threshold_angle = lo_b
    elif f_lo < 0 and f_hi < 0:
        threshold_angle = hi_b
    else:
        mid = 0.5 * (lo_b + hi_b)
        j = int(np.argmin(np.abs(probs - p_star)))
        threshold_angle = float(theta_grid[j])

    return {
        "threshold_angle": threshold_angle,
        "threshold_angle_deg": float(np.degrees(threshold_angle)),
        "confidence_level": conf,
        "p_target": p_star,
        "logistic_model": clf,
        "theta_grid": theta_grid,
        "p_chaotic_grid": probs,
    }


def chaos_probability_vs_theta(
    clf: LogisticRegression,
    theta: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Vectorized ``P(chaotic | θ₁)`` for plotting."""
    return clf.predict_proba(theta.reshape(-1, 1).astype(np.float64))[:, 1]
