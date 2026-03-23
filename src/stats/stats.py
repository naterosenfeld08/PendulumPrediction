"""Gaussian Process Regression for energy-ratio variance with bootstrap confidence bands."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel

# Order matches project spec: (θ₁₀, θ₂₀, m1, m2, L1, L2, ω₁₀, ω₂₀)
FEATURE_ORDER: list[str] = [
    "theta1",
    "theta2",
    "m1",
    "m2",
    "L1",
    "L2",
    "omega1",
    "omega2",
]


def fit_model(df: pd.DataFrame, config: Mapping[str, Any]) -> dict[str, Any]:
    """Fit a GPR mapping initial conditions to ``energy_ratio_variance``.

    For very large ensembles (e.g. N ≳ 1000), fitting and especially bootstrap
    prediction intervals become expensive. Set ``statistics.gpr_fit_max_samples``
    in ``config.yaml`` to cap the number of training rows (e.g. 500); predictions
    can still be evaluated on the full table afterward.
    """
    st = config["statistics"]
    X_full = df[FEATURE_ORDER].to_numpy(dtype=np.float64)
    y_full = df["energy_ratio_variance"].to_numpy(dtype=np.float64)

    max_samples = st.get("gpr_fit_max_samples")
    rng = np.random.default_rng(int(st.get("gpr_random_state", 0)))

    if max_samples is not None and len(df) > int(max_samples):
        idx = rng.choice(len(df), size=int(max_samples), replace=False)
        X_train = X_full[idx]
        y_train = y_full[idx]
    else:
        X_train = X_full.copy()
        y_train = y_full.copy()

    kernel = C(1.0, (1e-3, 1e3)) * RBF(
        length_scale=np.ones(len(FEATURE_ORDER)),
        length_scale_bounds=(1e-2, 1e3),
    ) + WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-10, 1.0))

    gp = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=3,
        alpha=1e-8,
        normalize_y=True,
        random_state=int(st.get("gpr_random_state", 0)),
    )
    gp.fit(X_train, y_train)

    return {
        "gpr": gp,
        "X_train": X_train,
        "y_train": y_train,
        "feature_order": FEATURE_ORDER,
    }


def predict_with_ci(
    model: Mapping[str, Any],
    X: NDArray[np.floating] | pd.DataFrame,
    config: Mapping[str, Any],
    *,
    n_bootstrap: int | None = None,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Bootstrap predictive mean and equal-tailed CI for variance predictions.

    Parameters
    ----------
    n_bootstrap
        If set, overrides ``statistics.bootstrap_iterations`` (for faster figures, etc.).

    Returns
    -------
    mean_prediction, lower_ci, upper_ci
        Each array shaped ``(n_points,)``.
    """
    if isinstance(X, pd.DataFrame):
        X_arr = X[model["feature_order"]].to_numpy(dtype=np.float64)
    else:
        X_arr = np.asarray(X, dtype=np.float64)

    st = config["statistics"]
    n_boot = int(n_bootstrap) if n_bootstrap is not None else int(st["bootstrap_iterations"])
    conf = float(st["confidence_level"])
    alpha_tail = (1.0 - conf) / 2.0

    X_train = model["X_train"]
    y_train = model["y_train"]
    gpr0 = model["gpr"]

    rng = np.random.default_rng(int(st.get("gpr_random_state", 0)) + 911)

    preds = np.empty((n_boot, X_arr.shape[0]), dtype=np.float64)
    n_tr = X_train.shape[0]

    for b in range(n_boot):
        idx = rng.integers(0, n_tr, size=n_tr)
        Xb = X_train[idx]
        yb = y_train[idx]
        gpr = GaussianProcessRegressor(
            kernel=gpr0.kernel_,
            alpha=gpr0.alpha,
            normalize_y=gpr0.normalize_y,
            optimizer=None,
            n_restarts_optimizer=0,
            random_state=b,
        )
        gpr.fit(Xb, yb)
        preds[b, :] = gpr.predict(X_arr)

    mean_prediction = np.mean(preds, axis=0)
    lower_ci = np.quantile(preds, alpha_tail, axis=0)
    upper_ci = np.quantile(preds, 1.0 - alpha_tail, axis=0)
    return mean_prediction, lower_ci, upper_ci
