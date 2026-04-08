"""Baseline models for multi-horizon energy forecasting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _flatten_targets(y: NDArray[np.floating]) -> NDArray[np.floating]:
    n, h, c = y.shape
    return y.reshape(n, h * c)


def _unflatten_targets(y_flat: NDArray[np.floating], horizons: int) -> NDArray[np.floating]:
    n, hc = y_flat.shape
    channels = hc // horizons
    return y_flat.reshape(n, horizons, channels)


@dataclass
class PersistenceBaseline:
    """Predict all future horizons as the last observed energy vector."""

    horizons: tuple[int, ...]

    def fit(self, X: NDArray[np.floating], y: NDArray[np.floating]) -> "PersistenceBaseline":
        _ = X, y
        return self

    def predict(
        self,
        X: NDArray[np.floating],
        *,
        last_energy: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        _ = X
        return np.repeat(last_energy[:, None, :], repeats=len(self.horizons), axis=1)


@dataclass
class SklearnForecastModel:
    """Wrap a sklearn regressor for tensor targets."""

    horizons: tuple[int, ...]
    estimator: Pipeline

    def fit(self, X: NDArray[np.floating], y: NDArray[np.floating]) -> "SklearnForecastModel":
        self.estimator.fit(X, _flatten_targets(y))
        return self

    def predict(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        y_flat = np.asarray(self.estimator.predict(X), dtype=np.float64)
        return _unflatten_targets(y_flat, horizons=len(self.horizons))


def build_linear_model(horizons: tuple[int, ...]) -> SklearnForecastModel:
    estimator = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )
    return SklearnForecastModel(horizons=horizons, estimator=estimator)


def build_mlp_model(
    horizons: tuple[int, ...],
    random_state: int = 0,
) -> SklearnForecastModel:
    estimator = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "model",
                MLPRegressor(
                    hidden_layer_sizes=(128, 64),
                    activation="relu",
                    learning_rate_init=1e-3,
                    max_iter=300,
                    random_state=random_state,
                ),
            ),
        ]
    )
    return SklearnForecastModel(horizons=horizons, estimator=estimator)
