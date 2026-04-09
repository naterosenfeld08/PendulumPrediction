"""Model zoo and wrappers for multi-horizon energy forecasting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:  # Optional dependency
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover - optional
    XGBRegressor = None


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
    estimator: Any

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


def build_ridge_model(horizons: tuple[int, ...], alpha: float = 1.0) -> SklearnForecastModel:
    estimator = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=float(alpha))),
        ]
    )
    return SklearnForecastModel(horizons=horizons, estimator=estimator)


def build_random_forest_model(
    horizons: tuple[int, ...],
    *,
    n_estimators: int = 200,
    max_depth: int | None = None,
    random_state: int = 0,
) -> SklearnForecastModel:
    estimator = MultiOutputRegressor(
        RandomForestRegressor(
            n_estimators=int(n_estimators),
            max_depth=max_depth,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        )
    )
    return SklearnForecastModel(horizons=horizons, estimator=estimator)


def build_gradient_boosting_model(
    horizons: tuple[int, ...],
    *,
    n_estimators: int = 250,
    learning_rate: float = 0.05,
    max_depth: int = 3,
    random_state: int = 0,
) -> SklearnForecastModel:
    estimator = MultiOutputRegressor(
        GradientBoostingRegressor(
            n_estimators=int(n_estimators),
            learning_rate=float(learning_rate),
            max_depth=int(max_depth),
            random_state=random_state,
        )
    )
    return SklearnForecastModel(horizons=horizons, estimator=estimator)


def build_mlp_model(
    horizons: tuple[int, ...],
    *,
    hidden_layer_sizes: tuple[int, ...] = (128, 64),
    learning_rate_init: float = 1e-3,
    alpha: float = 1e-4,
    max_iter: int = 300,
    random_state: int = 0,
) -> SklearnForecastModel:
    estimator = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "model",
                MLPRegressor(
                    hidden_layer_sizes=hidden_layer_sizes,
                    activation="relu",
                    learning_rate_init=float(learning_rate_init),
                    alpha=float(alpha),
                    early_stopping=True,
                    max_iter=int(max_iter),
                    random_state=random_state,
                ),
            ),
        ]
    )
    return SklearnForecastModel(horizons=horizons, estimator=estimator)


def build_xgboost_model(
    horizons: tuple[int, ...],
    *,
    n_estimators: int = 250,
    max_depth: int = 5,
    learning_rate: float = 0.05,
    random_state: int = 0,
) -> SklearnForecastModel:
    if XGBRegressor is None:  # pragma: no cover - optional dependency
        raise RuntimeError("xgboost is not installed.")
    estimator = MultiOutputRegressor(
        XGBRegressor(
            n_estimators=int(n_estimators),
            max_depth=int(max_depth),
            learning_rate=float(learning_rate),
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=-1,
        )
    )
    return SklearnForecastModel(horizons=horizons, estimator=estimator)


def model_names() -> list[str]:
    names = [
        "persistence",
        "linear",
        "ridge",
        "random_forest",
        "gradient_boosting",
        "mlp_small",
        "mlp_medium",
    ]
    if XGBRegressor is not None:
        names.append("xgboost")
    return names


def sample_hyperparameters(model_name: str, rng: np.random.Generator) -> dict[str, Any]:
    """Randomly sample hyperparameters for model search."""
    if model_name in {"persistence", "linear"}:
        return {}
    if model_name == "ridge":
        return {"alpha": float(10 ** rng.uniform(-3.0, 2.0))}
    if model_name == "random_forest":
        return {
            "n_estimators": int(rng.integers(120, 360)),
            "max_depth": int(rng.integers(4, 18)),
        }
    if model_name == "gradient_boosting":
        return {
            "n_estimators": int(rng.integers(100, 360)),
            "learning_rate": float(10 ** rng.uniform(-2.0, -0.2)),
            "max_depth": int(rng.integers(2, 6)),
        }
    if model_name == "mlp_small":
        return {
            "hidden_layer_sizes": (64, 32),
            "learning_rate_init": float(10 ** rng.uniform(-4.0, -2.2)),
            "alpha": float(10 ** rng.uniform(-6.0, -2.0)),
            "max_iter": 400,
        }
    if model_name == "mlp_medium":
        return {
            "hidden_layer_sizes": (128, 64, 32),
            "learning_rate_init": float(10 ** rng.uniform(-4.0, -2.2)),
            "alpha": float(10 ** rng.uniform(-6.0, -2.0)),
            "max_iter": 500,
        }
    if model_name == "xgboost":
        return {
            "n_estimators": int(rng.integers(120, 360)),
            "max_depth": int(rng.integers(3, 8)),
            "learning_rate": float(10 ** rng.uniform(-2.0, -0.2)),
        }
    raise ValueError(f"Unknown model for hyperparameter sampling: {model_name}")


def build_model(
    model_name: str,
    *,
    horizons: tuple[int, ...],
    random_state: int,
    hyperparams: dict[str, Any] | None = None,
):
    """Build a model instance by name."""
    p = hyperparams or {}
    if model_name == "persistence":
        return PersistenceBaseline(horizons=horizons)
    if model_name == "linear":
        return build_linear_model(horizons=horizons)
    if model_name == "ridge":
        return build_ridge_model(horizons=horizons, alpha=float(p.get("alpha", 1.0)))
    if model_name == "random_forest":
        return build_random_forest_model(
            horizons=horizons,
            n_estimators=int(p.get("n_estimators", 200)),
            max_depth=int(p.get("max_depth", 10)),
            random_state=random_state,
        )
    if model_name == "gradient_boosting":
        return build_gradient_boosting_model(
            horizons=horizons,
            n_estimators=int(p.get("n_estimators", 250)),
            learning_rate=float(p.get("learning_rate", 0.05)),
            max_depth=int(p.get("max_depth", 3)),
            random_state=random_state,
        )
    if model_name in {"mlp_small", "mlp_medium"}:
        return build_mlp_model(
            horizons=horizons,
            hidden_layer_sizes=tuple(p.get("hidden_layer_sizes", (128, 64))),
            learning_rate_init=float(p.get("learning_rate_init", 1e-3)),
            alpha=float(p.get("alpha", 1e-4)),
            max_iter=int(p.get("max_iter", 300)),
            random_state=random_state,
        )
    if model_name == "xgboost":
        return build_xgboost_model(
            horizons=horizons,
            n_estimators=int(p.get("n_estimators", 250)),
            max_depth=int(p.get("max_depth", 5)),
            learning_rate=float(p.get("learning_rate", 0.05)),
            random_state=random_state,
        )
    raise ValueError(f"Unknown model name: {model_name}")
