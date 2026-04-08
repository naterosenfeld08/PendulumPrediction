"""Evaluation helpers for multi-horizon forecasts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


def _r2_score(y_true: NDArray[np.floating], y_pred: NDArray[np.floating]) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    y_mean = float(np.mean(y_true))
    ss_tot = float(np.sum((y_true - y_mean) ** 2))
    if ss_tot == 0.0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def evaluate_forecasts(
    y_true: NDArray[np.floating],
    y_pred: NDArray[np.floating],
    horizons: tuple[int, ...],
) -> dict[str, object]:
    """Compute per-horizon MAE/RMSE/R2 and macro averages."""
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    if y_true.ndim != 3:
        raise ValueError("Expected target tensor shape (n_samples, n_horizons, n_channels).")

    horizon_metrics: list[dict[str, float]] = []
    for i, h in enumerate(horizons):
        yt = y_true[:, i, :].ravel()
        yp = y_pred[:, i, :].ravel()
        mae = float(np.mean(np.abs(yt - yp)))
        rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
        r2 = float(_r2_score(yt, yp))
        horizon_metrics.append({"horizon": int(h), "mae": mae, "rmse": rmse, "r2": r2})

    return {
        "per_horizon": horizon_metrics,
        "aggregate": {
            "mae_mean": float(np.mean([m["mae"] for m in horizon_metrics])),
            "rmse_mean": float(np.mean([m["rmse"] for m in horizon_metrics])),
            "r2_mean": float(np.mean([m["r2"] for m in horizon_metrics])),
        },
    }


def save_metrics(metrics: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def plot_metric_curves(metrics_by_model: dict[str, dict[str, object]], output_path: Path) -> None:
    """Plot MAE and RMSE vs horizon for each model."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for model_name, m in metrics_by_model.items():
        ph = m["per_horizon"]
        horizons = [row["horizon"] for row in ph]
        maes = [row["mae"] for row in ph]
        rmses = [row["rmse"] for row in ph]
        axes[0].plot(horizons, maes, marker="o", label=model_name)
        axes[1].plot(horizons, rmses, marker="o", label=model_name)

    axes[0].set_title("MAE vs Horizon")
    axes[1].set_title("RMSE vs Horizon")
    axes[0].set_xlabel("Horizon (steps)")
    axes[1].set_xlabel("Horizon (steps)")
    axes[0].set_ylabel("Error")
    axes[1].set_ylabel("Error")
    axes[0].grid(True, alpha=0.3)
    axes[1].grid(True, alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
