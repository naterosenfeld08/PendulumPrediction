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


def weighted_rmse(
    metrics: dict[str, object],
    *,
    horizon_weights: list[float] | tuple[float, ...],
) -> float:
    """Compute weighted RMSE from ``evaluate_forecasts`` output."""
    per_h = metrics["per_horizon"]
    weights = np.asarray(horizon_weights, dtype=np.float64)
    if len(per_h) != len(weights):
        raise ValueError("Number of horizon weights must match per_horizon entries.")
    if np.sum(weights) <= 0.0:
        raise ValueError("Horizon weights must have positive sum.")
    weights = weights / np.sum(weights)
    rmses = np.asarray([row["rmse"] for row in per_h], dtype=np.float64)
    return float(np.sum(weights * rmses))


def split_gap_report(
    train_metrics: dict[str, object],
    val_metrics: dict[str, object],
    test_metrics: dict[str, object] | None = None,
    *,
    gap_threshold: float = 0.25,
) -> dict[str, object]:
    """Compute overfitting diagnostics from split metrics."""
    tr_rmse = float(train_metrics["aggregate"]["rmse_mean"])
    va_rmse = float(val_metrics["aggregate"]["rmse_mean"])
    te_rmse = float(test_metrics["aggregate"]["rmse_mean"]) if test_metrics else None
    train_val_gap = va_rmse - tr_rmse
    val_test_gap = None if te_rmse is None else te_rmse - va_rmse
    flags = {
        "train_val_overfit": bool(tr_rmse > 0 and train_val_gap / tr_rmse > gap_threshold),
        "val_test_drop": bool(
            te_rmse is not None and va_rmse > 0 and (te_rmse - va_rmse) / va_rmse > gap_threshold
        ),
    }
    return {
        "train_rmse_mean": tr_rmse,
        "val_rmse_mean": va_rmse,
        "test_rmse_mean": te_rmse,
        "train_val_gap": float(train_val_gap),
        "val_test_gap": None if val_test_gap is None else float(val_test_gap),
        "flags": flags,
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


def plot_gap_curves(
    train_metrics: dict[str, object],
    val_metrics: dict[str, object],
    test_metrics: dict[str, object] | None,
    output_path: Path,
) -> None:
    """Plot RMSE curves per split to inspect overfitting by horizon."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    tr = train_metrics["per_horizon"]
    va = val_metrics["per_horizon"]
    h = [m["horizon"] for m in tr]
    ax.plot(h, [m["rmse"] for m in tr], marker="o", label="train")
    ax.plot(h, [m["rmse"] for m in va], marker="o", label="val")
    if test_metrics is not None:
        te = test_metrics["per_horizon"]
        ax.plot(h, [m["rmse"] for m in te], marker="o", label="test")
    ax.set_title("RMSE by Horizon Across Splits")
    ax.set_xlabel("Horizon (steps)")
    ax.set_ylabel("RMSE")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_residual_histogram(
    y_true: NDArray[np.floating],
    y_pred: NDArray[np.floating],
    output_path: Path,
) -> None:
    """Plot residual histogram for heteroscedasticity rough check."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    residual = (y_true - y_pred).ravel()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(residual, bins=50, alpha=0.85)
    ax.set_title("Residual Distribution")
    ax.set_xlabel("Residual")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
