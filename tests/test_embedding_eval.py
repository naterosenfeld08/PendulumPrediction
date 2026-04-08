"""Evaluation metric contract tests for multi-horizon forecasts."""

from __future__ import annotations

import numpy as np

from experiments.evaluation import evaluate_forecasts


def test_evaluate_forecasts_perfect_predictions() -> None:
    y_true = np.array(
        [
            [[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]],
            [[0.5, 1.5, 2.5], [0.75, 1.75, 2.75]],
        ],
        dtype=np.float64,
    )
    y_pred = y_true.copy()
    out = evaluate_forecasts(y_true, y_pred, horizons=(1, 5))
    for m in out["per_horizon"]:
        assert m["mae"] == 0.0
        assert m["rmse"] == 0.0
        assert m["r2"] == 1.0


def test_evaluate_forecasts_shape_mismatch_raises() -> None:
    y_true = np.zeros((3, 2, 3), dtype=np.float64)
    y_pred = np.zeros((3, 2, 2), dtype=np.float64)
    try:
        evaluate_forecasts(y_true, y_pred, horizons=(1, 3))
    except ValueError as exc:
        assert "same shape" in str(exc)
    else:
        raise AssertionError("Expected ValueError for mismatched target shapes.")
