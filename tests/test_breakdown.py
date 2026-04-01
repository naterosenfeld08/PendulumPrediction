"""OOF prediction breakdown: alignment and GPR loop smoke test."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from stats.breakdown import align_timeseries_to_dataframe, run_prediction_breakdown_oof


@pytest.fixture
def breakdown_config() -> dict:
    cfg_path = Path(__file__).resolve().parents[1] / "config.yaml"
    c = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    c = copy.deepcopy(c)
    c["prediction"]["cv_folds"] = 4
    c["prediction"]["n_time_samples"] = 6
    c["prediction"]["interval_method"] = "analytic"
    return c


def test_align_timeseries_to_dataframe_permutation() -> None:
    rng = np.random.default_rng(1)
    n, k = 12, 5
    er = rng.standard_normal((n, k))
    df = pd.DataFrame(
        {
            "run_id": np.arange(n, dtype=np.int64),
            "theta1": rng.random(n),
            "theta2": rng.random(n),
            "m1": rng.random(n),
            "m2": rng.random(n),
            "L1": rng.random(n),
            "L2": rng.random(n),
            "omega1": rng.random(n),
            "omega2": rng.random(n),
            "mle": rng.standard_normal(n),
            "is_chaotic": np.zeros(n, dtype=bool),
            "energy_ratio_mean": np.full(n, 0.5),
            "energy_ratio_variance": np.full(n, 0.01),
        }
    )
    rid_perm = np.arange(n - 1, -1, -1, dtype=np.int64)
    er_perm = np.stack([er[int(r), :] for r in rid_perm], axis=0)
    aligned = align_timeseries_to_dataframe(df, rid_perm, er_perm)
    np.testing.assert_allclose(aligned, er)


def test_run_prediction_breakdown_oof_shapes(breakdown_config: dict) -> None:
    rng = np.random.default_rng(2)
    n, k = 24, 6
    cols = ["theta1", "theta2", "m1", "m2", "L1", "L2", "omega1", "omega2"]
    X = rng.uniform(0.1, 1.0, size=(n, len(cols)))
    df = pd.DataFrame(X, columns=cols)
    df["run_id"] = np.arange(n, dtype=np.int64)
    df["mle"] = rng.standard_normal(n)
    df["is_chaotic"] = rng.random(n) > 0.5
    df["energy_ratio_mean"] = 0.5
    df["energy_ratio_variance"] = 0.01
    er = np.clip(rng.standard_normal((n, k)) * 0.08 + 0.5, 0.01, 0.99)
    t_sample = np.linspace(0.0, 5.0, k, dtype=np.float64)

    out = run_prediction_breakdown_oof(df, t_sample, er, breakdown_config)
    assert out["oof_width"].shape == (n, k)
    assert out["median_interval_width"].shape == (k,)
    assert out["fraction_outside_ci"].shape == (k,)
    assert out["t_breakdown"].shape == (n,)
