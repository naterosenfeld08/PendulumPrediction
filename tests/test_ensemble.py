"""LHS coverage and parallel reproducibility."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from ensemble.ensemble import run_ensemble
from ensemble.sampler import sample_parameters


@pytest.fixture(scope="module")
def base_config() -> dict:
    cfg_path = Path(__file__).resolve().parents[1] / "config.yaml"
    with cfg_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture
def fast_ensemble_config(base_config: dict) -> dict:
    c = copy.deepcopy(base_config)
    c["ensemble"]["n_pendulums"] = 28
    c["ensemble"]["checkpoint_every"] = 7
    c["ensemble"]["n_jobs"] = 2
    c["integration"]["n_steps"] = 100
    c["integration"]["t_span"] = [0.0, 5.0]
    c["lyapunov"]["t_lyapunov"] = 4.0
    return c


def test_lhs_parameter_ranges_span(fast_ensemble_config: dict) -> None:
    """Sampled sets should cover the configured boxes (not collapse to a corner)."""
    rng = np.random.default_rng(2025)
    n = 40
    samples = sample_parameters(n, fast_ensemble_config, rng)
    keys = ["m1", "m2", "L1", "L2", "theta1", "theta2", "omega1", "omega2"]
    p = fast_ensemble_config["parameters"]
    for k in keys:
        lo, hi = float(p[k][0]), float(p[k][1])
        span = hi - lo
        vals = [s[k] for s in samples]
        assert min(vals) <= lo + 0.12 * span, k
        assert max(vals) >= hi - 0.12 * span, k
        assert max(vals) - min(vals) > 0.55 * span, k


def test_parallel_reproducibility_same_seed(  # n_jobs=1 for bitwise-identical floats
    fast_ensemble_config: dict,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two ensemble runs with the same RNG seed must yield identical tables."""
    import ensemble.ensemble as ens_mod

    monkeypatch.setattr(ens_mod, "_project_root", lambda: tmp_path)

    c = copy.deepcopy(fast_ensemble_config)
    c["ensemble"]["n_jobs"] = 1

    rng_a = np.random.default_rng(9991)
    df_a = run_ensemble(c, rng=rng_a)
    rng_b = np.random.default_rng(9991)
    df_b = run_ensemble(c, rng=rng_b)

    pd.testing.assert_frame_equal(
        df_a.reset_index(drop=True),
        df_b.reset_index(drop=True),
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )
