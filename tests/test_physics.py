"""Physics validation: energy conservation, small-angle period, chaotic separation & MLE."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.signal import find_peaks

from ensemble.lyapunov import compute_mle, save_delta_curve, separation_timeseries
from physics.energy import compute_energy_timeseries, scaled_max_energy_drift
from physics.integrator import integrate
from physics.pendulum import dstate_dt


@pytest.fixture(scope="module")
def base_config() -> dict:
    cfg_path = Path(__file__).resolve().parents[1] / "config.yaml"
    with cfg_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_energy_conservation_hard_assertion(base_config: dict) -> None:
    """Integrate 30 s; ``compute_energy_timeseries`` must assert on energy drift."""
    params = {"m1": 1.0, "m2": 1.0, "L1": 1.0, "L2": 1.0, "g": 9.81}
    state0 = np.array([0.3, 0.0, 0.2, 0.0], dtype=np.float64)
    t, y = integrate(dstate_dt, state0, params, base_config)
    energies = compute_energy_timeseries(t, y, params, base_config)
    e0 = float(energies["E_total"][0])
    drift = scaled_max_energy_drift(energies["E_total"], e0, params)
    assert drift <= float(base_config["integration"]["energy_drift_max_relative"])


def test_small_angle_period_arm1(base_config: dict) -> None:
    """Small angles: θ₁ oscillation matches SHO 2π√(L₁/g) when the second bob is negligible.

    Equal-mass symmetric coupling shifts the in-phase period; a tiny *m₂* isolates arm 1
    as a simple pendulum while keeping the specified small initial angles.
    """
    params = {"m1": 1.0, "m2": 1e-9, "L1": 1.0, "L2": 1.0, "g": 9.81}
    state0 = np.array([0.05, 0.0, 0.05, 0.0], dtype=np.float64)
    t, y = integrate(dstate_dt, state0, params, base_config)
    theta1 = y[:, 0]
    peaks, _ = find_peaks(theta1, prominence=0.001)
    assert len(peaks) >= 3, "Need several peaks to estimate period"
    periods = np.diff(t[peaks])
    T_mean = float(np.mean(periods))
    T_sho = 2.0 * np.pi * np.sqrt(params["L1"] / params["g"])
    assert abs(T_mean - T_sho) / T_sho < 0.05


def test_known_chaotic_mle_and_separation_grows(base_config: dict) -> None:
    """Large angles: δ(t) should grow (log plot not flat/down); MLE > 0."""
    params = {"m1": 1.0, "m2": 1.0, "L1": 1.0, "L2": 1.0, "g": 9.81}
    state0 = np.array([2.5, 0.0, 2.5, 0.0], dtype=np.float64)
    t, y = integrate(dstate_dt, state0, params, base_config)

    t_sep, delta = separation_timeseries(state0, params, base_config)
    out_dir = Path(__file__).resolve().parents[1] / "data" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_path = out_dir / "delta_chaotic_validation.png"
    save_delta_curve(
        t_sep,
        delta,
        str(plot_path),
        title=r"δ(t) known-chaotic: θ₁₀=θ₂₀=2.5 rad, m₁=m₂=1, L₁=L₂=1",
    )

    # Perturbation must not be trivially flat or decaying overall
    i_mid = len(delta) // 2
    assert float(delta[-1]) > float(delta[i_mid]), (
        "δ(t) should increase in the second half for this chaotic reference; "
        "check θ₁-only perturbation and integration."
    )
    assert float(delta[-1]) > float(delta[0]) * 1e3, (
        "δ(t) should amplify strongly by t_end; flat δ implies wrong neighbor trajectory."
    )

    mle = compute_mle(t, y, params, base_config)
    assert mle > 0.0, f"Expected positive MLE for chaotic case; got {mle}"
