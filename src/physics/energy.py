"""Kinetic and potential energy timeseries; hard energy conservation check."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray


def compute_energy_timeseries(
    t_array: NDArray[np.floating],
    state_array: NDArray[np.floating],
    params: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, NDArray[np.floating]]:
    """Compute per-bob energies and KE₂/KE_total at each stored time.

    Convention matches ``pendulum.dstate_dt``: both angles from the downward vertical.

    Raises
    ------
    AssertionError
        If max |E(t) - E(0)| / |E(0)| exceeds ``integration.energy_drift_max_relative``.
    """
    m1 = float(params["m1"])
    m2 = float(params["m2"])
    L1 = float(params["L1"])
    L2 = float(params["L2"])
    g = float(params["g"])

    theta1 = state_array[:, 0]
    omega1 = state_array[:, 1]
    theta2 = state_array[:, 2]
    omega2 = state_array[:, 3]

    # Positions (y positive up; pivot at origin)
    y1 = -L1 * np.cos(theta1)
    y2 = y1 - L2 * np.cos(theta2)

    # Speeds of each bob
    v1_sq = (L1 * omega1) ** 2
    vx2 = L1 * omega1 * np.cos(theta1) + L2 * omega2 * np.cos(theta2)
    vy2 = L1 * omega1 * np.sin(theta1) + L2 * omega2 * np.sin(theta2)
    v2_sq = vx2**2 + vy2**2

    ke1 = 0.5 * m1 * v1_sq
    ke2 = 0.5 * m2 * v2_sq
    pe1 = m1 * g * y1
    pe2 = m2 * g * y2

    ke_total = ke1 + ke2
    with np.errstate(divide="ignore", invalid="ignore"):
        energy_ratio = np.where(ke_total > 0, ke2 / ke_total, np.nan)

    e_total = ke1 + ke2 + pe1 + pe2
    e0 = float(e_total[0])
    max_rel_drift = float(np.max(np.abs(e_total - e0)) / max(abs(e0), np.finfo(float).tiny))

    limit = float(config["integration"]["energy_drift_max_relative"])
    assert max_rel_drift <= limit, (
        "Energy conservation failed: max relative drift "
        f"{max_rel_drift:.3e} exceeds allowed {limit:.3e} (|E0|={abs(e0):.6e})."
    )

    return {
        "t": np.asarray(t_array, dtype=np.float64),
        "KE1": ke1,
        "KE2": ke2,
        "PE1": pe1,
        "PE2": pe2,
        "KE_total": ke_total,
        "energy_ratio": energy_ratio,
        "E_total": e_total,
    }
