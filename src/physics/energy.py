"""Kinetic and potential energy timeseries; hard energy conservation check."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray


def scaled_max_energy_drift(
    e_total: NDArray[np.floating],
    e0: float,
    params: Mapping[str, Any],
) -> float:
    """Max |E(t)-E(0)| divided by a robust scale.

    Using only ``|E(0)|`` in the denominator is ill-conditioned when total mechanical
    energy is very small (near-separatrix initial data): tiny integrator noise looks
    like a huge *relative* drift. We therefore compare drift to
    ``max(|E(0)|, 0.01 * E_char)``, where ``E_char`` is a gravitational energy scale
    built from masses and lengths (order of magnitude of |PE| for O(1) rad angles).
    """
    m1 = float(params["m1"])
    m2 = float(params["m2"])
    L1 = float(params["L1"])
    L2 = float(params["L2"])
    g = float(params["g"])
    e_char = float(m1 * g * L1 + m2 * g * (L1 + L2))
    denom = max(abs(float(e0)), 0.01 * e_char)
    return float(np.max(np.abs(e_total - float(e0))) / denom)


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
        If scaled drift (see ``scaled_max_energy_drift``) exceeds
        ``integration.energy_drift_max_relative``.
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
    max_rel_drift = scaled_max_energy_drift(e_total, e0, params)

    limit = float(config["integration"]["energy_drift_max_relative"])
    assert max_rel_drift <= limit, (
        "Energy conservation failed: scaled max energy drift "
        f"{max_rel_drift:.3e} exceeds allowed {limit:.3e} (E0={e0:.6e})."
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
