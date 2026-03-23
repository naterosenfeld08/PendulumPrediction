"""Nonlinear planar double pendulum: full Lagrangian equations of motion.

Angles θ₁, θ₂ are measured from the downward vertical (counterclockwise positive).
The state vector is [θ₁, ω₁, θ₂, ω₂] with ωᵢ = dθᵢ/dt.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray


def dstate_dt(
    t: float,
    state: NDArray[np.floating],
    params: Mapping[str, Any],
) -> NDArray[np.floating]:
    """Time derivative of state [θ₁, ω₁, θ₂, ω₂] for the planar double pendulum.

    Parameters
    ----------
    t
        Time (unused; ODE is autonomous).
    state
        Shape (4,): [θ₁, ω₁, θ₂, ω₂].
    params
        Must contain: m1, m2, L1, L2, g (gravity, m/s²).
    """
    theta1, omega1, theta2, omega2 = state
    m1 = float(params["m1"])
    m2 = float(params["m2"])
    L1 = float(params["L1"])
    L2 = float(params["L2"])
    g = float(params["g"])

    delta = theta1 - theta2
    cos_d = np.cos(delta)
    sin_d = np.sin(delta)

    # Mass matrix M @ [α₁, α₂]^T = b (angular accelerations)
    M11 = (m1 + m2) * L1**2
    M12 = m2 * L1 * L2 * cos_d
    M22 = m2 * L2**2

    b1 = (
        -m2 * L1 * L2 * sin_d * omega2**2
        - (m1 + m2) * g * L1 * np.sin(theta1)
    )
    b2 = (
        m2 * L1 * L2 * sin_d * omega1**2
        - m2 * g * L2 * np.sin(theta2)
    )

    det = M11 * M22 - M12**2
    if abs(det) < 1e-18:
        raise ValueError("Double pendulum mass matrix is singular; check geometry.")

    alpha1 = (M22 * b1 - M12 * b2) / det
    alpha2 = (-M12 * b1 + M11 * b2) / det

    return np.array([omega1, alpha1, omega2, alpha2], dtype=np.float64)
