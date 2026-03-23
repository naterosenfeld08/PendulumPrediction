"""High-accuracy time integration via scipy.integrate.solve_ivp (RK45)."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp


def integrate(
    dstate_dt: Callable[[float, NDArray[np.floating], Mapping[str, Any]], NDArray[np.floating]],
    state0: NDArray[np.floating],
    params: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Integrate the ODE from config integration settings.

    Returns
    -------
    t_array
        Shape (n_points,).
    state_array
        Shape (n_points, 4): rows are [θ₁, ω₁, θ₂, ω₂].
    """
    integ = config["integration"]
    t_span = tuple(float(x) for x in integ["t_span"])
    n_steps = int(integ["n_steps"])
    rtol = float(integ["rtol"])
    atol = float(integ["atol"])

    t_eval = np.linspace(t_span[0], t_span[1], n_steps + 1)

    def rhs(t: float, y: NDArray[np.floating]) -> NDArray[np.floating]:
        return dstate_dt(t, y, params)

    sol = solve_ivp(
        rhs,
        t_span,
        np.asarray(state0, dtype=np.float64),
        method="RK45",
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
    )
    if not sol.success:
        raise RuntimeError(f"ODE integration failed: {sol.message}")

    return sol.t, sol.y.T
