"""Two-trajectory separation and maximal Lyapunov exponent (θ₁-only perturbation)."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from physics.integrator import integrate
from physics.pendulum import dstate_dt


def separation_timeseries(
    state0: NDArray[np.floating],
    params: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Integrate reference and θ₁-perturbed trajectories; return ``t`` and Euclidean δ(t).

    The perturbed initial state matches ``state0`` except ``θ₁ ← θ₁ + delta0`` (from config).
    """
    ly = config["lyapunov"]
    delta0 = float(ly["delta0"])

    state0 = np.asarray(state0, dtype=np.float64).copy()
    t, y_ref = integrate(dstate_dt, state0, params, config)

    state0_p = state0.copy()
    state0_p[0] += delta0
    _, y_pert = integrate(dstate_dt, state0_p, params, config)

    delta = np.linalg.norm(y_ref - y_pert, axis=1)
    return t, delta


def compute_mle(
    t_array: NDArray[np.floating],
    state_array: NDArray[np.floating],
    params: Mapping[str, Any],
    config: Mapping[str, Any],
) -> float:
    """MLE via two trajectories; reference is ``state_array`` at matching ``t_array``.

    Re-integrates the reference from ``state_array[0]`` and a θ₁-perturbed copy so that
    ``t`` sampling matches the Lyapunov config (same as ensemble integration).
    """
    ly = config["lyapunov"]
    delta0 = float(ly["delta0"])
    t_horizon = float(ly["t_lyapunov"])

    state0 = np.asarray(state_array[0], dtype=np.float64)
    t, delta = separation_timeseries(state0, params, config)

    if t_horizon < t[0] or t_horizon > t[-1]:
        raise ValueError(
            f"t_lyapunov={t_horizon} must lie within integration span [{t[0]}, {t[-1]}]."
        )

    if delta0 <= 0:
        raise ValueError("lyapunov.delta0 must be > 0.")

    delta_end = float(np.interp(t_horizon, t, delta))
    # Numerical underflow can produce non-positive separation for very tiny delta0.
    delta_end = max(delta_end, float(np.finfo(float).tiny))

    return (1.0 / t_horizon) * np.log(delta_end / delta0)


def save_delta_curve(
    t: NDArray[np.floating],
    delta: NDArray[np.floating],
    out_path: str,
    title: str = "δ(t) = ‖state_ref − state_pert‖",
) -> None:
    """Write a diagnostic plot of separation vs time (requires matplotlib)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(t, delta, color="C0", lw=1.0)
    ax.set_xlabel("t (s)")
    ax.set_ylabel("δ(t)")
    ax.set_title(title)
    ax.grid(True, which="both", ls=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
