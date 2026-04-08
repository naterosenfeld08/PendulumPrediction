"""Driven damped single pendulum simulator."""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from data.schema import TrajectoryRecord, validate_record
from systems.base import SimulationSpec


class DrivenDampedPendulumSystem:
    """Generate trajectories for a forced damped single pendulum."""

    name = "driven_damped_single"

    def simulate(self, spec: SimulationSpec) -> TrajectoryRecord:
        p = spec.parameters
        m = float(p.get("m", 1.0))
        L = float(p["L"])
        g = float(p.get("g", 9.81))
        damping = float(p["damping"])
        drive_amp = float(p["drive_amplitude"])
        drive_freq = float(p["drive_frequency"])
        theta0 = float(p["theta"])
        omega0 = float(p["omega"])

        t_eval = np.linspace(0.0, float(spec.duration_s), int(spec.n_steps) + 1)

        def rhs(t: float, y: np.ndarray) -> np.ndarray:
            theta, omega = y
            dtheta = omega
            domega = -(g / L) * np.sin(theta) - damping * omega + drive_amp * np.sin(
                drive_freq * t
            )
            return np.array([dtheta, domega], dtype=np.float64)

        sol = solve_ivp(
            rhs,
            (0.0, float(spec.duration_s)),
            np.array([theta0, omega0], dtype=np.float64),
            t_eval=t_eval,
            method="RK45",
            rtol=1e-9,
            atol=1e-12,
        )
        if not sol.success:
            raise RuntimeError(f"Driven pendulum integration failed: {sol.message}")

        states = sol.y.T
        theta = states[:, 0]
        omega = states[:, 1]
        kinetic = 0.5 * m * (L * omega) ** 2
        potential = m * g * L * (1.0 - np.cos(theta))
        total = kinetic + potential

        record = TrajectoryRecord(
            trajectory_id=spec.trajectory_id,
            system_name=self.name,
            t=sol.t,
            states=states,
            state_channels=("theta", "omega"),
            energies={"kinetic": kinetic, "potential": potential, "total": total},
            parameters={
                "m": m,
                "L": L,
                "g": g,
                "damping": damping,
                "drive_amplitude": drive_amp,
                "drive_frequency": drive_freq,
                "theta": theta0,
                "omega": omega0,
            },
            metadata={"seed": int(spec.seed), "noise_source": "none"},
        )
        validate_record(record)
        return record
