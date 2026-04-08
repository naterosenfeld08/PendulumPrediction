"""Adapter that wraps the existing nonlinear double pendulum solver."""

from __future__ import annotations

import copy

import numpy as np

from data.schema import TrajectoryRecord, validate_record
from physics.energy import compute_energy_timeseries
from physics.integrator import integrate
from physics.pendulum import dstate_dt
from systems.base import SimulationSpec


class DoublePendulumSystem:
    """Generate trajectories for the canonical double pendulum model."""

    name = "double_pendulum"

    def __init__(self, base_config: dict) -> None:
        self._base_config = copy.deepcopy(base_config)

    def simulate(self, spec: SimulationSpec) -> TrajectoryRecord:
        rng = np.random.default_rng(spec.seed)
        config = copy.deepcopy(self._base_config)
        config["integration"]["t_span"] = [0.0, float(spec.duration_s)]
        config["integration"]["n_steps"] = int(spec.n_steps)

        params = {
            "m1": float(spec.parameters["m1"]),
            "m2": float(spec.parameters["m2"]),
            "L1": float(spec.parameters["L1"]),
            "L2": float(spec.parameters["L2"]),
            "g": float(spec.parameters.get("g", 9.81)),
        }
        state0 = np.array(
            [
                float(spec.parameters["theta1"]),
                float(spec.parameters["omega1"]),
                float(spec.parameters["theta2"]),
                float(spec.parameters["omega2"]),
            ],
            dtype=np.float64,
        )

        t, states = integrate(dstate_dt, state0, params, config)
        e = compute_energy_timeseries(t, states, params, config)
        record = TrajectoryRecord(
            trajectory_id=spec.trajectory_id,
            system_name=self.name,
            t=t,
            states=states,
            state_channels=("theta1", "omega1", "theta2", "omega2"),
            energies={
                "kinetic": np.asarray(e["KE_total"], dtype=np.float64),
                "potential": np.asarray(e["PE1"] + e["PE2"], dtype=np.float64),
                "total": np.asarray(e["E_total"], dtype=np.float64),
            },
            parameters=params
            | {
                "theta1": float(state0[0]),
                "omega1": float(state0[1]),
                "theta2": float(state0[2]),
                "omega2": float(state0[3]),
            },
            metadata={
                "seed": int(spec.seed),
                "noise_source": "none",
                "rng_probe": float(rng.uniform()),
            },
        )
        validate_record(record)
        return record
