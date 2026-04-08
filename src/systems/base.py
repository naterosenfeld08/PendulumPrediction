"""Simulation interfaces for supported pendulum systems."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from data.schema import TrajectoryRecord


@dataclass(frozen=True)
class SimulationSpec:
    """Configuration for generating one trajectory."""

    trajectory_id: str
    duration_s: float
    n_steps: int
    seed: int
    parameters: dict[str, float]


class PendulumSystem(Protocol):
    """Interface each physical system adapter must implement."""

    name: str

    def simulate(self, spec: SimulationSpec) -> TrajectoryRecord:
        """Simulate a trajectory and return a validated record."""
