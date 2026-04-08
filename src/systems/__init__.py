"""System adapters for trajectory generation."""

from systems.base import SimulationSpec
from systems.double_pendulum import DoublePendulumSystem
from systems.driven_pendulum import DrivenDampedPendulumSystem
from systems.registry import build_systems

__all__ = [
    "SimulationSpec",
    "DoublePendulumSystem",
    "DrivenDampedPendulumSystem",
    "build_systems",
]
