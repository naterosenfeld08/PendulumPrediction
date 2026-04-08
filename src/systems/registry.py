"""Factory for supported pendulum systems."""

from __future__ import annotations

from systems.double_pendulum import DoublePendulumSystem
from systems.driven_pendulum import DrivenDampedPendulumSystem


def build_systems(base_config: dict) -> dict[str, object]:
    """Return all supported system adapters keyed by system name."""
    return {
        DoublePendulumSystem.name: DoublePendulumSystem(base_config=base_config),
        DrivenDampedPendulumSystem.name: DrivenDampedPendulumSystem(),
    }
