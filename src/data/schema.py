"""Unified trajectory schema used by all dynamical systems."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray


REQUIRED_ENERGY_CHANNELS = ("kinetic", "potential", "total")


@dataclass(frozen=True)
class TrajectoryRecord:
    """One simulated trajectory plus metadata and derived energy channels."""

    trajectory_id: str
    system_name: str
    t: NDArray[np.floating]
    states: NDArray[np.floating]
    state_channels: tuple[str, ...]
    energies: dict[str, NDArray[np.floating]]
    parameters: dict[str, float]
    metadata: dict[str, Any]


def validate_record(record: TrajectoryRecord) -> None:
    """Validate shape and channel constraints for a trajectory record."""
    t = np.asarray(record.t, dtype=np.float64)
    states = np.asarray(record.states, dtype=np.float64)
    if t.ndim != 1:
        raise ValueError("Trajectory time array must be 1D.")
    if states.ndim != 2:
        raise ValueError("Trajectory states array must be 2D.")
    if states.shape[0] != t.shape[0]:
        raise ValueError("States first dimension must match time length.")
    if states.shape[1] != len(record.state_channels):
        raise ValueError("State channel count must match states.shape[1].")
    if not record.trajectory_id:
        raise ValueError("Trajectory id must be non-empty.")
    if not record.system_name:
        raise ValueError("System name must be non-empty.")

    for name in REQUIRED_ENERGY_CHANNELS:
        if name not in record.energies:
            raise ValueError(f"Missing required energy channel: {name}")

    for name, values in record.energies.items():
        arr = np.asarray(values, dtype=np.float64)
        if arr.ndim != 1:
            raise ValueError(f"Energy channel '{name}' must be 1D.")
        if arr.shape[0] != t.shape[0]:
            raise ValueError(
                f"Energy channel '{name}' length must match time length."
            )


def as_serializable_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Convert metadata values into JSON-compatible primitives."""
    out: dict[str, Any] = {}
    for k, v in metadata.items():
        if isinstance(v, (int, float, str, bool)) or v is None:
            out[str(k)] = v
        else:
            out[str(k)] = str(v)
    return out
