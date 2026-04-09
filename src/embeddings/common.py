"""Shared embedding helpers."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


CANONICAL_CHANNELS = ("theta1", "omega1", "theta2", "omega2")


def canonicalize_states_window(
    states_window: NDArray[np.floating],
    state_channels: tuple[str, ...],
) -> NDArray[np.floating]:
    """Map per-system state channels to canonical 4-channel representation."""
    n = states_window.shape[0]
    out = np.zeros((n, len(CANONICAL_CHANNELS)), dtype=np.float64)
    idx = {name: i for i, name in enumerate(state_channels)}

    if "theta1" in idx:
        out[:, 0] = states_window[:, idx["theta1"]]
    elif "theta" in idx:
        out[:, 0] = states_window[:, idx["theta"]]
    if "omega1" in idx:
        out[:, 1] = states_window[:, idx["omega1"]]
    elif "omega" in idx:
        out[:, 1] = states_window[:, idx["omega"]]
    if "theta2" in idx:
        out[:, 2] = states_window[:, idx["theta2"]]
    if "omega2" in idx:
        out[:, 3] = states_window[:, idx["omega2"]]
    return out
