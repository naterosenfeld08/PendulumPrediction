"""Embedding protocol definitions."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class WindowEmbedder(Protocol):
    """Convert one trajectory window into a fixed-length feature vector."""

    name: str

    def transform(
        self,
        t_window: NDArray[np.floating],
        states_window: NDArray[np.floating],
        energies_window: dict[str, NDArray[np.floating]],
        state_channels: tuple[str, ...],
    ) -> NDArray[np.floating]:
        """Return deterministic feature vector for one window."""
