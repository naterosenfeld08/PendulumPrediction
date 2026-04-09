"""Raw-window embedding from flattened states."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from embeddings.common import canonicalize_states_window


class RawWindowEmbedder:
    """Flatten the full state window into one feature vector."""

    name = "raw_window"

    def transform(
        self,
        t_window: NDArray[np.floating],
        states_window: NDArray[np.floating],
        energies_window: dict[str, NDArray[np.floating]],
        state_channels: tuple[str, ...],
    ) -> NDArray[np.floating]:
        _ = t_window, energies_window
        canonical = canonicalize_states_window(states_window, state_channels)
        return np.asarray(canonical, dtype=np.float64).ravel()
