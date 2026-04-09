"""Hybrid embedding that concatenates physics and FFT features."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from embeddings.fft_features import FFTFeatureEmbedder
from embeddings.physics_features import PhysicsFeatureEmbedder


class HybridPhysicsFFTEmbedder:
    """Concatenate physics-informed and spectral features."""

    name = "hybrid_physics_fft"

    def __init__(self) -> None:
        self._physics = PhysicsFeatureEmbedder()
        self._fft = FFTFeatureEmbedder()

    def transform(
        self,
        t_window: NDArray[np.floating],
        states_window: NDArray[np.floating],
        energies_window: dict[str, NDArray[np.floating]],
        state_channels: tuple[str, ...],
    ) -> NDArray[np.floating]:
        f1 = self._physics.transform(t_window, states_window, energies_window, state_channels)
        f2 = self._fft.transform(t_window, states_window, energies_window, state_channels)
        return np.concatenate([f1, f2], axis=0).astype(np.float64)
