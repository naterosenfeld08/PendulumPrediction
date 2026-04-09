"""Spectral embedding from state and energy channels."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from embeddings.common import canonicalize_states_window


def _channel_fft_features(x: NDArray[np.floating]) -> list[float]:
    x0 = np.asarray(x, dtype=np.float64) - np.mean(x)
    mag = np.abs(np.fft.rfft(x0))
    if len(mag) == 0:
        return [0.0] * 6
    total = float(np.sum(mag) + 1e-12)
    p = mag / total
    idx_peak = int(np.argmax(mag))
    centroid = float(np.sum(np.arange(len(mag)) * p))
    spread = float(np.sqrt(np.sum(((np.arange(len(mag)) - centroid) ** 2) * p)))
    entropy = float(-np.sum(p * np.log(p + 1e-12)) / np.log(len(mag) + 1e-12))
    return [
        float(mag[idx_peak]),
        float(idx_peak),
        centroid,
        spread,
        entropy,
        float(np.mean(mag)),
    ]


class FFTFeatureEmbedder:
    """Embed window using FFT summary statistics."""

    name = "fft_features"

    def transform(
        self,
        t_window: NDArray[np.floating],
        states_window: NDArray[np.floating],
        energies_window: dict[str, NDArray[np.floating]],
        state_channels: tuple[str, ...],
    ) -> NDArray[np.floating]:
        _ = t_window
        canonical = canonicalize_states_window(states_window, state_channels)
        feats: list[float] = []
        for i in range(canonical.shape[1]):
            feats.extend(_channel_fft_features(canonical[:, i]))
        for key in ("kinetic", "potential", "total"):
            feats.extend(_channel_fft_features(np.asarray(energies_window[key], dtype=np.float64)))
        return np.asarray(feats, dtype=np.float64)
