"""Physics-informed embedding for fixed trajectory windows."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _safe_stats(x: NDArray[np.floating]) -> list[float]:
    return [
        float(np.mean(x)),
        float(np.std(x)),
        float(np.min(x)),
        float(np.max(x)),
    ]


def _slope(t: NDArray[np.floating], x: NDArray[np.floating]) -> float:
    if len(t) < 2:
        return 0.0
    dt = float(t[-1] - t[0])
    if dt == 0.0:
        return 0.0
    return float((x[-1] - x[0]) / dt)


def _spectral_entropy(x: NDArray[np.floating]) -> float:
    x0 = np.asarray(x, dtype=np.float64)
    x0 = x0 - np.mean(x0)
    fft_mag = np.abs(np.fft.rfft(x0)) + 1e-12
    p = fft_mag / np.sum(fft_mag)
    ent = -np.sum(p * np.log(p))
    return float(ent / np.log(len(p)))


class PhysicsFeatureEmbedder:
    """System-agnostic, physically motivated window features."""

    name = "physics_features_v1"

    def transform(
        self,
        t_window: NDArray[np.floating],
        states_window: NDArray[np.floating],
        energies_window: dict[str, NDArray[np.floating]],
        state_channels: tuple[str, ...],
    ) -> NDArray[np.floating]:
        theta_idx = [i for i, ch in enumerate(state_channels) if "theta" in ch]
        omega_idx = [i for i, ch in enumerate(state_channels) if "omega" in ch]

        theta = (
            np.mean(states_window[:, theta_idx], axis=1)
            if theta_idx
            else np.zeros(states_window.shape[0], dtype=np.float64)
        )
        omega = (
            np.mean(states_window[:, omega_idx], axis=1)
            if omega_idx
            else np.zeros(states_window.shape[0], dtype=np.float64)
        )
        kinetic = np.asarray(energies_window["kinetic"], dtype=np.float64)
        potential = np.asarray(energies_window["potential"], dtype=np.float64)
        total = np.asarray(energies_window["total"], dtype=np.float64)

        eps = 1e-12
        kinetic_frac = kinetic / (np.abs(total) + eps)
        total_grad = np.gradient(total, t_window) if len(t_window) > 1 else np.zeros_like(total)

        feats: list[float] = []
        for signal in (theta, omega, kinetic, potential, total, kinetic_frac, total_grad):
            feats.extend(_safe_stats(signal))
            feats.append(_slope(t_window, signal))
            feats.append(_spectral_entropy(signal))

        theta_cross = float(np.mean(theta * omega))
        omega_abs = float(np.mean(np.abs(omega)))
        zero_crossings = float(np.sum(np.diff(np.signbit(theta)).astype(np.int32)))
        feats.extend([theta_cross, omega_abs, zero_crossings])
        return np.asarray(feats, dtype=np.float64)
