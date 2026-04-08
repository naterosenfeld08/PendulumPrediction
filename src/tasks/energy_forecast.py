"""Multi-horizon energy forecasting dataset builder."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from data.io import load_record
from embeddings.base import WindowEmbedder


@dataclass(frozen=True)
class ForecastTaskConfig:
    """Configuration for the supervised energy forecasting task."""

    window_size: int
    horizons: tuple[int, ...]
    stride: int


@dataclass(frozen=True)
class SupervisedDataset:
    """Feature matrix, target tensor, and provenance."""

    X: NDArray[np.floating]
    y: NDArray[np.floating]
    last_energy: NDArray[np.floating]
    trajectory_ids: list[str]


def _target_tensor(
    kinetic: NDArray[np.floating],
    potential: NDArray[np.floating],
    total: NDArray[np.floating],
    anchor_idx: int,
    horizons: tuple[int, ...],
) -> NDArray[np.floating]:
    rows = []
    for h in horizons:
        idx = anchor_idx + h
        rows.append([kinetic[idx], potential[idx], total[idx]])
    return np.asarray(rows, dtype=np.float64)


def build_supervised_dataset(
    trajectories_dir: Path,
    split: str,
    task_cfg: ForecastTaskConfig,
    embedder: WindowEmbedder,
) -> SupervisedDataset:
    """Load trajectories in a split and build a leakage-safe window dataset."""
    split_dir = trajectories_dir / split
    npz_files = sorted(split_dir.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No trajectory files found in split: {split_dir}")

    max_h = max(task_cfg.horizons)
    features: list[NDArray[np.floating]] = []
    targets: list[NDArray[np.floating]] = []
    last_energy_rows: list[NDArray[np.floating]] = []
    traj_ids: list[str] = []

    for npz_path in npz_files:
        rec = load_record(npz_path)
        n = len(rec.t)
        for start in range(0, n - task_cfg.window_size - max_h, task_cfg.stride):
            end = start + task_cfg.window_size
            anchor = end - 1
            t_window = rec.t[start:end]
            states_window = rec.states[start:end, :]
            energies_window = {
                k: np.asarray(v[start:end], dtype=np.float64) for k, v in rec.energies.items()
            }
            x = embedder.transform(
                t_window=t_window,
                states_window=states_window,
                energies_window=energies_window,
                state_channels=rec.state_channels,
            )
            y = _target_tensor(
                kinetic=np.asarray(rec.energies["kinetic"], dtype=np.float64),
                potential=np.asarray(rec.energies["potential"], dtype=np.float64),
                total=np.asarray(rec.energies["total"], dtype=np.float64),
                anchor_idx=anchor,
                horizons=task_cfg.horizons,
            )
            features.append(x)
            targets.append(y)
            last_energy_rows.append(
                np.asarray(
                    [
                        rec.energies["kinetic"][anchor],
                        rec.energies["potential"][anchor],
                        rec.energies["total"][anchor],
                    ],
                    dtype=np.float64,
                )
            )
            traj_ids.append(rec.trajectory_id)

    X = np.vstack(features).astype(np.float64)
    y = np.stack(targets, axis=0).astype(np.float64)
    last_energy = np.stack(last_energy_rows, axis=0).astype(np.float64)
    return SupervisedDataset(X=X, y=y, last_energy=last_energy, trajectory_ids=traj_ids)
