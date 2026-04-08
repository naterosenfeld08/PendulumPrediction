"""Trajectory-level data splitting utilities."""

from __future__ import annotations

from collections.abc import Sequence


def split_ids(
    trajectory_ids: Sequence[str],
    train_frac: float,
    val_frac: float,
) -> dict[str, list[str]]:
    """Split trajectory IDs into train/val/test contiguous partitions."""
    if not 0.0 < train_frac < 1.0:
        raise ValueError("train_frac must be in (0, 1).")
    if not 0.0 <= val_frac < 1.0:
        raise ValueError("val_frac must be in [0, 1).")
    if train_frac + val_frac >= 1.0:
        raise ValueError("train_frac + val_frac must be < 1.")

    n = len(trajectory_ids)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train = list(trajectory_ids[:n_train])
    val = list(trajectory_ids[n_train : n_train + n_val])
    test = list(trajectory_ids[n_train + n_val :])
    return {"train": train, "val": val, "test": test}
