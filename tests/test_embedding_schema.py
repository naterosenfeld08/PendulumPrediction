"""Contract tests for unified trajectory schema and task dataset shapes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from data.io import load_record, save_record
from data.schema import TrajectoryRecord, validate_record
from embeddings.physics_features import PhysicsFeatureEmbedder
from tasks.energy_forecast import ForecastTaskConfig, build_supervised_dataset


def test_trajectory_record_roundtrip(tmp_path: Path) -> None:
    t = np.linspace(0.0, 1.0, 11)
    states = np.column_stack([np.sin(t), np.cos(t)])
    rec = TrajectoryRecord(
        trajectory_id="driven_00001",
        system_name="driven_damped_single",
        t=t,
        states=states,
        state_channels=("theta", "omega"),
        energies={
            "kinetic": np.abs(np.sin(t)),
            "potential": np.abs(np.cos(t)),
            "total": np.abs(np.sin(t)) + np.abs(np.cos(t)),
        },
        parameters={"L": 1.0, "damping": 0.1},
        metadata={"seed": 1},
    )
    validate_record(rec)
    save_record(rec, tmp_path, split="train")
    loaded = load_record(tmp_path / "train" / "driven_00001.npz")
    assert loaded.trajectory_id == rec.trajectory_id
    assert loaded.system_name == rec.system_name
    np.testing.assert_allclose(loaded.t, rec.t)
    np.testing.assert_allclose(loaded.states, rec.states)
    np.testing.assert_allclose(loaded.energies["total"], rec.energies["total"])


def test_schema_rejects_missing_energy_channel() -> None:
    t = np.linspace(0.0, 1.0, 8)
    rec = TrajectoryRecord(
        trajectory_id="bad",
        system_name="x",
        t=t,
        states=np.column_stack([t, t]),
        state_channels=("theta", "omega"),
        energies={"kinetic": t, "potential": t},
        parameters={},
        metadata={},
    )
    with pytest.raises(ValueError, match="Missing required energy channel"):
        validate_record(rec)


def test_build_supervised_dataset_shapes(tmp_path: Path) -> None:
    t = np.linspace(0.0, 2.0, 41)
    states = np.column_stack([np.sin(t), np.cos(t)])
    rec = TrajectoryRecord(
        trajectory_id="driven_00002",
        system_name="driven_damped_single",
        t=t,
        states=states,
        state_channels=("theta", "omega"),
        energies={
            "kinetic": np.abs(np.sin(t)),
            "potential": np.abs(np.cos(t)),
            "total": np.abs(np.sin(t)) + np.abs(np.cos(t)),
        },
        parameters={"L": 1.0, "damping": 0.1},
        metadata={"seed": 2},
    )
    save_record(rec, tmp_path, split="train")
    ds = build_supervised_dataset(
        trajectories_dir=tmp_path,
        split="train",
        task_cfg=ForecastTaskConfig(window_size=8, horizons=(1, 3), stride=2),
        embedder=PhysicsFeatureEmbedder(),
    )
    assert ds.X.ndim == 2
    assert ds.y.ndim == 3
    assert ds.y.shape[1:] == (2, 3)
    assert ds.last_energy.shape[1] == 3
