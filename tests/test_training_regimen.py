"""Tests for split integrity and validation-only selection/lock behavior."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from data.io import save_record
from data.schema import TrajectoryRecord
from data.splits import split_ids
from experiments.runner import ExperimentConfig, locked_test_evaluate, select_top_candidates


def _write_minimal_record(root: Path, split: str, traj_id: str) -> None:
    t = np.linspace(0.0, 1.0, 21)
    states = np.column_stack([np.sin(t), np.cos(t)])
    rec = TrajectoryRecord(
        trajectory_id=traj_id,
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
        metadata={},
    )
    save_record(rec, root, split=split)


def test_split_ids_70_20_10_and_no_overlap() -> None:
    ids = [f"id_{i}" for i in range(100)]
    out = split_ids(ids, train_frac=0.70, val_frac=0.20)
    assert len(out["train"]) == 70
    assert len(out["val"]) == 20
    assert len(out["test"]) == 10
    assert set(out["train"]).isdisjoint(set(out["val"]))
    assert set(out["train"]).isdisjoint(set(out["test"]))
    assert set(out["val"]).isdisjoint(set(out["test"]))


def test_select_top_candidates_uses_validation_objective(tmp_path: Path) -> None:
    cfg = ExperimentConfig(
        trajectories_dir=tmp_path / "traj",
        artifacts_dir=tmp_path / "art",
        horizons=(1, 3),
        horizon_weights=(0.7, 0.3),
        top_k=2,
    )
    trials = [
        {"embedding": "a", "model": "m1", "seed": 1, "hyperparams": {}, "objective": 2.5},
        {"embedding": "b", "model": "m2", "seed": 2, "hyperparams": {}, "objective": 0.9},
        {"embedding": "c", "model": "m3", "seed": 3, "hyperparams": {}, "objective": 1.3},
    ]
    selected = select_top_candidates(cfg, trials)
    assert [x["model"] for x in selected] == ["m2", "m3"]


def test_locked_test_evaluate_requires_selected_file_when_missing(tmp_path: Path) -> None:
    cfg = ExperimentConfig(
        trajectories_dir=tmp_path / "traj",
        artifacts_dir=tmp_path / "art",
        horizons=(1, 3),
        horizon_weights=(0.7, 0.3),
    )
    with pytest.raises(FileNotFoundError, match="selected_candidates.json"):
        locked_test_evaluate(cfg)


def test_locked_test_evaluate_runs_for_provided_selection(tmp_path: Path) -> None:
    traj_dir = tmp_path / "traj"
    _write_minimal_record(traj_dir, "train", "r_train")
    _write_minimal_record(traj_dir, "val", "r_val")
    _write_minimal_record(traj_dir, "test", "r_test")

    cfg = ExperimentConfig(
        trajectories_dir=traj_dir,
        artifacts_dir=tmp_path / "art",
        horizons=(1, 2),
        window_size=6,
        stride=2,
        horizon_weights=(0.7, 0.3),
    )
    selected = [
        {
            "embedding": "physics_features_v1",
            "model": "persistence",
            "seed": 1,
            "hyperparams": {},
            "objective": 1.0,
        }
    ]
    out = locked_test_evaluate(cfg, selected=selected)
    assert len(out) == 1
    assert out[0]["model"] == "persistence"
