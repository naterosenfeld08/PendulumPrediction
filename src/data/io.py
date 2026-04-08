"""Read/write utilities for trajectory records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from data.schema import TrajectoryRecord, as_serializable_metadata, validate_record


def save_record(record: TrajectoryRecord, root_dir: Path, split: str) -> Path:
    """Persist one trajectory record as an NPZ + JSON sidecar."""
    validate_record(record)
    split_dir = root_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)

    npz_path = split_dir / f"{record.trajectory_id}.npz"
    json_path = split_dir / f"{record.trajectory_id}.json"
    np.savez_compressed(
        npz_path,
        t=np.asarray(record.t, dtype=np.float64),
        states=np.asarray(record.states, dtype=np.float64),
        state_channels=np.asarray(record.state_channels, dtype=object),
        **{f"energy__{k}": np.asarray(v, dtype=np.float64) for k, v in record.energies.items()},
    )
    payload = {
        "trajectory_id": record.trajectory_id,
        "system_name": record.system_name,
        "parameters": {k: float(v) for k, v in record.parameters.items()},
        "metadata": as_serializable_metadata(record.metadata),
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return npz_path


def load_record(npz_path: Path) -> TrajectoryRecord:
    """Load a trajectory record stored with :func:`save_record`."""
    json_path = npz_path.with_suffix(".json")
    if not json_path.is_file():
        raise FileNotFoundError(f"Missing sidecar metadata file: {json_path}")

    payload: dict[str, Any] = json.loads(json_path.read_text(encoding="utf-8"))
    with np.load(npz_path, allow_pickle=True) as data:
        energies = {
            k.removeprefix("energy__"): np.asarray(v, dtype=np.float64)
            for k, v in data.items()
            if k.startswith("energy__")
        }
        record = TrajectoryRecord(
            trajectory_id=str(payload["trajectory_id"]),
            system_name=str(payload["system_name"]),
            t=np.asarray(data["t"], dtype=np.float64),
            states=np.asarray(data["states"], dtype=np.float64),
            state_channels=tuple(str(x) for x in data["state_channels"].tolist()),
            energies=energies,
            parameters={k: float(v) for k, v in payload["parameters"].items()},
            metadata=dict(payload.get("metadata", {})),
        )
    validate_record(record)
    return record
