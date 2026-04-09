"""Trajectory generation pipeline across supported systems."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from data.io import save_record
from data.splits import split_ids
from systems.base import SimulationSpec
from systems.registry import build_systems


@dataclass(frozen=True)
class GenerationConfig:
    """Config for generating trajectory corpora."""

    output_dir: Path
    n_per_system: int
    duration_s: float
    n_steps: int
    seed: int
    train_frac: float
    val_frac: float


def _sample_double_params(rng: np.random.Generator) -> dict[str, float]:
    return {
        "m1": float(rng.uniform(0.6, 1.8)),
        "m2": float(rng.uniform(0.6, 1.8)),
        "L1": float(rng.uniform(0.6, 1.4)),
        "L2": float(rng.uniform(0.6, 1.4)),
        "theta1": float(rng.uniform(0.05, np.pi - 0.05)),
        "theta2": float(rng.uniform(0.05, np.pi - 0.05)),
        "omega1": float(rng.uniform(-1.0, 1.0)),
        "omega2": float(rng.uniform(-1.0, 1.0)),
        "g": 9.81,
    }


def _sample_driven_params(rng: np.random.Generator) -> dict[str, float]:
    return {
        "m": float(rng.uniform(0.8, 1.2)),
        "L": float(rng.uniform(0.7, 1.4)),
        "theta": float(rng.uniform(-np.pi, np.pi)),
        "omega": float(rng.uniform(-1.2, 1.2)),
        "damping": float(rng.uniform(0.02, 0.25)),
        "drive_amplitude": float(rng.uniform(0.2, 1.2)),
        "drive_frequency": float(rng.uniform(0.5, 2.0)),
        "g": 9.81,
    }


def generate_trajectory_corpus(base_config: dict, cfg: GenerationConfig) -> dict[str, int]:
    """Generate trajectories, then split by trajectory ID before windowing."""
    rng = np.random.default_rng(cfg.seed)
    systems = build_systems(base_config)
    all_ids: list[str] = []
    records_meta: list[dict[str, str]] = []
    count = 0

    for system_name, system in systems.items():
        for i in range(cfg.n_per_system):
            traj_id = f"{system_name}_{i:05d}"
            all_ids.append(traj_id)
            params = (
                _sample_double_params(rng)
                if system_name == "double_pendulum"
                else _sample_driven_params(rng)
            )
            spec = SimulationSpec(
                trajectory_id=traj_id,
                duration_s=cfg.duration_s,
                n_steps=cfg.n_steps,
                seed=int(rng.integers(0, 2**31 - 1)),
                parameters=params,
            )
            record = system.simulate(spec)
            records_meta.append({"trajectory_id": traj_id, "system_name": system_name})
            save_record(record, cfg.output_dir, split="all")
            count += 1

    shuffled = list(all_ids)
    rng.shuffle(shuffled)
    split_map = split_ids(shuffled, train_frac=cfg.train_frac, val_frac=cfg.val_frac)
    split_membership = {traj_id: split for split, ids in split_map.items() for traj_id in ids}

    for split, ids in split_map.items():
        for traj_id in ids:
            src_npz = cfg.output_dir / "all" / f"{traj_id}.npz"
            src_json = cfg.output_dir / "all" / f"{traj_id}.json"
            split_dir = cfg.output_dir / split
            split_dir.mkdir(parents=True, exist_ok=True)
            dst_npz = split_dir / src_npz.name
            dst_json = split_dir / src_json.name
            dst_npz.write_bytes(src_npz.read_bytes())
            dst_json.write_text(src_json.read_text(encoding="utf-8"), encoding="utf-8")

    system_split_counts: dict[str, dict[str, int]] = {}
    for rec in records_meta:
        sys_name = rec["system_name"]
        split = split_membership[rec["trajectory_id"]]
        if sys_name not in system_split_counts:
            system_split_counts[sys_name] = {"train": 0, "val": 0, "test": 0}
        system_split_counts[sys_name][split] += 1

    overlap_count = len(set(split_map["train"]) & set(split_map["val"])) + len(
        set(split_map["train"]) & set(split_map["test"])
    ) + len(set(split_map["val"]) & set(split_map["test"]))

    manifest = {
        "n_total": count,
        "splits": {k: len(v) for k, v in split_map.items()},
        "records": records_meta,
        "split_integrity": {
            "train_frac": cfg.train_frac,
            "val_frac": cfg.val_frac,
            "test_frac": 1.0 - cfg.train_frac - cfg.val_frac,
            "overlap_count": int(overlap_count),
            "system_split_counts": system_split_counts,
        },
    }
    (cfg.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (cfg.output_dir / "split_summary.json").write_text(
        json.dumps(
            {
                "splits": {k: len(v) for k, v in split_map.items()},
                "overlap_count": int(overlap_count),
                "system_split_counts": system_split_counts,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return manifest["splits"]
