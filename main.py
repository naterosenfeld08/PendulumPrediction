"""Experiment-first CLI for trajectory-embedding pendulum forecasting."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from experiments.runner import ExperimentConfig, generate_data, train_and_evaluate  # noqa: E402


def _load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Expected mapping in config: {path}")
    return cfg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Embedding-first pendulum workflow: generate trajectories, "
            "train baselines, and evaluate multi-horizon energy forecasts."
        )
    )
    p.add_argument(
        "--config",
        type=Path,
        default=_ROOT / "config.yaml",
        help="Base physics config path used by the double pendulum adapter.",
    )
    p.add_argument(
        "--embedding-config",
        type=Path,
        default=_ROOT / "embedding_config.yaml",
        help="Embedding-first experiment config path.",
    )
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("generate", help="Generate trajectory dataset for all systems.")
    sub.add_parser("train", help="Train and evaluate forecasting baselines.")
    sub.add_parser("run-all", help="Generate data then train/evaluate.")
    return p.parse_args()


def _build_experiment_config(cfg: dict) -> ExperimentConfig:
    d = cfg.get("dataset", {})
    t = cfg.get("task", {})
    return ExperimentConfig(
        trajectories_dir=(_ROOT / d.get("trajectories_dir", "data/trajectories")).resolve(),
        artifacts_dir=(_ROOT / d.get("artifacts_dir", "data/embedding_artifacts")).resolve(),
        n_per_system=int(d.get("n_per_system", 80)),
        duration_s=float(d.get("duration_s", 30.0)),
        n_steps=int(d.get("n_steps", 1200)),
        seed=int(d.get("seed", 42)),
        train_frac=float(d.get("train_frac", 0.7)),
        val_frac=float(d.get("val_frac", 0.15)),
        window_size=int(t.get("window_size", 64)),
        horizons=tuple(int(h) for h in t.get("horizons", [1, 5, 10, 20])),
        stride=int(t.get("stride", 4)),
    )


def main() -> None:
    args = parse_args()
    base_config = _load_yaml(args.config.resolve())
    emb_cfg = _load_yaml(args.embedding_config.resolve())
    exp_cfg = _build_experiment_config(emb_cfg)

    if args.command in {"generate", "run-all"}:
        split_sizes = generate_data(base_config=base_config, cfg=exp_cfg)
        print("Generated trajectories:", split_sizes)
    if args.command in {"train", "run-all"}:
        metrics = train_and_evaluate(cfg=exp_cfg)
        print("Saved metrics for models:", ", ".join(sorted(metrics.keys())))


if __name__ == "__main__":
    main()
