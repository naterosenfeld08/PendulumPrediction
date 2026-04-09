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

from experiments.runner import (  # noqa: E402
    ExperimentConfig,
    build_leaderboard,
    generate_data,
    locked_test_evaluate,
    select_top_candidates,
    sweep_train_val,
    train_and_evaluate,
)


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
    sub.add_parser("train", help="Quick compatibility baseline run.")
    sub.add_parser("sweep", help="Run train/val model search.")
    sub.add_parser("select", help="Select top-k candidates from sweep trials.")
    sub.add_parser("final-eval", help="Locked test evaluation for selected candidates.")
    sub.add_parser("run-all", help="Generate, sweep, select, and final-eval.")
    return p.parse_args()


def _build_experiment_config(cfg: dict) -> ExperimentConfig:
    d = cfg.get("dataset", {})
    t = cfg.get("task", {})
    e = cfg.get("experiment", {})
    s = cfg.get("search", {})
    sel = cfg.get("selection", {})
    return ExperimentConfig(
        trajectories_dir=(_ROOT / d.get("trajectories_dir", "data/trajectories")).resolve(),
        artifacts_dir=(_ROOT / d.get("artifacts_dir", "data/embedding_artifacts")).resolve(),
        n_per_system=int(d.get("n_per_system", 80)),
        duration_s=float(d.get("duration_s", 30.0)),
        n_steps=int(d.get("n_steps", 1200)),
        seed=int(d.get("seed", 42)),
        train_frac=float(d.get("train_frac", 0.7)),
        val_frac=float(d.get("val_frac", 0.20)),
        window_size=int(t.get("window_size", 64)),
        horizons=tuple(int(h) for h in t.get("horizons", [1, 5, 10, 20])),
        stride=int(t.get("stride", 4)),
        seed_grid=tuple(int(x) for x in e.get("seed_grid", [11, 29, 47, 71, 97])),
        embedding_list=tuple(
            str(x)
            for x in e.get(
                "embedding_list",
                ["physics_features_v1", "raw_window", "fft_features", "hybrid_physics_fft"],
            )
        ),
        model_list=tuple(
            str(x)
            for x in e.get(
                "model_list",
                [
                    "persistence",
                    "linear",
                    "ridge",
                    "random_forest",
                    "gradient_boosting",
                    "mlp_small",
                    "mlp_medium",
                ],
            )
        ),
        trials_per_model=int(s.get("trials_per_model", 20)),
        max_parallel_jobs=int(s.get("max_parallel_jobs", 1)),
        primary_metric=str(sel.get("primary_metric", "weighted_rmse")),
        horizon_weights=tuple(float(x) for x in sel.get("horizon_weights", [0.45, 0.30, 0.15, 0.10])),
        top_k=int(sel.get("top_k", 3)),
    )


def main() -> None:
    args = parse_args()
    base_config = _load_yaml(args.config.resolve())
    emb_cfg = _load_yaml(args.embedding_config.resolve())
    exp_cfg = _build_experiment_config(emb_cfg)

    if args.command in {"generate", "run-all"}:
        split_sizes = generate_data(base_config=base_config, cfg=exp_cfg)
        print("Generated trajectories:", split_sizes)
    if args.command == "train":
        metrics = train_and_evaluate(cfg=exp_cfg)
        print("Saved metrics for models:", ", ".join(sorted(metrics.keys())))
    if args.command in {"sweep", "run-all"}:
        trials = sweep_train_val(cfg=exp_cfg)
        leaderboard = build_leaderboard(exp_cfg, trials)
        print(f"Sweep completed: {len(trials)} trials, leaderboard entries={len(leaderboard)}")
    if args.command in {"select", "run-all"}:
        trials_path = exp_cfg.artifacts_dir / "trials.json"
        if not trials_path.is_file():
            raise FileNotFoundError(f"Missing trials file: {trials_path}. Run sweep first.")
        import json

        trials = json.loads(trials_path.read_text(encoding="utf-8"))
        selected = select_top_candidates(exp_cfg, trials)
        print(f"Selected {len(selected)} candidates.")
    if args.command in {"final-eval", "run-all"}:
        final = locked_test_evaluate(exp_cfg)
        print(f"Final test evaluation completed for {len(final)} selected candidates.")


if __name__ == "__main__":
    main()
