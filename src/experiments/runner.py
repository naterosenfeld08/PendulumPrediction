"""Long-run training, model search, selection, and locked test evaluation."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from numpy.typing import NDArray

from data import GenerationConfig, generate_trajectory_corpus
from embeddings import build_embedder
from experiments.evaluation import (
    evaluate_forecasts,
    plot_gap_curves,
    plot_metric_curves,
    plot_residual_histogram,
    save_metrics,
    split_gap_report,
    weighted_rmse,
)
from models import build_model, model_names, sample_hyperparameters
from tasks import ForecastTaskConfig, SupervisedDataset, build_supervised_dataset


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level config for generation, search, selection, and final evaluation."""

    trajectories_dir: Path
    artifacts_dir: Path
    n_per_system: int = 80
    duration_s: float = 30.0
    n_steps: int = 1200
    seed: int = 42
    train_frac: float = 0.7
    val_frac: float = 0.2
    window_size: int = 64
    horizons: tuple[int, ...] = (1, 5, 10, 20)
    stride: int = 4
    seed_grid: tuple[int, ...] = (11, 29, 47, 71, 97)
    embedding_list: tuple[str, ...] = (
        "physics_features_v1",
        "raw_window",
        "fft_features",
        "hybrid_physics_fft",
    )
    model_list: tuple[str, ...] = (
        "persistence",
        "linear",
        "ridge",
        "random_forest",
        "gradient_boosting",
        "mlp_small",
        "mlp_medium",
    )
    trials_per_model: int = 20
    max_parallel_jobs: int = 1
    primary_metric: str = "weighted_rmse"
    horizon_weights: tuple[float, ...] = (0.45, 0.30, 0.15, 0.10)
    top_k: int = 3


def generate_data(base_config: dict, cfg: ExperimentConfig) -> dict[str, int]:
    """Generate trajectory corpus with configured split fractions."""
    gen_cfg = GenerationConfig(
        output_dir=cfg.trajectories_dir,
        n_per_system=cfg.n_per_system,
        duration_s=cfg.duration_s,
        n_steps=cfg.n_steps,
        seed=cfg.seed,
        train_frac=cfg.train_frac,
        val_frac=cfg.val_frac,
    )
    return generate_trajectory_corpus(base_config=base_config, cfg=gen_cfg)


def _task_cfg(cfg: ExperimentConfig) -> ForecastTaskConfig:
    return ForecastTaskConfig(
        window_size=cfg.window_size,
        horizons=cfg.horizons,
        stride=cfg.stride,
    )


def _load_split_dataset(cfg: ExperimentConfig, embedding_name: str, split: str) -> SupervisedDataset:
    embedder = build_embedder(embedding_name)
    return build_supervised_dataset(
        trajectories_dir=cfg.trajectories_dir,
        split=split,
        task_cfg=_task_cfg(cfg),
        embedder=embedder,
    )


def _predict(
    model: Any,
    X: NDArray[np.floating],
    last_energy: NDArray[np.floating],
    model_name: str,
) -> NDArray[np.floating]:
    if model_name == "persistence":
        return model.predict(X, last_energy=last_energy)
    return model.predict(X)


def _objective(metrics: dict[str, object], cfg: ExperimentConfig) -> float:
    if cfg.primary_metric != "weighted_rmse":
        raise ValueError(f"Unsupported primary metric: {cfg.primary_metric}")
    score = weighted_rmse(metrics, horizon_weights=cfg.horizon_weights)
    if not np.isfinite(score):
        return float("inf")
    return float(score)


def _record_trial(
    *,
    embedding_name: str,
    model_name: str,
    seed: int,
    hyperparams: dict[str, Any],
    train_metrics: dict[str, object],
    val_metrics: dict[str, object],
    objective: float,
) -> dict[str, Any]:
    return {
        "embedding": embedding_name,
        "model": model_name,
        "seed": int(seed),
        "hyperparams": hyperparams,
        "objective": float(objective),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
    }


def sweep_train_val(cfg: ExperimentConfig) -> list[dict[str, Any]]:
    """Run randomized search over embeddings/models using train+val only."""
    cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)
    trials: list[dict[str, Any]] = []

    for embedding_name in cfg.embedding_list:
        train_ds = _load_split_dataset(cfg, embedding_name, split="train")
        val_ds = _load_split_dataset(cfg, embedding_name, split="val")
        for model_name in cfg.model_list:
            if model_name not in model_names():
                continue
            for seed in cfg.seed_grid:
                rng = np.random.default_rng(seed)
                n_trials = 1 if model_name in {"persistence", "linear"} else cfg.trials_per_model
                for _ in range(n_trials):
                    hyperparams = sample_hyperparameters(model_name, rng)
                    model = build_model(
                        model_name=model_name,
                        horizons=cfg.horizons,
                        random_state=seed,
                        hyperparams=hyperparams,
                    )
                    model.fit(train_ds.X, train_ds.y)
                    y_train = _predict(model, train_ds.X, train_ds.last_energy, model_name)
                    y_val = _predict(model, val_ds.X, val_ds.last_energy, model_name)
                    train_metrics = evaluate_forecasts(train_ds.y, y_train, horizons=cfg.horizons)
                    val_metrics = evaluate_forecasts(val_ds.y, y_val, horizons=cfg.horizons)
                    obj = _objective(val_metrics, cfg)
                    trials.append(
                        _record_trial(
                            embedding_name=embedding_name,
                            model_name=model_name,
                            seed=seed,
                            hyperparams=hyperparams,
                            train_metrics=train_metrics,
                            val_metrics=val_metrics,
                            objective=obj,
                        )
                    )

    trials_path = cfg.artifacts_dir / "trials.json"
    trials_path.write_text(json.dumps(trials, indent=2), encoding="utf-8")
    return trials


def select_top_candidates(cfg: ExperimentConfig, trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select top-K candidates by validation objective only."""
    cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)
    ranked = sorted(trials, key=lambda t: float(t["objective"]))
    selected = ranked[: cfg.top_k]
    (cfg.artifacts_dir / "selected_candidates.json").write_text(
        json.dumps(selected, indent=2), encoding="utf-8"
    )
    return selected


def _merge_train_val(a: SupervisedDataset, b: SupervisedDataset) -> SupervisedDataset:
    return SupervisedDataset(
        X=np.vstack([a.X, b.X]),
        y=np.concatenate([a.y, b.y], axis=0),
        last_energy=np.vstack([a.last_energy, b.last_energy]),
        trajectory_ids=[*a.trajectory_ids, *b.trajectory_ids],
    )


def locked_test_evaluate(
    cfg: ExperimentConfig,
    selected: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Retrain selected candidates on train+val and evaluate once on test."""
    cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)
    if selected is None:
        sel_path = cfg.artifacts_dir / "selected_candidates.json"
        if not sel_path.is_file():
            raise FileNotFoundError("selected_candidates.json not found. Run select first.")
        selected = json.loads(sel_path.read_text(encoding="utf-8"))

    results: list[dict[str, Any]] = []
    for i, cand in enumerate(selected):
        embedding_name = str(cand["embedding"])
        model_name = str(cand["model"])
        seed = int(cand["seed"])
        hyperparams = dict(cand.get("hyperparams", {}))

        train_ds = _load_split_dataset(cfg, embedding_name, split="train")
        val_ds = _load_split_dataset(cfg, embedding_name, split="val")
        train_val_ds = _merge_train_val(train_ds, val_ds)
        test_ds = _load_split_dataset(cfg, embedding_name, split="test")

        model = build_model(
            model_name=model_name,
            horizons=cfg.horizons,
            random_state=seed,
            hyperparams=hyperparams,
        )
        model.fit(train_val_ds.X, train_val_ds.y)

        y_train = _predict(model, train_val_ds.X, train_val_ds.last_energy, model_name)
        y_test = _predict(model, test_ds.X, test_ds.last_energy, model_name)
        y_val = _predict(model, val_ds.X, val_ds.last_energy, model_name)

        train_metrics = evaluate_forecasts(train_val_ds.y, y_train, horizons=cfg.horizons)
        val_metrics = evaluate_forecasts(val_ds.y, y_val, horizons=cfg.horizons)
        test_metrics = evaluate_forecasts(test_ds.y, y_test, horizons=cfg.horizons)
        gap = split_gap_report(train_metrics, val_metrics, test_metrics)

        result = {
            "rank": i + 1,
            "embedding": embedding_name,
            "model": model_name,
            "seed": seed,
            "hyperparams": hyperparams,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "test_objective": _objective(test_metrics, cfg),
            "gap_report": gap,
        }
        results.append(result)

        model_dir = cfg.artifacts_dir / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_dir / f"rank_{i+1}_{embedding_name}_{model_name}.joblib")

        diag_dir = cfg.artifacts_dir / "diagnostics" / f"rank_{i+1}_{embedding_name}_{model_name}"
        diag_dir.mkdir(parents=True, exist_ok=True)
        save_metrics(train_metrics, diag_dir / "train_metrics.json")
        save_metrics(val_metrics, diag_dir / "val_metrics.json")
        save_metrics(test_metrics, diag_dir / "test_metrics.json")
        (diag_dir / "gap_report.json").write_text(json.dumps(gap, indent=2), encoding="utf-8")
        plot_gap_curves(train_metrics, val_metrics, test_metrics, diag_dir / "gap_curves.png")
        plot_residual_histogram(test_ds.y, y_test, diag_dir / "test_residual_hist.png")

    out = cfg.artifacts_dir / "final_eval.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def build_leaderboard(cfg: ExperimentConfig, trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate trials by embedding/model and write leaderboard files."""
    cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for t in trials:
        key = (str(t["embedding"]), str(t["model"]))
        val_obj = float(t["objective"])
        if key not in rows or val_obj < float(rows[key]["best_objective"]):
            rows[key] = {
                "embedding": key[0],
                "model": key[1],
                "best_objective": val_obj,
                "seed": int(t["seed"]),
                "hyperparams": t["hyperparams"],
            }
    leaderboard = sorted(rows.values(), key=lambda r: float(r["best_objective"]))

    (cfg.artifacts_dir / "leaderboard.json").write_text(
        json.dumps(leaderboard, indent=2), encoding="utf-8"
    )
    with (cfg.artifacts_dir / "leaderboard.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["embedding", "model", "best_objective", "seed", "hyperparams"]
        )
        writer.writeheader()
        for row in leaderboard:
            writer.writerow(
                {
                    "embedding": row["embedding"],
                    "model": row["model"],
                    "best_objective": row["best_objective"],
                    "seed": row["seed"],
                    "hyperparams": json.dumps(row["hyperparams"], sort_keys=True),
                }
            )
    return leaderboard


def train_and_evaluate(cfg: ExperimentConfig) -> dict[str, dict[str, object]]:
    """Compatibility wrapper for quick runs (single trial per model)."""
    small_cfg = ExperimentConfig(
        **{
            **cfg.__dict__,
            "embedding_list": ("physics_features_v1",),
            "model_list": ("persistence", "linear", "mlp_small"),
            "seed_grid": (cfg.seed,),
            "trials_per_model": 1,
            "top_k": 1,
        }
    )
    trials = sweep_train_val(small_cfg)
    selected = select_top_candidates(small_cfg, trials)
    final = locked_test_evaluate(small_cfg, selected=selected)
    by_model = {f'{r["embedding"]}:{r["model"]}': r["test_metrics"] for r in final}
    plot_metric_curves(by_model, cfg.artifacts_dir / "horizon_errors.png")
    return by_model
