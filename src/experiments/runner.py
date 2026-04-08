"""Experiment runner for trajectory embedding forecasts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from numpy.typing import NDArray

from data import GenerationConfig, generate_trajectory_corpus
from embeddings import PhysicsFeatureEmbedder
from experiments.evaluation import evaluate_forecasts, plot_metric_curves, save_metrics
from models import PersistenceBaseline, build_linear_model, build_mlp_model
from tasks import ForecastTaskConfig, build_supervised_dataset


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level config for generation/training/evaluation."""

    trajectories_dir: Path
    artifacts_dir: Path
    n_per_system: int = 80
    duration_s: float = 30.0
    n_steps: int = 1200
    seed: int = 42
    train_frac: float = 0.7
    val_frac: float = 0.15
    window_size: int = 64
    horizons: tuple[int, ...] = (1, 5, 10, 20)
    stride: int = 4


def generate_data(base_config: dict, cfg: ExperimentConfig) -> dict[str, int]:
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


def _train_models(
    X_train: NDArray[np.floating],
    y_train: NDArray[np.floating],
    horizons: tuple[int, ...],
    seed: int,
) -> dict[str, object]:
    models: dict[str, object] = {
        "persistence": PersistenceBaseline(horizons=horizons),
        "linear": build_linear_model(horizons=horizons),
        "mlp": build_mlp_model(horizons=horizons, random_state=seed),
    }
    for name, model in models.items():
        if name == "persistence":
            model.fit(X_train, y_train)
        else:
            model.fit(X_train, y_train)
    return models


def train_and_evaluate(cfg: ExperimentConfig) -> dict[str, dict[str, object]]:
    """Train baseline models and evaluate on test split."""
    embedder = PhysicsFeatureEmbedder()
    task_cfg = ForecastTaskConfig(
        window_size=cfg.window_size,
        horizons=cfg.horizons,
        stride=cfg.stride,
    )
    train_ds = build_supervised_dataset(
        trajectories_dir=cfg.trajectories_dir,
        split="train",
        task_cfg=task_cfg,
        embedder=embedder,
    )
    test_ds = build_supervised_dataset(
        trajectories_dir=cfg.trajectories_dir,
        split="test",
        task_cfg=task_cfg,
        embedder=embedder,
    )

    models = _train_models(
        X_train=train_ds.X,
        y_train=train_ds.y,
        horizons=cfg.horizons,
        seed=cfg.seed,
    )
    metrics_by_model: dict[str, dict[str, object]] = {}
    cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        if name == "persistence":
            y_pred = model.predict(test_ds.X, last_energy=test_ds.last_energy)
        else:
            y_pred = model.predict(test_ds.X)
        metrics = evaluate_forecasts(
            y_true=test_ds.y,
            y_pred=y_pred,
            horizons=cfg.horizons,
        )
        metrics_by_model[name] = metrics
        save_metrics(metrics, cfg.artifacts_dir / f"{name}_metrics.json")
        joblib.dump(model, cfg.artifacts_dir / f"{name}.joblib")

    plot_metric_curves(metrics_by_model, cfg.artifacts_dir / "horizon_errors.png")
    (cfg.artifacts_dir / "experiment_config.json").write_text(
        json.dumps(
            {
                "n_per_system": cfg.n_per_system,
                "duration_s": cfg.duration_s,
                "n_steps": cfg.n_steps,
                "window_size": cfg.window_size,
                "horizons": list(cfg.horizons),
                "stride": cfg.stride,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return metrics_by_model
