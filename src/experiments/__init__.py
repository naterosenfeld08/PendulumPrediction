"""Experiment orchestration for embedding-first workflows."""

from experiments.evaluation import (
    evaluate_forecasts,
    plot_gap_curves,
    plot_metric_curves,
    plot_residual_histogram,
    save_metrics,
    split_gap_report,
    weighted_rmse,
)
from experiments.runner import (
    ExperimentConfig,
    build_leaderboard,
    generate_data,
    locked_test_evaluate,
    select_top_candidates,
    sweep_train_val,
    train_and_evaluate,
)

__all__ = [
    "ExperimentConfig",
    "build_leaderboard",
    "evaluate_forecasts",
    "generate_data",
    "locked_test_evaluate",
    "plot_gap_curves",
    "plot_metric_curves",
    "plot_residual_histogram",
    "save_metrics",
    "select_top_candidates",
    "split_gap_report",
    "sweep_train_val",
    "train_and_evaluate",
    "weighted_rmse",
]
