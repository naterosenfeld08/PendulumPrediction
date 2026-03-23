"""Orchestrate ensemble simulation, statistics, figures, and summary report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ensemble.ensemble import (  # noqa: E402
    ensemble_results_path,
    load_ensemble_results,
    run_ensemble,
)
from output.report import write_report  # noqa: E402
from output.visualize import generate_all_figures  # noqa: E402
from stats.inverse import angle_from_variance_target  # noqa: E402
from stats.stats import fit_model  # noqa: E402
from stats.threshold import find_chaos_threshold  # noqa: E402


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config root must be a mapping.")
    return cfg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Double pendulum ensemble: simulate, model variance, chaos threshold, plots."
    )
    p.add_argument(
        "--config",
        type=Path,
        default=_ROOT / "config.yaml",
        help="Path to YAML configuration.",
    )
    p.add_argument(
        "--n",
        type=int,
        default=None,
        choices=(500, 1000, 1500),
        help="Override ensemble.n_pendulums (must be 500, 1000, or 1500).",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-run ensemble even if results Parquet already exists.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    config = load_config(config_path)
    if args.n is not None:
        config["ensemble"]["n_pendulums"] = args.n

    out_path = ensemble_results_path(config)
    if args.force or not out_path.is_file():
        import numpy as np

        rng = np.random.default_rng(int(config["ensemble"]["seed"]))
        df = run_ensemble(config, rng=rng)
    else:
        df = load_ensemble_results(out_path)

    model = fit_model(df, config)
    threshold_info = find_chaos_threshold(df, config)
    target_var = float(config["inverse"]["target_variance"])
    inv_angle = angle_from_variance_target(model, target_var, config)

    generate_all_figures(df, model, threshold_info, config)
    write_report(
        df,
        threshold_info,
        inv_angle,
        target_var,
        model,
        config,
    )


if __name__ == "__main__":
    main()
