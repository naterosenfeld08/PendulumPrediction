"""Plain-text summary of ensemble statistics, thresholds, and inverse result."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from stats.inverse import upper_ci_theta1


def _bootstrap_ci_mean(
    values: np.ndarray,
    config: Mapping[str, Any],
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    n = len(values)
    n_boot = int(config["statistics"]["bootstrap_iterations"])
    conf = float(config["statistics"]["confidence_level"])
    alpha = (1.0 - conf) / 2.0
    means = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[b] = float(np.mean(values[idx]))
    return float(np.mean(values)), float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha))


def build_report_text(
    df: pd.DataFrame,
    threshold_info: Mapping[str, Any],
    inverse_angle_rad: float,
    target_variance: float,
    model: Mapping[str, Any],
    config: Mapping[str, Any],
    breakdown_info: Mapping[str, Any] | None = None,
) -> str:
    """Assemble multi-line report string."""
    n = len(df)
    frac_chaotic = float(df["is_chaotic"].mean())

    reg = df.loc[~df["is_chaotic"], "energy_ratio_variance"].to_numpy()
    cha = df.loc[df["is_chaotic"], "energy_ratio_variance"].to_numpy()

    rng = np.random.default_rng(int(config["statistics"].get("gpr_random_state", 0)) + 404)
    reg_mean, reg_lo, reg_hi = _bootstrap_ci_mean(reg, config, rng) if reg.size else (float("nan"),) * 3
    cha_mean, cha_lo, cha_hi = _bootstrap_ci_mean(cha, config, rng) if cha.size else (float("nan"),) * 3

    th = float(threshold_info["threshold_angle"])
    th_deg = float(threshold_info["threshold_angle_deg"])
    conf = float(threshold_info["confidence_level"])

    report_bs = int(config["statistics"].get("figure_bootstrap_iterations", 80))
    u_at_th = upper_ci_theta1(model, th, config, n_bootstrap=report_bs)
    u_at_inv = upper_ci_theta1(model, float(inverse_angle_rad), config, n_bootstrap=report_bs)

    lines = [
        "Double pendulum ensemble — summary",
        "===================================",
        "",
        f"Pendulums simulated:     {n}",
        f"Fraction chaotic:        {frac_chaotic:.4f}",
        "",
        f"Chaos threshold θ₁₀:     {th:.6f} rad  ({th_deg:.4f} deg)",
        f"  (logistic P(chaotic) = 1 - confidence = {1.0 - conf:.4f} at α = {conf:.4f})",
        "",
        "Energy-ratio variance var(KE₂/KE_tot) over integration window:",
        f"  Regular subset — mean: {reg_mean:.6e},  bootstrap CI: [{reg_lo:.6e}, {reg_hi:.6e}]",
        f"  Chaotic subset — mean: {cha_mean:.6e},  bootstrap CI: [{cha_lo:.6e}, {cha_hi:.6e}]",
        "",
        "Inverse problem (GPR bootstrap upper bound, other features at training medians):",
        f"  Target max variance:   {target_variance:.6e}",
        f"  Implied max θ₁₀:       {inverse_angle_rad:.6f} rad  ({np.degrees(inverse_angle_rad):.4f} deg)",
        "",
        "Diagnostics:",
        f"  GPR upper-CI variance at logistic threshold θ₁₀: {u_at_th:.6e}",
        f"  GPR upper-CI variance at implied max θ₁₀:        {u_at_inv:.6e}",
        "",
    ]
    if breakdown_info is not None:
        t_b = np.asarray(breakdown_info["t_breakdown"], dtype=np.float64)
        t_s = np.asarray(breakdown_info["t_sample"], dtype=np.float64)
        frac_hit = float(np.mean(np.isfinite(t_b))) if t_b.size else 0.0
        med_t = float(np.nanmedian(t_b)) if np.any(np.isfinite(t_b)) else float("nan")
        j_max = int(np.nanargmax(breakdown_info["fraction_outside_ci"]))
        lines.extend(
            [
                "Prediction breakdown (OOF GPR: initial state → KE₂/KE_tot at time t):",
                f"  Interval method:     {breakdown_info.get('interval_method', '?')}",
                f"  CV folds:            {breakdown_info.get('cv_folds', '?')}",
                f"  Share with breakdown: {frac_hit:.4f}  (finite t* before simulation end)",
                f"  Median t* (if any):   {med_t:.4f} s",
                f"  Largest frac. outside CI at t = {t_s[j_max]:.4f} s",
                "",
            ]
        )
    return "\n".join(lines)


def write_report(
    df: pd.DataFrame,
    threshold_info: Mapping[str, Any],
    inverse_angle_rad: float,
    target_variance: float,
    model: Mapping[str, Any],
    config: Mapping[str, Any],
    path: Path | None = None,
    *,
    breakdown_info: Mapping[str, Any] | None = None,
) -> str:
    """Print report and write to ``data/results/summary_report.txt``."""
    text = build_report_text(
        df,
        threshold_info,
        inverse_angle_rad,
        target_variance,
        model,
        config,
        breakdown_info=breakdown_info,
    )
    print(text)
    out = path or (Path(__file__).resolve().parents[2] / "data" / "results" / "summary_report.txt")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return text
