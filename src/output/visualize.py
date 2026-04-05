"""Matplotlib diagnostics saved under ``data/results/``."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from physics.integrator import integrate
from physics.pendulum import dstate_dt
from stats.stats import predict_with_ci
from stats.threshold import chaos_probability_vs_theta


def _results_dir(config: Mapping[str, Any] | None = None) -> Path:
    root = Path(__file__).resolve().parents[2]
    d = root / "data" / "results"
    d.mkdir(parents=True, exist_ok=True)
    return d


def plot_theta1_vs_mle(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    reg = df.loc[~df["is_chaotic"]]
    cha = df.loc[df["is_chaotic"]]
    ax.scatter(reg["theta1"], reg["mle"], c="C0", s=12, alpha=0.65, label="regular")
    ax.scatter(cha["theta1"], cha["mle"], c="C3", s=12, alpha=0.65, label="chaotic")
    ax.set_xlabel(r"$\theta_1\ \mathrm{(rad)}$")
    ax.set_ylabel("MLE estimate")
    ax.legend()
    ax.grid(True, ls=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_dir / "scatter_theta1_mle.png", dpi=150)
    plt.close(fig)


def plot_theta1_vs_variance_with_ci(
    df: pd.DataFrame,
    model: Mapping[str, Any],
    config: Mapping[str, Any],
    out_dir: Path,
) -> None:
    th = np.linspace(
        float(config["parameters"]["theta1"][0]),
        float(config["parameters"]["theta1"][1]),
        80,
    )
    med = np.median(model["X_train"], axis=0)
    X_rows = np.tile(med, (len(th), 1))
    idx = model["feature_order"].index("theta1")
    X_rows[:, idx] = th
    fig_bs = int(config["statistics"].get("figure_bootstrap_iterations", 80))
    mean_p, lo, hi = predict_with_ci(model, X_rows, config, n_bootstrap=fig_bs)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(df["theta1"], df["energy_ratio_variance"], s=14, alpha=0.35, c="0.4")
    ax.plot(th, mean_p, color="C0", lw=2, label="GPR bootstrap mean")
    ax.fill_between(th, lo, hi, color="C0", alpha=0.2, label="Bootstrap CI")
    ax.set_xlabel(r"$\theta_1\ \mathrm{(rad)}$")
    ax.set_ylabel(r"$\mathrm{var}(\mathrm{KE}_2/\mathrm{KE}_{\mathrm{tot}})$")
    ax.legend()
    ax.grid(True, ls=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_dir / "scatter_theta1_variance_ci.png", dpi=150)
    plt.close(fig)


def plot_mle_histogram(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df["mle"], bins=40, color="C2", edgecolor="white", alpha=0.9)
    ax.set_xlabel("MLE estimate")
    ax.set_ylabel("count")
    ax.grid(True, ls=":", alpha=0.5, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "hist_mle.png", dpi=150)
    plt.close(fig)


def _integrate_row(row: pd.Series, config: Mapping[str, Any]) -> NDArray[np.floating]:
    params = {
        "m1": float(row["m1"]),
        "m2": float(row["m2"]),
        "L1": float(row["L1"]),
        "L2": float(row["L2"]),
        "g": 9.81,
    }
    state0 = np.array(
        [row["theta1"], row["omega1"], row["theta2"], row["omega2"]],
        dtype=np.float64,
    )
    _, y = integrate(dstate_dt, state0, params, config)
    return y


def plot_phase_portraits_three_runs(
    df: pd.DataFrame,
    config: Mapping[str, Any],
    threshold_angle: float,
    out_dir: Path,
) -> None:
    reg = df.loc[~df["is_chaotic"]]
    cha = df.loc[df["is_chaotic"]]
    if reg.empty or cha.empty:
        return

    i_reg = int(reg["mle"].idxmin())
    i_cha = int(cha["mle"].idxmax())
    mid = df.iloc[(df["theta1"] - threshold_angle).abs().argsort()[:1]].index[0]

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharex=False, sharey=False)
    titles = ["regular (low MLE)", "near threshold", "chaotic (high MLE)"]
    idxs = [i_reg, mid, i_cha]
    for ax, title, ix in zip(axes, titles, idxs):
        y = _integrate_row(df.loc[ix], config)
        ax.plot(y[:, 0], y[:, 1], lw=0.8, color="C0")
        ax.set_xlabel(r"$\theta_1$")
        ax.set_ylabel(r"$\omega_1$")
        ax.set_title(title)
        ax.grid(True, ls=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_dir / "phase_portraits_three.png", dpi=150)
    plt.close(fig)


def plot_logistic_threshold(
    threshold_info: Mapping[str, Any],
    config: Mapping[str, Any],
    out_dir: Path,
) -> None:
    th = np.linspace(
        float(config["parameters"]["theta1"][0]),
        float(config["parameters"]["theta1"][1]),
        400,
    )
    clf = threshold_info["logistic_model"]
    if clf is None:
        p_val = float(threshold_info["p_chaotic_grid"][0])
        p = np.full_like(th, p_val, dtype=float)
    else:
        p = chaos_probability_vs_theta(clf, th)
    threshold_angle = float(threshold_info["threshold_angle"])
    p_target = float(threshold_info["p_target"])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(th, p, color="C1", lw=2, label=r"$P(\mathrm{chaotic}\mid \theta_1)$")
    ax.axhline(
        p_target,
        color="0.5",
        ls="--",
        lw=1,
        label=rf"$1-\alpha = {p_target:.3f}$",
    )
    ax.axvline(threshold_angle, color="C3", ls="--", lw=1, label="threshold angle")
    ax.set_xlabel(r"$\theta_1\ \mathrm{(rad)}$")
    ax.set_ylabel("probability")
    ax.legend()
    ax.grid(True, ls=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_dir / "logistic_chaos_threshold.png", dpi=150)
    plt.close(fig)


def plot_prediction_breakdown(
    breakdown_info: Mapping[str, Any],
    out_dir: Path,
) -> None:
    """Sharpness (OOF interval width) and miscalibration proxy vs time."""
    t = np.asarray(breakdown_info["t_sample"], dtype=np.float64)
    w = np.asarray(breakdown_info["median_interval_width"], dtype=np.float64)
    f_out = np.asarray(breakdown_info["fraction_outside_ci"], dtype=np.float64)

    fig, ax1 = plt.subplots(figsize=(8.2, 4.6))
    ax1.plot(t, w, color="C0", lw=2.2, label="median OOF CI width")
    ax1.fill_between(t, 0.0, w, color="C0", alpha=0.12)
    ax1.set_xlabel("t (s)")
    ax1.set_ylabel(r"width of $KE_2/KE_{\mathrm{tot}}$ interval")
    ax1.grid(True, ls=":", alpha=0.5)

    ax2 = ax1.twinx()
    ax2.plot(t, f_out, color="C3", lw=2.0, ls="--", label="fraction outside OOF CI")
    ax2.set_ylabel("fraction outside interval")
    mx = float(np.nanmax(f_out))
    if not np.isfinite(mx):
        mx = 0.0
    ax2.set_ylim(0.0, max(0.05, mx * 1.15))

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", framealpha=0.92)
    fig.suptitle(
        "Prediction sharpness vs time (OOF GPR: initial state → instantaneous energy share)",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "prediction_breakdown_sharpness.png", dpi=150)
    plt.close(fig)


def plot_breakdown_time_vs_mle(
    df: pd.DataFrame,
    breakdown_info: Mapping[str, Any],
    out_dir: Path,
    *,
    t_end: float,
) -> None:
    """First time truth leaves OOF interval vs finite-time MLE proxy."""
    t_b = np.asarray(breakdown_info["t_breakdown"], dtype=np.float64)
    mle = df["mle"].to_numpy(dtype=np.float64)
    mask = np.isfinite(t_b)

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    if np.any(mask):
        ax.scatter(mle[mask], t_b[mask], s=22, alpha=0.55, c="C2", edgecolors="none")
    n_no = int(np.sum(~np.isfinite(t_b)))
    if n_no:
        ax.text(
            0.02,
            0.98,
            f"{n_no} run(s): no breakdown before t={t_end:.1f} s",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            color="0.35",
        )
    ax.set_xlabel("MLE estimate")
    ax.set_ylabel(r"breakdown time $t^*$ (s)")
    ax.grid(True, ls=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_dir / "prediction_breakdown_vs_mle.png", dpi=150)
    plt.close(fig)


def generate_all_figures(
    df: pd.DataFrame,
    model: Mapping[str, Any],
    threshold_info: Mapping[str, Any],
    config: Mapping[str, Any],
    out_dir: Path | None = None,
    *,
    breakdown_info: Mapping[str, Any] | None = None,
) -> None:
    """Produce standard figures; optional breakdown diagnostics."""
    out = out_dir or _results_dir(config)
    plot_theta1_vs_mle(df, out)
    plot_theta1_vs_variance_with_ci(df, model, config, out)
    plot_mle_histogram(df, out)
    plot_logistic_threshold(threshold_info, config, out)
    plot_phase_portraits_three_runs(
        df,
        config,
        float(threshold_info["threshold_angle"]),
        out,
    )
    if breakdown_info is not None:
        t_end = float(config["integration"]["t_span"][1])
        plot_prediction_breakdown(breakdown_info, out)
        plot_breakdown_time_vs_mle(df, breakdown_info, out, t_end=t_end)
