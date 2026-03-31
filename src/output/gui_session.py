"""Build Manim-compatible session assets from an existing ensemble ``DataFrame``.

Ensures the same parameter rows used for statistics drive the visualization, unlike
``export_manim_session`` which draws a fresh LHS sample.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from physics.energy import compute_energy_timeseries
from physics.integrator import integrate
from physics.pendulum import dstate_dt

# Keep in sync with ``manim_export`` (avoid importing that module from here).
EXPORT_CONTRACT_VERSION = "1.0"
SCENE3_NPZ_REQUIRED_KEYS: tuple[str, ...] = (
    "t",
    "delta_regular",
    "delta_chaotic",
    "mle_regular",
    "mle_chaotic",
    "theta1_grid",
    "p_chaotic_grid",
    "threshold_angle",
    "threshold_angle_deg",
    "confidence_level",
    "p_target",
)


def _wrap_angle(theta: NDArray[np.floating]) -> NDArray[np.floating]:
    return (theta + np.pi) % (2.0 * np.pi) - np.pi


def _integrate_both_trajectories(
    state0: NDArray[np.floating],
    params: Mapping[str, float],
    config: Mapping[str, Any],
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    ly = config["lyapunov"]
    delta0 = float(ly["delta0"])
    t, y_ref = integrate(dstate_dt, state0, params, config)
    state0_p = np.asarray(state0, dtype=np.float64).copy()
    state0_p[0] += delta0
    _, y_pert = integrate(dstate_dt, state0_p, params, config)
    delta = np.linalg.norm(y_ref - y_pert, axis=1)
    return t, y_ref, y_pert, delta


def _params_from_row(row: pd.Series) -> dict[str, float]:
    return {
        "m1": float(row["m1"]),
        "m2": float(row["m2"]),
        "L1": float(row["L1"]),
        "L2": float(row["L2"]),
        "theta1": float(row["theta1"]),
        "theta2": float(row["theta2"]),
        "omega1": float(row["omega1"]),
        "omega2": float(row["omega2"]),
        "g": 9.81,
    }


def _integrate_slot_trajectory(
    p: dict[str, float],
    config: Mapping[str, Any],
    vid_idx: NDArray[np.integer],
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Downsampled θ₁, θ₂, ω₂ on ``vid_idx`` (matches ensemble run used for statistics)."""
    state0 = np.array([p["theta1"], p["omega1"], p["theta2"], p["omega2"]], dtype=np.float64)
    t, y_ref, _, delta = _integrate_both_trajectories(state0, p, config)
    _ = compute_energy_timeseries(t, y_ref, p, config)
    theta1_v = y_ref[vid_idx, 0]
    theta2_v = y_ref[vid_idx, 2]
    omega2_v = y_ref[vid_idx, 3]
    return theta1_v, theta2_v, omega2_v


def export_gui_session_from_dataframe(
    df: pd.DataFrame,
    config: Mapping[str, Any],
    threshold_info: Mapping[str, Any],
    session_dir: Path,
    *,
    rng_seed: int,
    max_visual_runs: int = 48,
    video_frames: int = 120,
    density_frames: int = 16,
    density_bins: tuple[int, int] = (64, 64),
    density_cmap: str = "inferno",
) -> Path:
    """Write ``ensemble_scene_data.npz``, density PNGs, ``delta_threshold_data.npz``, manifest.

    Uses up to ``max_visual_runs`` rows spread across ``df`` for the ensemble animation.
    Logistic / threshold curves come from ``threshold_info`` (already fit on full ``df``).
    Delta(t) curves use the most regular-like and most chaotic-like rows from **full** ``df``.
    """
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "density_frames_regular").mkdir(exist_ok=True)
    (session_dir / "density_frames_chaotic").mkdir(exist_ok=True)

    df = df.reset_index(drop=True)
    n = len(df)
    if n < 1:
        raise ValueError("DataFrame is empty.")

    n_vis = min(int(max_visual_runs), n)
    idx_vis = np.unique(np.linspace(0, n - 1, n_vis, dtype=int))
    n_vis = len(idx_vis)

    integ = config["integration"]
    n_steps = int(integ["n_steps"])
    t_start, t_end = float(integ["t_span"][0]), float(integ["t_span"][1])
    t_grid = np.linspace(t_start, t_end, n_steps + 1, dtype=np.float64)
    video_frames = int(video_frames)
    density_frames = int(density_frames)
    vid_idx = np.linspace(0, len(t_grid) - 1, video_frames, dtype=int)
    den_idx = np.linspace(0, len(t_grid) - 1, density_frames, dtype=int)

    theta1_all = np.empty((n_vis, len(vid_idx)), dtype=np.float64)
    theta2_all = np.empty((n_vis, len(vid_idx)), dtype=np.float64)
    omega2_all = np.empty((n_vis, len(vid_idx)), dtype=np.float64)
    L1_all = np.empty(n_vis, dtype=np.float64)
    L2_all = np.empty(n_vis, dtype=np.float64)
    chaotic = np.zeros(n_vis, dtype=bool)
    mle_all = np.empty(n_vis, dtype=np.float64)
    omega2_collect: list[np.ndarray] = []

    for j, irow in enumerate(idx_vis):
        row = df.iloc[int(irow)]
        p = _params_from_row(row)
        th1, th2, om2 = _integrate_slot_trajectory(p, config, vid_idx)
        theta1_all[j, :] = th1
        theta2_all[j, :] = th2
        omega2_all[j, :] = om2
        L1_all[j] = float(p["L1"])
        L2_all[j] = float(p["L2"])
        chaotic[j] = bool(row["is_chaotic"])
        mle_all[j] = float(row["mle"])
        omega2_collect.append(omega2_all[j, :].copy())

    omega2_collect_all = np.concatenate(omega2_collect, axis=0)
    theta2_wrap = _wrap_angle(theta2_all)
    omega2_abs = np.abs(omega2_collect_all)
    q99 = float(np.quantile(omega2_abs, 0.99))
    if not np.isfinite(q99) or q99 <= 0.0:
        q99 = float(np.max(omega2_abs))
    omega2_min, omega2_max = -1.15 * q99, 1.15 * q99

    n_theta_bins, n_omega_bins = density_bins
    theta_edges = np.linspace(-np.pi, np.pi, n_theta_bins + 1, dtype=np.float64)
    omega_edges = np.linspace(omega2_min, omega2_max, n_omega_bins + 1, dtype=np.float64)

    den_positions = np.searchsorted(vid_idx, den_idx)
    den_positions = np.clip(den_positions, 0, len(vid_idx) - 1)
    unique_positions: list[int] = []
    for pos in den_positions:
        if not unique_positions or pos != unique_positions[-1]:
            unique_positions.append(int(pos))
    den_positions = unique_positions
    density_frames = len(den_positions)

    cmap = cm.get_cmap(density_cmap)
    log_counts_max = 0.0
    for chaos_flag in (False, True):
        counts = np.zeros((n_theta_bins, n_omega_bins), dtype=np.float64)
        for tpos in range(den_positions[-1] + 1):
            mask = chaotic if chaos_flag else ~chaotic
            th_step = theta2_wrap[mask, tpos]
            om_step = omega2_all[mask, tpos]
            if th_step.size == 0:
                continue
            H, _, _ = np.histogram2d(th_step, om_step, bins=[theta_edges, omega_edges])
            counts += H
        log_counts_max = max(log_counts_max, float(np.log1p(counts).max(initial=0.0)))
    log_counts_max = max(log_counts_max, 1.0)

    def save_heatmap_frame(counts: NDArray[np.floating], frame_path: Path) -> None:
        log_img = np.log1p(counts.T) / log_counts_max
        rgba = cmap(np.clip(log_img, 0.0, 1.0))
        plt.imsave(frame_path, rgba, origin="lower")

    for chaos_mode, folder in ((False, "density_frames_regular"), (True, "density_frames_chaotic")):
        mask_base = chaotic if chaos_mode else ~chaotic
        counts = np.zeros((n_theta_bins, n_omega_bins), dtype=np.float64)
        out_dir = session_dir / folder
        frame_counter = 0
        for tpos in range(den_positions[-1] + 1):
            th_step = theta2_wrap[mask_base, tpos]
            om_step = omega2_all[mask_base, tpos]
            if th_step.size:
                H, _, _ = np.histogram2d(th_step, om_step, bins=[theta_edges, omega_edges])
                counts += H
            if tpos in den_positions:
                save_heatmap_frame(counts, out_dir / f"frame_{frame_counter:03d}.png")
                frame_counter += 1

    t_video = t_grid[vid_idx]
    ensemble_npz_path = session_dir / "ensemble_scene_data.npz"
    np.savez_compressed(
        ensemble_npz_path,
        t=t_video,
        theta1=theta1_all,
        theta2=theta2_all,
        omega2=omega2_all,
        L1=L1_all,
        L2=L2_all,
        is_chaotic=chaotic,
        mle=mle_all,
        vid_idx=vid_idx,
        theta_edges=theta_edges,
        omega_edges=omega_edges,
        theta2_wrap=theta2_wrap,
    )

    reg_mask = ~df["is_chaotic"].to_numpy(dtype=bool)
    cha_mask = df["is_chaotic"].to_numpy(dtype=bool)
    if np.any(reg_mask):
        reg_iloc = int(np.argmin(np.where(reg_mask, df["mle"].to_numpy(), np.inf)))
    else:
        reg_iloc = int(df["mle"].values.argmin())
    if np.any(cha_mask):
        cha_iloc = int(np.argmax(np.where(cha_mask, df["mle"].to_numpy(), -np.inf)))
    else:
        cha_iloc = int(df["mle"].values.argmax())

    row_reg = df.iloc[reg_iloc]
    row_cha = df.iloc[cha_iloc]
    p_reg = _params_from_row(row_reg)
    p_cha = _params_from_row(row_cha)

    def _delta_curve(row: pd.Series) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        p = _params_from_row(row)
        state0 = np.array([p["theta1"], p["omega1"], p["theta2"], p["omega2"]], dtype=np.float64)
        t_full, y_ref, _, delta_full = _integrate_both_trajectories(state0, p, config)
        _ = compute_energy_timeseries(t_full, y_ref, p, config)
        return t_video, delta_full[vid_idx].astype(np.float64)

    t_reg, delta_reg = _delta_curve(row_reg)
    _, delta_cha = _delta_curve(row_cha)

    theta_grid = threshold_info["theta_grid"]
    if hasattr(theta_grid, "astype"):
        theta_grid = theta_grid.astype(np.float64)
    p_grid = threshold_info["p_chaotic_grid"]
    if hasattr(p_grid, "astype"):
        p_grid = p_grid.astype(np.float64)

    delta_npz_path = session_dir / "delta_threshold_data.npz"
    np.savez_compressed(
        delta_npz_path,
        t=t_reg,
        delta_regular=delta_reg,
        delta_chaotic=delta_cha,
        mle_regular=float(row_reg["mle"]),
        mle_chaotic=float(row_cha["mle"]),
        reg_run_theta1=float(row_reg["theta1"]),
        cha_run_theta1=float(row_cha["theta1"]),
        threshold_angle=float(threshold_info["threshold_angle"]),
        threshold_angle_deg=float(threshold_info["threshold_angle_deg"]),
        confidence_level=float(threshold_info["confidence_level"]),
        p_target=float(threshold_info["p_target"]),
        theta1_grid=theta_grid,
        p_chaotic_grid=p_grid,
    )

    manifest = {
        "export_contract_version": EXPORT_CONTRACT_VERSION,
        "scene3_npz_required_keys": list(SCENE3_NPZ_REQUIRED_KEYS),
        "n_ensemble": int(n_vis),
        "n_statistical_runs": int(n),
        "rng_seed": int(rng_seed),
        "t_video_length": int(len(t_video)),
        "density_frames": int(density_frames),
        "video_frames": int(video_frames),
        "density_bins": [int(n_theta_bins), int(n_omega_bins)],
        "theta2_range": [-math.pi, math.pi],
        "omega2_range": [float(omega2_min), float(omega2_max)],
        "ensemble_npz": str(ensemble_npz_path.name),
        "delta_threshold_npz": str(delta_npz_path.name),
        "density_frame_regular_dir": "density_frames_regular",
        "density_frame_chaotic_dir": "density_frames_chaotic",
        "source": "gui_session_from_dataframe",
    }
    (session_dir / "session_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    npz_path = session_dir / str(manifest["delta_threshold_npz"])
    z = np.load(npz_path)
    missing = [k for k in SCENE3_NPZ_REQUIRED_KEYS if k not in z.files]
    if missing:
        raise ValueError(f"delta_threshold_data.npz missing keys: {missing}")
    return session_dir
