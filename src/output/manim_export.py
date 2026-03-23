"""Export precomputed trajectories and density frames for Manim scenes.

Manim renders from exported assets so it does not re-run physics while animating.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from ensemble.sampler import sample_parameters
from physics.energy import compute_energy_timeseries
from physics.integrator import integrate
from physics.pendulum import dstate_dt
from stats.threshold import find_chaos_threshold


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _results_dir() -> Path:
    d = _project_root() / "data" / "results"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _wrap_angle(theta: NDArray[np.floating]) -> NDArray[np.floating]:
    """Wrap angle to [-pi, pi] for stable phase-density visualization."""
    return (theta + np.pi) % (2.0 * np.pi) - np.pi


def _integrate_both_trajectories(
    state0: NDArray[np.floating],
    params: Mapping[str, float],
    config: Mapping[str, Any],
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Integrate reference and θ1-perturbed trajectories.

    Returns
    -------
    t
        Shape (n_points,).
    y_ref
        Shape (n_points, 4).
    y_pert
        Shape (n_points, 4).
    delta
        Euclidean separation at each time: ||y_ref - y_pert||.
    """
    ly = config["lyapunov"]
    delta0 = float(ly["delta0"])

    t, y_ref = integrate(dstate_dt, state0, params, config)

    state0_p = np.asarray(state0, dtype=np.float64).copy()
    state0_p[0] += delta0
    _, y_pert = integrate(dstate_dt, state0_p, params, config)

    delta = np.linalg.norm(y_ref - y_pert, axis=1)
    return t, y_ref, y_pert, delta


def _compute_mle_from_delta(
    t: NDArray[np.floating],
    delta: NDArray[np.floating],
    config: Mapping[str, Any],
) -> float:
    ly = config["lyapunov"]
    delta0 = float(ly["delta0"])
    t_lyapunov = float(ly["t_lyapunov"])
    delta_end = float(np.interp(t_lyapunov, t, delta))
    return (1.0 / t_lyapunov) * float(np.log(delta_end / delta0))


def _bob_positions(
    theta1: NDArray[np.floating],
    theta2: NDArray[np.floating],
    L1: float,
    L2: float,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Convert angles to (x1,y1,x2,y2) with downward vertical angles."""
    x1 = L1 * np.sin(theta1)
    y1 = -L1 * np.cos(theta1)
    x2 = x1 + L2 * np.sin(theta2)
    y2 = y1 - L2 * np.cos(theta2)
    return x1, y1, x2, y2


def export_manim_session(
    config: Mapping[str, Any],
    session_dir: Path,
    rng_seed: int,
    *,
    n_ensemble: int = 32,
    video_frames: int = 120,
    density_frames: int = 20,
    density_bins: tuple[int, int] = (80, 80),
    density_cmap: str = "inferno",
) -> Path:
    """Export all assets needed for three Manim scenes.

    Assets produced under `session_dir/`:
    - `session_manifest.json`
    - `ensemble_scene_data.npz`
    - `density_frames_regular/` and `density_frames_chaotic/` (PNG heatmaps)
    - `delta_threshold_data.npz`
    """
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "density_frames_regular").mkdir(exist_ok=True)
    (session_dir / "density_frames_chaotic").mkdir(exist_ok=True)

    rng = np.random.default_rng(int(rng_seed))
    params_seed = sample_parameters(n_ensemble, config, rng)

    integ = config["integration"]
    n_steps = int(integ["n_steps"])
    t_start, t_end = float(integ["t_span"][0]), float(integ["t_span"][1])
    # Integration grid: integrate() uses t_eval of length n_steps+1.
    t_grid = np.linspace(t_start, t_end, n_steps + 1, dtype=np.float64)
    video_frames = int(video_frames)
    density_frames = int(density_frames)
    if video_frames < 2:
        raise ValueError("video_frames must be >= 2")
    if density_frames < 2:
        raise ValueError("density_frames must be >= 2")

    # Indices to sample the integration grid
    vid_idx = np.linspace(0, len(t_grid) - 1, video_frames, dtype=int)
    den_idx = np.linspace(0, len(t_grid) - 1, density_frames, dtype=int)

    theta1_all = np.empty((n_ensemble, len(vid_idx)), dtype=np.float64)
    theta2_all = np.empty((n_ensemble, len(vid_idx)), dtype=np.float64)
    omega2_all = np.empty((n_ensemble, len(vid_idx)), dtype=np.float64)
    L1_all = np.empty(n_ensemble, dtype=np.float64)
    L2_all = np.empty(n_ensemble, dtype=np.float64)
    chaotic = np.zeros(n_ensemble, dtype=bool)
    mle_all = np.empty(n_ensemble, dtype=np.float64)

    # Collect omega2 distribution for density binning
    omega2_collect: list[np.ndarray] = []

    # Phase density accumulation uses all runs split by chaos label.
    # Keep the strict energy assertion, but resample runs that fail it so GUI export remains robust.
    valid_params: list[dict[str, float]] = []
    max_attempts_per_slot = 30
    for i in range(n_ensemble):
        base = params_seed[i]
        accepted = False
        for attempt in range(max_attempts_per_slot):
            p = base if attempt == 0 else sample_parameters(1, config, rng)[0]
            state0 = np.array(
                [p["theta1"], p["omega1"], p["theta2"], p["omega2"]],
                dtype=np.float64,
            )
            try:
                # Integrate ref and perturbed; compute delta and MLE (directional perturbation).
                t, y_ref, _, delta = _integrate_both_trajectories(state0, p, config)
                # Hard energy conservation check (raises if drift exceeds tolerance).
                _ = compute_energy_timeseries(t, y_ref, p, config)
            except AssertionError:
                continue

            L1_all[i] = float(p["L1"])
            L2_all[i] = float(p["L2"])
            mle = _compute_mle_from_delta(t, delta, config)
            mle_all[i] = mle
            eps = float(config["lyapunov"]["epsilon"])
            chaotic[i] = mle > eps

            # Store downsampled coordinates for visualization.
            theta1_all[i, :] = y_ref[vid_idx, 0]
            theta2_all[i, :] = y_ref[vid_idx, 2]
            omega2_all[i, :] = y_ref[vid_idx, 3]
            omega2_collect.append(omega2_all[i, :].copy())
            valid_params.append(p)
            accepted = True
            break

        if not accepted:
            raise RuntimeError(
                "Unable to sample a valid energy-conserving trajectory for Manim export "
                f"after {max_attempts_per_slot} attempts at slot {i}. "
                "Try narrowing parameter ranges or increasing integration resolution."
            )

    omega2_collect_all = np.concatenate(omega2_collect, axis=0)

    # Density binning: theta2 wrapped to [-pi, pi]; omega2 chosen from 1%..99% quantiles.
    theta2_wrap = _wrap_angle(theta2_all)
    omega2_abs = np.abs(omega2_collect_all)
    q99 = float(np.quantile(omega2_abs, 0.99))
    if not np.isfinite(q99) or q99 <= 0.0:
        q99 = float(np.max(omega2_abs))
    omega2_min, omega2_max = -1.15 * q99, 1.15 * q99

    n_theta_bins, n_omega_bins = density_bins
    theta_edges = np.linspace(-np.pi, np.pi, n_theta_bins + 1, dtype=np.float64)
    omega_edges = np.linspace(omega2_min, omega2_max, n_omega_bins + 1, dtype=np.float64)

    # Precompute density frames by incrementally accumulating histogram counts in time.
    # We use vid_idx for the stored data and den_idx for output frames.
    # Map den_idx to corresponding vid positions (since vid_idx is a subset of t_grid).
    # den_idx values are indexes into t_grid; convert to positions within vid_idx.
    den_positions = np.searchsorted(vid_idx, den_idx)
    den_positions = np.clip(den_positions, 0, len(vid_idx) - 1)

    # Ensure increasing unique frame indices.
    unique_positions: list[int] = []
    for pos in den_positions:
        if not unique_positions or pos != unique_positions[-1]:
            unique_positions.append(int(pos))

    den_positions = unique_positions
    # If we reduced unique frames, adjust density_frames accordingly for manifest.
    density_frames = len(den_positions)

    cmap = cm.get_cmap(density_cmap)
    log_counts_max = 0.0

    # First pass: compute maximum log scale for stable color normalization.
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

    def save_heatmap_frame(
        counts: NDArray[np.floating],
        frame_path: Path,
    ) -> None:
        """Save a single heatmap image (no axes) as PNG."""
        # `counts` is indexed as (theta_bin, omega_bin). Convert to an image array
        # indexed as (row_y, col_x) by transposing to (omega_bin, theta_bin).
        log_img = np.log1p(counts.T) / log_counts_max
        rgba = cmap(np.clip(log_img, 0.0, 1.0))
        # rgba shape: (n_omega_bins, n_theta_bins, 4)
        plt.imsave(
            frame_path,
            rgba,
            origin="lower",
        )

    # Build per-chaos-frame series
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
                frame_path = out_dir / f"frame_{frame_counter:03d}.png"
                save_heatmap_frame(counts, frame_path)
                frame_counter += 1

    # Export ensemble scene data
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

    # Prepare delta + threshold data.
    # Choose one clearly regular and one clearly chaotic run.
    if np.any(~chaotic):
        reg_i = int(np.argmin(np.where(~chaotic, mle_all, np.inf)))
    else:
        reg_i = int(np.argmin(mle_all))
    if np.any(chaotic):
        cha_i = int(np.argmax(np.where(chaotic, mle_all, -np.inf)))
    else:
        cha_i = int(np.argmax(mle_all))

    def _extract_delta_curve(i: int) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        p = valid_params[i]
        state0 = np.array([p["theta1"], p["omega1"], p["theta2"], p["omega2"]], dtype=np.float64)
        t_full, y_ref, _, delta_full = _integrate_both_trajectories(state0, p, config)
        # Ensure this second integration run also passes the strict energy check.
        _ = compute_energy_timeseries(t_full, y_ref, p, config)
        # Downsample delta using vid_idx so it matches the rest of the session.
        delta_vid = delta_full[vid_idx]
        return t_video, delta_vid

    # NOTE: We integrate again here for the two selected trajectories to keep the exporter simple.
    t_reg, delta_reg = _extract_delta_curve(reg_i)
    t_cha, delta_cha = _extract_delta_curve(cha_i)

    # Logistic chaos threshold uses θ1 -> is_chaotic.
    df_threshold = {
            "theta1": np.array([p["theta1"] for p in valid_params], dtype=np.float64),
        "is_chaotic": chaotic.astype(int),
        # Dummy columns required by threshold API? We only use theta1 and is_chaotic.
    }
    # Manually construct minimal DataFrame without adding pandas dependency here.
    import pandas as pd

    df = pd.DataFrame(
        {
            "theta1": df_threshold["theta1"],
            "is_chaotic": df_threshold["is_chaotic"].astype(bool),
        }
    )
    try:
        threshold_info = find_chaos_threshold(df, config)
    except ValueError:
        # Logistic regression needs both classes. If a tiny ensemble produces
        # only regular or only chaotic samples, fall back to a degenerate model.
        st = config["statistics"]
        conf = float(st["confidence_level"])
        p_target = 1.0 - conf

        theta1_min = float(config["parameters"]["theta1"][0])
        theta1_max = float(config["parameters"]["theta1"][1])
        theta_grid = np.linspace(theta1_min, theta1_max, 512, dtype=np.float64)

        all_chaotic = bool(df["is_chaotic"].all())
        p_const = 1.0 if all_chaotic else 0.0
        p_grid = np.full_like(theta_grid, p_const, dtype=np.float64)

        # Choose a threshold consistent with the (degenerate) constant probability.
        if p_const >= p_target:
            threshold_angle = theta1_min
        else:
            threshold_angle = theta1_max

        threshold_info = {
            "threshold_angle": float(threshold_angle),
            "threshold_angle_deg": float(np.degrees(threshold_angle)),
            "confidence_level": conf,
            "p_target": p_target,
            "theta_grid": theta_grid,
            "p_chaotic_grid": p_grid,
            "logistic_model": None,
        }

    # Store logistic curve grid values for stable rendering in Manim.
    theta_grid = threshold_info["theta_grid"].astype(np.float64)
    p_grid = threshold_info["p_chaotic_grid"].astype(np.float64)

    delta_npz_path = session_dir / "delta_threshold_data.npz"
    np.savez_compressed(
        delta_npz_path,
        t=t_video,
        delta_regular=delta_reg,
        delta_chaotic=delta_cha,
        mle_regular=float(mle_all[reg_i]),
        mle_chaotic=float(mle_all[cha_i]),
        reg_run_theta1=float(valid_params[reg_i]["theta1"]),
        cha_run_theta1=float(valid_params[cha_i]["theta1"]),
        threshold_angle=float(threshold_info["threshold_angle"]),
        threshold_angle_deg=float(threshold_info["threshold_angle_deg"]),
        confidence_level=float(threshold_info["confidence_level"]),
        p_target=float(threshold_info["p_target"]),
        theta1_grid=theta_grid,
        p_chaotic_grid=p_grid,
    )

    manifest = {
        "n_ensemble": int(n_ensemble),
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
    }
    (session_dir / "session_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    return session_dir

