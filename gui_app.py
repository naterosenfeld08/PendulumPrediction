from __future__ import annotations

import copy
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _load_config() -> dict[str, Any]:
    import yaml

    with (PROJECT_ROOT / "config.yaml").open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _inject_style() -> None:
    st.markdown(
        """
        <style>
        :root {
          --bg0: #070A12;
          --bg1: #0B1630;
          --card: rgba(255,255,255,0.06);
          --text: rgba(255,255,255,0.92);
          --muted: rgba(255,255,255,0.68);
          --border: rgba(255,255,255,0.12);
          --accent: #7C5CFF;
        }
        body {
          background: radial-gradient(1200px 700px at 20% -10%, rgba(124,92,255,0.25), transparent 60%),
                      linear-gradient(180deg, var(--bg0), var(--bg1));
          color: var(--text);
        }
        .stApp { padding-top: 10px; }
        div[data-testid="stSidebar"] {
          background: rgba(0,0,0,0.25);
          border-right: 1px solid var(--border);
        }
        .card {
          background: var(--card);
          border: 1px solid var(--border);
          border-radius: 16px;
          padding: 14px 16px;
          box-shadow: 0 10px 40px rgba(0,0,0,0.25);
        }
        .subtle { color: var(--muted); font-size: 0.9rem; }
        .accent-text { color: rgba(124,92,255,0.95); font-weight: 700; }
        div.stButton > button {
          background: linear-gradient(135deg, rgba(124,92,255,0.95), rgba(46,233,166,0.6));
          border: 0px;
          color: white;
          font-weight: 700;
          border-radius: 12px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# θ₁, θ₂, ω₁, ω₂ ranges; m₁, m₂, L₁, L₂ always taken from config.yaml unless Custom edits all.
PRESET_ANGLE_VELOCITY: dict[str, dict[str, list[float]]] = {
    "Mixed": {},  # identical to config.yaml
    "Mild": {
        "theta1": [0.05, 0.7],
        "theta2": [0.05, 0.7],
        "omega1": [0.0, 0.3],
        "omega2": [0.0, 0.3],
    },
    "Wild": {
        "theta1": [0.12, 3.14],
        "theta2": [0.12, 3.14],
        "omega1": [0.0, 1.0],
        "omega2": [0.0, 1.0],
    },
}


def _parameters_for_run(
    cfg0: dict[str, Any],
    preset: str,
    params_ui: dict[str, list[float]] | None,
) -> dict[str, list[float]]:
    """Full parameter box for LHS: preset merges angle/ω; Custom uses ``params_ui``."""
    keys = ["m1", "m2", "L1", "L2", "theta1", "theta2", "omega1", "omega2"]
    if preset == "Custom":
        assert params_ui is not None
        return {k: list(params_ui[k]) for k in keys}
    base = {k: list(cfg0["parameters"][k]) for k in keys}
    for k, v in PRESET_ANGLE_VELOCITY.get(preset, {}).items():
        base[k] = [float(v[0]), float(v[1])]
    return base


def _run_statistical_pipeline(config: dict[str, Any]) -> tuple[Any, Any, Any, Any, Any]:
    """Run ensemble → GPR → logistic threshold → inverse angle; return ``df`` and model objects."""
    import numpy as np
    from ensemble.ensemble import run_ensemble
    from stats.inverse import angle_from_variance_target
    from stats.stats import fit_model
    from stats.threshold import find_chaos_threshold

    rng = np.random.default_rng(int(config["ensemble"]["seed"]))
    df = run_ensemble(config, rng=rng)
    model = fit_model(df, config)
    threshold_info = find_chaos_threshold(df, config)
    target_var = float(config["inverse"]["target_variance"])
    try:
        inv_angle = angle_from_variance_target(model, target_var, config)
    except Exception:
        inv_angle = float("nan")
    return df, model, threshold_info, inv_angle, target_var


def _fig_logistic(threshold_info: dict[str, Any], config: dict[str, Any]) -> plt.Figure:
    from stats.threshold import chaos_probability_vs_theta

    th = np.linspace(
        float(config["parameters"]["theta1"][0]),
        float(config["parameters"]["theta1"][1]),
        400,
    )
    clf = threshold_info["logistic_model"]
    if clf is None:
        p = np.full_like(th, float(threshold_info["p_chaotic_grid"][0]), dtype=float)
    else:
        p = chaos_probability_vs_theta(clf, th)
    p_target = float(threshold_info["p_target"])
    th_star = float(threshold_info["threshold_angle"])

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(th, p, color="C1", lw=2, label=r"$P(\mathrm{chaotic}\mid \theta_1)$")
    ax.axhline(
        p_target,
        color="0.5",
        ls="--",
        lw=1,
        label=rf"$1-\alpha = {p_target:.3f}$",
    )
    ax.axvline(
        th_star,
        color="C3",
        ls="--",
        lw=1,
        label=rf"$\theta_1^* = {th_star:.3f}\ \mathrm{{rad}}$",
    )
    ax.set_xlabel(r"$\theta_1\ \mathrm{(rad)}$")
    ax.set_ylabel("probability")
    ax.legend()
    ax.grid(True, ls=":", alpha=0.5)
    fig.tight_layout()
    return fig


def _fig_phase_three(
    df: pd.DataFrame,
    config: dict[str, Any],
    *,
    plane: str,
) -> plt.Figure:
    """``plane`` is ``arm1`` or ``arm2``: plot $(\\theta_i, \\omega_i)$.

    Middle panel: member whose initial $\\theta_1$ is **nearest the sample median** $\\theta_1$.
    Side panels: min / max $\\lambda$.
    """
    from physics.integrator import integrate
    from physics.pendulum import dstate_dt

    df = df.reset_index(drop=True)
    mle = df["mle"].to_numpy()
    th1 = df["theta1"].to_numpy()
    med_th1 = float(np.median(th1))
    i_low = int(np.argmin(mle))
    i_high = int(np.argmax(mle))
    i_mid = int(np.argmin(np.abs(th1 - med_th1)))

    if plane == "arm2":
        i_x, i_y = 2, 3
        xlab, ylab = r"$\theta_2\ \mathrm{(rad)}$", r"$\omega_2\ \mathrm{(rad/s)}$"
        plane_name = r"$(\theta_2,\,\omega_2)$"
    else:
        i_x, i_y = 0, 1
        xlab, ylab = r"$\theta_1\ \mathrm{(rad)}$", r"$\omega_1\ \mathrm{(rad/s)}$"
        plane_name = r"$(\theta_1,\,\omega_1)$"

    triple = [
        (i_low, "Lowest λ (most regular-like)"),
        (i_mid, f"Median θ₁ (nearest med θ₁={med_th1:.3f})"),
        (i_high, "Highest λ (most chaotic-like)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), sharex=False, sharey=False)
    for ax, (ix, title) in zip(axes, triple):
        row = df.iloc[ix]
        params = {
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
        state0 = np.array(
            [params["theta1"], params["omega1"], params["theta2"], params["omega2"]],
            dtype=np.float64,
        )
        _, y = integrate(dstate_dt, state0, params, config)
        ax.plot(y[:, i_x], y[:, i_y], lw=0.8, color="C0")
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.set_title(title + f"\nλ={float(row['mle']):.3f}, θ₁₀={float(row['theta1']):.3f}")
        ax.grid(True, ls=":", alpha=0.5)
    fig.suptitle(f"Phase space in {plane_name}", fontsize=11, y=1.02)
    fig.tight_layout()
    return fig


def _render_live_ensemble(npz_path: Path, frame_pos: float) -> None:
    data = np.load(npz_path)
    theta1 = data["theta1"]
    theta2 = data["theta2"]
    L1 = data["L1"]
    L2 = data["L2"]
    chaotic = data["is_chaotic"].astype(bool)
    t = data["t"]

    n_ensemble, n_frames = theta1.shape
    if n_frames < 2:
        k0 = k1 = 0
        alpha = 0.0
    else:
        x = float(frame_pos) % float(n_frames - 1)
        k0 = int(np.floor(x))
        k1 = min(k0 + 1, n_frames - 1)
        alpha = x - k0

    cols = int(np.ceil(np.sqrt(n_ensemble)))
    rows = int(np.ceil(n_ensemble / cols))
    frame_w, frame_h = 13.0, 7.4
    pad_x = frame_w / max(cols, 1)
    pad_y = frame_h / max(rows, 1)

    ltot_max = float(np.max(L1 + L2))
    if ltot_max <= 0:
        ltot_max = 1.0
    scale = min(0.60 * pad_y, 0.42 * pad_x) / ltot_max

    fig, ax = plt.subplots(figsize=(12.0, 6.8), dpi=120)
    fig.patch.set_facecolor("#0b1220")
    ax.set_facecolor("#0b1220")

    for i in range(n_ensemble):
        r = i // cols
        c = i % cols
        x0 = (c - (cols - 1) / 2.0) * pad_x
        y0 = ((rows - 1) / 2.0 - r) * pad_y
        th1 = float((1.0 - alpha) * theta1[i, k0] + alpha * theta1[i, k1])
        th2 = float((1.0 - alpha) * theta2[i, k0] + alpha * theta2[i, k1])
        l1 = float(L1[i]) * scale
        l2 = float(L2[i]) * scale
        x1 = x0 + l1 * np.sin(th1)
        y1 = y0 - l1 * np.cos(th1)
        x2 = x1 + l2 * np.sin(th2)
        y2 = y1 - l2 * np.cos(th2)
        color = "#ff8f3f" if chaotic[i] else "#4ea1ff"
        ax.plot([x0, x1], [y0, y1], color=color, lw=1.8, alpha=0.9)
        ax.plot([x1, x2], [y1, y2], color=color, lw=1.8, alpha=0.9)
        ax.scatter([x1, x2], [y1, y2], color=color, s=7)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-frame_w / 2 - 0.4, frame_w / 2 + 0.4)
    ax.set_ylim(-frame_h / 2 - 0.4, frame_h / 2 + 0.4)
    ax.set_xlabel("x (m)", color="#dce5f0")
    ax.set_ylabel("y (m)", color="#dce5f0")
    ax.tick_params(colors="#dce5f0")
    for sp in ax.spines.values():
        sp.set_color("#95a2b3")
    n_reg = int((~chaotic).sum())
    n_cha = int(chaotic.sum())
    ax.text(
        0.01,
        0.98,
        f"t = {float((1.0 - alpha) * t[k0] + alpha * t[k1]):.2f} s\nregular={n_reg}, chaotic={n_cha}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        color="#cfd8e3",
        fontsize=11,
    )
    st.pyplot(fig, clear_figure=True, use_container_width=True)


def _render_live_phase_density(session_dir: Path, frame_idx: int) -> int:
    npz = np.load(session_dir / "ensemble_scene_data.npz")
    omega_edges = npz["omega_edges"]
    omega_min, omega_max = float(omega_edges[0]), float(omega_edges[-1])

    reg_paths = sorted((session_dir / "density_frames_regular").glob("frame_*.png"))
    cha_paths = sorted((session_dir / "density_frames_chaotic").glob("frame_*.png"))
    if not reg_paths or not cha_paths:
        st.warning("No density frames found for live display.")
        return 1
    n_frames = min(len(reg_paths), len(cha_paths))
    k = int(frame_idx % n_frames)

    reg = plt.imread(reg_paths[k])
    cha = plt.imread(cha_paths[k])
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.3), dpi=120)
    fig.patch.set_facecolor("#0b1220")
    extent = [-np.pi, np.pi, omega_min, omega_max]
    for ax, img, title in zip(
        axes,
        [reg, cha],
        ["Regular subset", "Chaotic subset"],
    ):
        ax.imshow(img, origin="lower", aspect="auto", extent=extent, interpolation="nearest")
        ax.set_xlabel(r"$\theta_2\ \mathrm{(rad)}$")
        ax.set_ylabel(r"$\omega_2\ \mathrm{(rad/s)}$")
        ax.set_title(title, color="#dce5f0")
        ax.tick_params(colors="#dce5f0")
        for sp in ax.spines.values():
            sp.set_color("#95a2b3")
    st.pyplot(fig, clear_figure=True, use_container_width=True)
    return n_frames


def _render_live_delta_threshold(npz_path: Path, frame_idx: int) -> int:
    d = np.load(npz_path)
    t = d["t"].astype(float)
    dr = d["delta_regular"].astype(float)
    dc = d["delta_chaotic"].astype(float)
    mle_r = float(d["mle_regular"])
    mle_c = float(d["mle_chaotic"])
    th = d["theta1_grid"].astype(float)
    p = d["p_chaotic_grid"].astype(float)
    p_target = float(d["p_target"])
    th_star = float(d["threshold_angle"])

    n_frames = len(t)
    k = int(frame_idx % max(1, n_frames))

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.0), dpi=120)
    fig.patch.set_facecolor("#0b1220")
    for ax in axes:
        ax.set_facecolor("#0b1220")
        ax.tick_params(colors="#dce5f0")
        for sp in ax.spines.values():
            sp.set_color("#95a2b3")

    ax1, ax2 = axes
    ax1.plot(
        t[: k + 1],
        np.log10(np.clip(dr[: k + 1], 1e-30, None)),
        color="#4ea1ff",
        lw=2.4,
        label=f"regular  λ={mle_r:.2f}",
    )
    ax1.plot(
        t[: k + 1],
        np.log10(np.clip(dc[: k + 1], 1e-30, None)),
        color="#ff8f3f",
        lw=2.4,
        label=f"chaotic  λ={mle_c:.2f}",
    )
    yk = np.log10(max(float(dc[k]), 1e-30))
    ax1.scatter([t[k]], [yk], s=26, color="#ff8f3f", zorder=3)
    ax1.set_xlabel("t (s)")
    ax1.set_ylabel(r"$\log_{10}\delta(t)\ \mathrm{(dimensionless)}$")
    ax1.legend(frameon=False, labelcolor="#dce5f0")

    ax2.plot(th, p, color="#bf8cff", lw=2.4, label=r"$P(\mathrm{chaotic}\mid\theta_1)$")
    ax2.axhline(p_target, color="#adb5bd", ls="--", lw=1.4, label=f"target={p_target:.2f}")
    ax2.axvline(
        th_star,
        color="#ff4d4d",
        ls="--",
        lw=1.4,
        label=rf"$\theta_1^* = {th_star:.3f}\ \mathrm{{rad}}$",
    )
    ax2.scatter([th_star], [p_target], s=30, color="#ff4d4d")
    ax2.set_xlabel(r"$\theta_1\ \mathrm{(rad)}$")
    ax2.set_ylabel("P (dimensionless)")
    ax2.legend(frameon=False, labelcolor="#dce5f0")

    st.pyplot(fig, clear_figure=True, use_container_width=True)
    return n_frames


def main() -> None:
    st.set_page_config(page_title="Double pendulum — interactive analysis", layout="wide")
    _inject_style()

    st.markdown(
        """
        <div class="card">
          <div style="font-size: 1.15rem; margin-bottom: 6px;">
            <span class="accent-text">Interactive ensemble</span> — choose ranges, run statistics, explore chaos
          </div>
          <div class="subtle">
            Latin Hypercube samples fill your parameter boxes. Each run gets a finite-time Lyapunov-style λ and a
            chaos label (λ &gt; ε). Logistic regression maps θ₁ to P(chaotic) at your confidence level.
            Visuals use the <strong>same</strong> ensemble rows as the statistics.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cfg0 = _load_config()

    if "analysis" not in st.session_state:
        st.session_state["analysis"] = None
    if "session_dir" not in st.session_state:
        st.session_state["session_dir"] = None
    if "frame_idx" not in st.session_state:
        st.session_state["frame_idx"] = 0
    if "frame_pos" not in st.session_state:
        st.session_state["frame_pos"] = 0.0

    with st.sidebar:
        st.header("1. Ensemble & sampling")
        n_pend = st.slider("Number of pendulums", min_value=8, max_value=300, value=48, step=1)
        rng_seed = st.number_input("Random seed (LHS)", min_value=0, value=int(cfg0["ensemble"]["seed"]), step=1)

        st.subheader("2. Integration")
        t_end = st.slider("Simulation end time (s)", 5.0, 40.0, float(cfg0["integration"]["t_span"][1]), 1.0)
        n_steps = int(cfg0["integration"]["n_steps"])
        st.caption(f"Fixed n_steps={n_steps} (energy drift check).")

        st.subheader("3. Confidence & Lyapunov")
        conf_pct = st.selectbox("Confidence level (1−α)", options=[90, 95, 99], index=1)
        epsilon = st.number_input(
            "Chaos threshold ε (λ > ε)",
            value=float(cfg0["lyapunov"]["epsilon"]),
            format="%.4f",
            help="0 = any positive λ counts as chaotic (strict).",
        )
        delta0 = st.number_input(
            "Initial separation δ₀",
            min_value=1e-14,
            value=max(float(cfg0["lyapunov"]["delta0"]), 1e-14),
            format="%.2e",
            help="Must be positive.",
        )
        t_ly_default = min(float(cfg0["lyapunov"]["t_lyapunov"]), float(t_end))
        t_ly = st.slider("Lyapunov horizon T (s)", 1.0, float(t_end), float(t_ly_default), 1.0)

        st.subheader("4. Parameter ranges (LHS)")
        preset = st.selectbox(
            "Angle & velocity preset",
            ["Mixed", "Mild", "Wild", "Custom"],
            index=0,
            help=(
                "Mixed = ranges from config.yaml. Mild / Wild shrink or widen θ₁, θ₂ and ω boxes. "
                "Custom: edit all eight parameter ranges below."
            ),
        )
        if preset == "Mild":
            st.caption("Mild: smaller angles and angular velocities (often more regular motion).")
        elif preset == "Wild":
            st.caption("Wild: larger angle and velocity boxes (more exploration).")
        elif preset == "Mixed":
            st.caption("Mixed: same θ/ω boxes as `config.yaml` (masses/lengths unchanged).")

        params_ui: dict[str, list[float]] | None = None
        if preset == "Custom":
            with st.expander("Custom ranges (masses, lengths, angles, velocities)", expanded=True):
                params_ui = {}
                for k in ["m1", "m2", "L1", "L2", "theta1", "theta2", "omega1", "omega2"]:
                    lo0, hi0 = float(cfg0["parameters"][k][0]), float(cfg0["parameters"][k][1])
                    c1, c2 = st.columns(2)
                    with c1:
                        lo = st.number_input(f"{k} min", key=f"pmin_{k}", value=lo0, format="%.4f")
                    with c2:
                        hi = st.number_input(f"{k} max", key=f"pmax_{k}", value=hi0, format="%.4f")
                    params_ui[k] = [float(lo), float(hi)]

        st.subheader("5. Statistics speed")
        boot = st.slider("GPR bootstrap iterations", min_value=50, max_value=600, value=200, step=50)
        fig_boot = st.slider("Figure bootstrap (CIs)", min_value=30, max_value=150, value=60, step=10)

        st.subheader("6. Visualization export")
        max_vis = st.slider("Max pendulums in motion view", 8, 64, min(48, n_pend), 1)
        video_frames = st.slider("Animation frames", 40, 120, 80, 10)
        density_frames = st.slider("Density frames", 8, 24, 12, 2)

        st.subheader("7. Playback")
        autoplay = st.checkbox("Autoplay live scenes", value=True)
        fps = st.slider("Live FPS", 2, 24, 12, 1)
        if st.button("Reset animation clock"):
            st.session_state["frame_idx"] = 0
            st.session_state["frame_pos"] = 0.0

        run_btn = st.button("Run ensemble & statistics", type="primary")

    if run_btn:
        st.session_state["analysis"] = None
        st.session_state["session_dir"] = None
        cfg = copy.deepcopy(cfg0)
        cfg["ensemble"]["n_pendulums"] = int(n_pend)
        cfg["ensemble"]["seed"] = int(rng_seed)
        cfg["statistics"]["confidence_level"] = float(conf_pct) / 100.0
        cfg["lyapunov"]["epsilon"] = float(epsilon)
        cfg["lyapunov"]["delta0"] = float(delta0)
        cfg["lyapunov"]["t_lyapunov"] = float(t_ly)
        cfg["integration"]["t_span"] = [float(cfg0["integration"]["t_span"][0]), float(t_end)]
        cfg["statistics"]["bootstrap_iterations"] = int(boot)
        cfg["statistics"]["figure_bootstrap_iterations"] = int(fig_boot)
        cfg["parameters"] = _parameters_for_run(cfg0, preset, params_ui)

        with st.status("Running ensemble and statistics…", expanded=True) as status:
            try:
                df, model, threshold_info, inv_angle, target_var = _run_statistical_pipeline(cfg)
            except Exception as exc:
                status.update(label="Failed", state="error")
                st.error(str(exc))
                st.stop()
            status.update(label="Building visualization session…", state="running")

            session_id = time.strftime("%Y%m%d_%H%M%S") + f"_{rng_seed}"
            session_dir = PROJECT_ROOT / "data" / "results" / "gui_sessions" / session_id
            from output.gui_session import export_gui_session_from_dataframe

            try:
                export_gui_session_from_dataframe(
                    df,
                    cfg,
                    threshold_info,
                    session_dir,
                    rng_seed=int(rng_seed),
                    max_visual_runs=min(int(max_vis), int(n_pend)),
                    video_frames=int(video_frames),
                    density_frames=int(density_frames),
                )
            except Exception as exc:
                status.update(label="Visualization export failed", state="error")
                st.error(str(exc))
                st.stop()

            status.update(label="Done.", state="complete")

        st.session_state["analysis"] = {
            "df": df,
            "model": model,
            "threshold_info": threshold_info,
            "inv_angle": inv_angle,
            "target_var": target_var,
            "config": cfg,
            "preset": preset,
        }
        st.session_state["session_dir"] = str(session_dir)
        st.session_state["frame_idx"] = 0
        st.session_state["frame_pos"] = 0.0
        st.rerun()

    analysis = st.session_state.get("analysis")
    session_dir_str = st.session_state.get("session_dir")

    if not analysis or not session_dir_str:
        st.info(
            "Configure the sidebar and click **Run ensemble & statistics** to sample initial conditions, "
            "integrate, label chaos, fit logistic P(chaotic|θ₁), and build the scenes below."
        )
        return

    df: pd.DataFrame = analysis["df"]
    threshold_info: dict[str, Any] = analysis["threshold_info"]
    inv_angle = analysis["inv_angle"]
    target_var = analysis["target_var"]
    cfg = analysis["config"]

    tabs = st.tabs(
        [
            "Summary & statistics",
            "Ensemble motion",
            "Phase portraits (3)",
            "Phase-space density",
            "δ(t) + threshold",
        ]
    )

    with tabs[0]:
        st.subheader("Chaos labels")
        preset_used = analysis.get("preset")
        if preset_used:
            st.caption(f"Angle/velocity preset for this run: **{preset_used}**")
        n_cha = int(df["is_chaotic"].sum())
        n_tot = len(df)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Fraction chaotic", f"{n_cha / max(1, n_tot):.1%}")
        c2.metric("θ₁* (logistic)", f"{float(threshold_info['threshold_angle']):.3f} rad")
        c3.metric("Target P", f"{float(threshold_info['p_target']):.3f}")
        c4.metric("Inverse θ₁ (var cap)", f"{float(inv_angle):.3f} rad" if np.isfinite(inv_angle) else "n/a")

        st.caption(
            "If every run is chaotic (or all regular), the logistic curve is degenerate; see README limitations."
        )
        st.dataframe(
            df[
                [
                    "theta1",
                    "theta2",
                    "mle",
                    "energy_ratio_variance",
                    "is_chaotic",
                ]
            ].head(20),
            use_container_width=True,
        )

        fig_log = _fig_logistic(threshold_info, cfg)
        st.pyplot(fig_log, clear_figure=True, use_container_width=True)
        plt.close(fig_log)

        if np.isfinite(inv_angle):
            st.markdown(
                f"**Inverse problem (illustrative):** max θ₁ such that predicted var(KE₂/KE) at upper CI "
                f"stays below **{target_var:.4f}** → **{float(inv_angle):.4f}** rad."
            )
        else:
            st.markdown("*Inverse angle not available (GPR / bootstrap failed for this run).*")

    with tabs[1]:
        st.caption("Blue: regular (λ ≤ ε). Orange: chaotic (λ > ε). Same LHS rows as the table.")
        session_dir = Path(session_dir_str)
        _render_live_ensemble(session_dir / "ensemble_scene_data.npz", float(st.session_state["frame_pos"]))

    with tabs[2]:
        st.caption(
            "Three members: **min λ**, **median θ₁** (run whose θ₁,₀ is nearest the sample median θ₁), **max λ**. "
            "Toggle the phase plane for arm 1 or arm 2."
        )
        plane_choice = st.radio(
            "Phase plane",
            ["Arm 1 (θ₁, ω₁)", "Arm 2 (θ₂, ω₂)"],
            horizontal=True,
            key="phase_plane_toggle",
        )
        plane = "arm2" if plane_choice.startswith("Arm 2") else "arm1"
        fig_ph = _fig_phase_three(df, cfg, plane=plane)
        st.pyplot(fig_ph, clear_figure=True, use_container_width=True)
        plt.close(fig_ph)

    with tabs[3]:
        st.caption("Accumulated density in $(\\theta_2, \\omega_2)$ over time (regular vs chaotic subsets).")
        _render_live_phase_density(Path(session_dir_str), int(st.session_state["frame_idx"]))

    with tabs[4]:
        st.caption("Representative neighbor separation δ(t) and logistic threshold from this run.")
        _render_live_delta_threshold(Path(session_dir_str) / "delta_threshold_data.npz", int(st.session_state["frame_idx"]))

    if autoplay and session_dir_str:
        st.session_state["frame_idx"] = int(st.session_state["frame_idx"]) + 1
        st.session_state["frame_pos"] = float(st.session_state["frame_pos"]) + 0.75
        time.sleep(1.0 / max(1, int(fps)))
        st.rerun()


if __name__ == "__main__":
    main()
