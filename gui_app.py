from __future__ import annotations

import base64
import copy
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import shutil


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from output.manim_export import export_manim_session
from output.manim_render import render_scenes


def _load_config() -> dict[str, Any]:
    import yaml

    cfg_path = PROJECT_ROOT / "config.yaml"
    with cfg_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _video_base64(path: Path) -> str:
    raw = path.read_bytes()
    return base64.b64encode(raw).decode("utf-8")


def _embed_looping_video(b64: str) -> None:
    # Keep the embedded video small; the encoder settings are chosen when you pick Manim quality.
    html = f"""
    <video
      src="data:video/mp4;base64,{b64}"
      controls
      autoplay
      muted
      loop
      playsinline
      style="width:100%; border-radius:12px; box-shadow: 0 10px 30px rgba(0,0,0,0.35);"
    ></video>
    """
    st.components.v1.html(html, height=480)


def _render_live_ensemble(npz_path: Path, frame_idx: int) -> None:
    data = np.load(npz_path)
    theta1 = data["theta1"]
    theta2 = data["theta2"]
    L1 = data["L1"]
    L2 = data["L2"]
    chaotic = data["is_chaotic"].astype(bool)
    t = data["t"]

    n_ensemble, n_frames = theta1.shape
    k = int(np.clip(frame_idx, 0, n_frames - 1))

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
        th1 = float(theta1[i, k])
        th2 = float(theta2[i, k])
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
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.text(
        0.01,
        0.98,
        f"t = {float(t[k]):.2f} s",
        transform=ax.transAxes,
        va="top",
        ha="left",
        color="#cfd8e3",
        fontsize=11,
    )
    st.pyplot(fig, clear_figure=True, use_container_width=True)


def _render_live_phase_density(session_dir: Path, frame_idx: int) -> None:
    npz = np.load(session_dir / "ensemble_scene_data.npz")
    omega_edges = npz["omega_edges"]
    omega_min, omega_max = float(omega_edges[0]), float(omega_edges[-1])

    reg_paths = sorted((session_dir / "density_frames_regular").glob("frame_*.png"))
    cha_paths = sorted((session_dir / "density_frames_chaotic").glob("frame_*.png"))
    if not reg_paths or not cha_paths:
        st.warning("No density frames found for live display.")
        return
    n_frames = min(len(reg_paths), len(cha_paths))
    k = int(np.clip(frame_idx, 0, n_frames - 1))

    reg = plt.imread(reg_paths[k])
    cha = plt.imread(cha_paths[k])
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.3), dpi=120)
    fig.patch.set_facecolor("#0b1220")
    extent = [-np.pi, np.pi, omega_min, omega_max]
    for ax, img, title in zip(
        axes,
        [reg, cha],
        ["Regular subset", "Chaotic subset"],
        strict=True,
    ):
        ax.imshow(img, origin="lower", aspect="auto", extent=extent, interpolation="nearest")
        ax.set_xlabel(r"$\theta_2$ (rad)")
        ax.set_ylabel(r"$\omega_2$ (rad/s)")
        ax.set_title(title, color="#dce5f0")
        ax.tick_params(colors="#dce5f0")
        for sp in ax.spines.values():
            sp.set_color("#95a2b3")
    st.pyplot(fig, clear_figure=True, use_container_width=True)


def _render_live_delta_threshold(npz_path: Path, frame_idx: int) -> None:
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
    k = int(np.clip(frame_idx, 0, n_frames - 1))

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.0), dpi=120)
    fig.patch.set_facecolor("#0b1220")
    for ax in axes:
        ax.set_facecolor("#0b1220")
        ax.tick_params(colors="#dce5f0")
        for sp in ax.spines.values():
            sp.set_color("#95a2b3")

    ax1, ax2 = axes
    ax1.plot(t, np.log10(np.clip(dr, 1e-30, None)), color="#4ea1ff", lw=2.4, label=f"regular  λ={mle_r:.2f}")
    ax1.plot(t, np.log10(np.clip(dc, 1e-30, None)), color="#ff8f3f", lw=2.4, label=f"chaotic  λ={mle_c:.2f}")
    yk = np.log10(max(float(dc[k]), 1e-30))
    ax1.scatter([t[k]], [yk], s=26, color="#ff8f3f", zorder=3)
    ax1.set_xlabel("t (s)")
    ax1.set_ylabel(r"$\log_{10}\delta(t)$")
    ax1.legend(frameon=False, labelcolor="#dce5f0")

    ax2.plot(th, p, color="#bf8cff", lw=2.4, label=r"$P(\mathrm{chaotic}\mid\theta_{1,0})$")
    ax2.axhline(p_target, color="#adb5bd", ls="--", lw=1.4, label=f"target={p_target:.2f}")
    ax2.axvline(th_star, color="#ff4d4d", ls="--", lw=1.4, label=fr"$\theta_1^*={th_star:.3f}$ rad")
    ax2.scatter([th_star], [p_target], s=30, color="#ff4d4d")
    ax2.set_xlabel(r"$\theta_{1,0}$ (rad)")
    ax2.set_ylabel("probability")
    ax2.legend(frameon=False, labelcolor="#dce5f0")

    st.pyplot(fig, clear_figure=True, use_container_width=True)


def _scene_descriptions() -> dict[str, str]:
    return {
        "Ensemble": (
            "Each mini-system is one double pendulum from the ensemble. "
            "Blue trajectories are classified regular and orange trajectories chaotic "
            "from the finite-time Lyapunov criterion."
        ),
        "Phase density": (
            "This is accumulated phase-space density in (theta2, omega2) over time. "
            "Left: regular subset, Right: chaotic subset. Brighter regions indicate "
            "more time spent by the ensemble in that region."
        ),
        "δ(t) + threshold": (
            "Top: log10 separation delta(t) for a representative regular and chaotic pair. "
            "Bottom: logistic chaos probability versus theta1 with the configured threshold marker."
        ),
    }


def _inject_style() -> None:
    # Dark, modern style inspired by UniPhi Collective (cards, gradient background).
    st.markdown(
        """
        <style>
        :root {
          --bg0: #070A12;
          --bg1: #0B1630;
          --card: rgba(255,255,255,0.06);
          --card2: rgba(255,255,255,0.09);
          --text: rgba(255,255,255,0.92);
          --muted: rgba(255,255,255,0.68);
          --border: rgba(255,255,255,0.12);
          --accent: #7C5CFF;
          --accent2: #2EE9A6;
        }
        body {
          background: radial-gradient(1200px 700px at 20% -10%, rgba(124,92,255,0.25), transparent 60%),
                      radial-gradient(900px 600px at 90% 0%, rgba(46,233,166,0.18), transparent 55%),
                      linear-gradient(180deg, var(--bg0), var(--bg1));
          color: var(--text);
        }
        .stApp {
          padding-top: 10px;
        }
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
        .subtle {
          color: var(--muted);
          font-size: 0.9rem;
        }
        .accent-text {
          color: rgba(124,92,255,0.95);
          font-weight: 700;
        }
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


def main() -> None:
    st.set_page_config(page_title="Double Pendulum Manim Explorer", layout="wide")
    _inject_style()

    st.markdown(
        """
        <div class="card">
          <div style="font-size: 1.2rem; line-height: 1.2; margin-bottom: 6px;">
            <span class="accent-text">Double Pendulum</span> — Manim visualization explorer
          </div>
          <div class="subtle">
            Render from precomputed simulation assets into three continuously looping scenes:
            ensemble motion, accumulated phase-space density in (θ₂, ω₂), and δ(t) + logistic threshold.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cfg0 = _load_config()
    if "assets" not in st.session_state:
        st.session_state["assets"] = {}
    if "frame_idx" not in st.session_state:
        st.session_state["frame_idx"] = 0

    with st.sidebar:
        st.header("Parameters")
        n_ensemble = st.slider("Ensemble size", min_value=8, max_value=64, value=32, step=1)
        rng_seed = st.number_input("Random seed", min_value=0, value=int(cfg0["ensemble"]["seed"]), step=1)

        st.divider()
        st.subheader("Integration + export")
        sim_t_span_start = float(cfg0["integration"]["t_span"][0])
        sim_t_span_end = st.slider(
            "Simulation end time (s)",
            min_value=5.0,
            max_value=40.0,
            value=float(cfg0["integration"]["t_span"][1]),
            step=1.0,
        )
        # Keep n_steps the default unless the user explicitly changes it; energy conservation checks are strict.
        n_steps = int(cfg0["integration"]["n_steps"])
        st.caption(f"Using n_steps={n_steps} (energy drift is hard-checked).")

        video_frames = st.slider("Video frames per scene", min_value=60, max_value=240, value=120, step=10)
        density_frames = st.slider("Density frames", min_value=10, max_value=40, value=20, step=5)

        st.divider()
        st.subheader("Manim quality")
        quality = st.selectbox("Render quality", options=["low", "medium", "high", "2k"], index=2)
        render_manim_mp4 = st.checkbox("Also render MP4 scenes (optional)", value=True)

        st.divider()
        st.subheader("Live playback")
        autoplay = st.checkbox("Autoplay live scenes", value=False)
        fps = st.slider("Live FPS", min_value=2, max_value=24, value=8, step=1)
        if st.button("Reset frame"):
            st.session_state["frame_idx"] = 0

        st.divider()
        st.subheader("Parameter ranges (for LHS)")

        def range_inputs(key: str, default: list[float]) -> list[float]:
            lo0, hi0 = float(default[0]), float(default[1])
            lo = st.number_input(f"{key} min", value=lo0, step=(hi0 - lo0) / 100.0 if hi0 != lo0 else 0.01)
            hi = st.number_input(f"{key} max", value=hi0, step=(hi0 - lo0) / 100.0 if hi0 != lo0 else 0.01)
            if hi < lo:
                st.error(f"{key} max must be >= min.")
            return [float(lo), float(hi)]

        params_ui: dict[str, list[float]] = {}
        for k in ["m1", "m2", "L1", "L2", "theta1", "theta2", "omega1", "omega2"]:
            params_ui[k] = range_inputs(k, cfg0["parameters"][k])

        render_btn = st.button("Render scenes", type="primary")

    if render_btn:
        st.session_state["assets"] = {}
        session_id = time.strftime("%Y%m%d_%H%M%S") + f"_{rng_seed}"
        session_dir = PROJECT_ROOT / "data" / "results" / "manim_sessions" / session_id
        st.info(f"Exporting assets to: `{session_dir}`")

        cfg = copy.deepcopy(cfg0)
        cfg["ensemble"]["n_pendulums"] = int(n_ensemble)
        cfg["integration"]["t_span"] = [float(sim_t_span_start), float(sim_t_span_end)]
        for k, v in params_ui.items():
            cfg["parameters"][k] = v

        progress = st.progress(0)
        progress.progress(5)
        with st.spinner("Exporting simulation assets (trajectories + density frames) ..."):
            export_manim_session(
                cfg,
                session_dir,
                int(rng_seed),
                n_ensemble=int(n_ensemble),
                video_frames=int(video_frames),
                density_frames=int(density_frames),
            )
        progress.progress(40)
        st.session_state["assets"]["session_dir"] = str(session_dir)
        st.session_state["frame_idx"] = 0

        if render_manim_mp4:
            if shutil.which("manim") is None:
                st.warning("Manim not found on PATH, skipping MP4 rendering.")
            else:
                scene_specs = [
                    (PROJECT_ROOT / "manim_scenes" / "ensemble_scene.py", "EnsembleChaosScene"),
                    (PROJECT_ROOT / "manim_scenes" / "phase_density_scene.py", "PhaseDensityAccumulationScene"),
                    (PROJECT_ROOT / "manim_scenes" / "delta_threshold_scene.py", "DeltaAndThresholdScene"),
                ]

                with st.spinner("Rendering Manim scenes to MP4 ... (this can take a while)"):
                    rendered = render_scenes(
                        session_dir,
                        scene_specs,
                        quality=str(quality),
                        media_subdir="manim_media",
                    )
                st.session_state["assets"]["videos"] = {
                    k: _video_base64(v) for k, v in rendered.items()
                }
        progress.progress(95)
        progress.progress(100)

    st.markdown("## Scenes")
    tabs = st.tabs(["Ensemble", "Phase density", "δ(t) + threshold"])

    videos = st.session_state["assets"].get("videos", {})
    desc = _scene_descriptions()
    session_dir_str = st.session_state["assets"].get("session_dir")
    has_live = bool(session_dir_str)
    frame_idx = int(st.session_state.get("frame_idx", 0))

    with tabs[0]:
        st.markdown("### Ensemble motion")
        st.caption(desc["Ensemble"])
        if has_live:
            session_dir = Path(session_dir_str)
            _render_live_ensemble(session_dir / "ensemble_scene_data.npz", frame_idx)
        elif videos:
            _embed_looping_video(videos.get("EnsembleChaosScene", next(iter(videos.values()))))
        else:
            st.info("Click **Render scenes** to export data and visualize this tab.")

    with tabs[1]:
        st.markdown("### Accumulated phase-space density")
        st.caption(desc["Phase density"])
        if has_live:
            session_dir = Path(session_dir_str)
            _render_live_phase_density(session_dir, frame_idx)
        elif videos:
            _embed_looping_video(videos.get("PhaseDensityAccumulationScene", next(iter(videos.values()))))
        else:
            st.info("Click **Render scenes** to export data and visualize this tab.")

    with tabs[2]:
        st.markdown("### Separation and threshold")
        st.caption(desc["δ(t) + threshold"])
        if has_live:
            session_dir = Path(session_dir_str)
            _render_live_delta_threshold(session_dir / "delta_threshold_data.npz", frame_idx)
        elif videos:
            _embed_looping_video(videos.get("DeltaAndThresholdScene", next(iter(videos.values()))))
        else:
            st.info("Click **Render scenes** to export data and visualize this tab.")

    if has_live and autoplay:
        n_frames = 1
        try:
            data = np.load(Path(session_dir_str) / "ensemble_scene_data.npz")
            n_frames = int(data["theta1"].shape[1])
        except Exception:
            n_frames = 1
        st.session_state["frame_idx"] = (frame_idx + 1) % max(1, n_frames)
        time.sleep(1.0 / max(1, int(fps)))
        st.rerun()


if __name__ == "__main__":
    main()

