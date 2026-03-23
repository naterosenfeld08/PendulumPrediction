from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from manim import Axes, Dot, Line, Scene, Text, MathTex, FadeIn, Create, UP, RIGHT, DOWN, RED


def _session_dir() -> Path:
    sd = os.environ.get("MANIM_SESSION_DIR", "").strip()
    if not sd:
        raise RuntimeError("MANIM_SESSION_DIR is not set.")
    return Path(sd).expanduser().resolve()


class DeltaAndThresholdScene(Scene):
    """Animate separation δ(t), finite-time MLE, and logistic chaos threshold."""

    def construct(self) -> None:
        session_dir = _session_dir()
        data = np.load(session_dir / "delta_threshold_data.npz")

        t = data["t"].astype(float)
        delta_regular = data["delta_regular"].astype(float)
        delta_chaotic = data["delta_chaotic"].astype(float)
        mle_regular = float(data["mle_regular"])
        mle_chaotic = float(data["mle_chaotic"])

        theta1_grid = data["theta1_grid"].astype(float)
        p_grid = data["p_chaotic_grid"].astype(float)
        threshold_angle = float(data["threshold_angle"])
        p_target = float(data["p_target"])

        # ---- δ(t) plot (log10 scale) ----
        # Avoid log of 0 by clipping.
        eps = 1e-30
        y_reg = np.log10(np.clip(delta_regular, eps, None))
        y_cha = np.log10(np.clip(delta_chaotic, eps, None))

        y_min = float(min(y_reg.min(initial=0.0), y_cha.min(initial=0.0)))
        y_max = float(max(y_reg.max(initial=1.0), y_cha.max(initial=2.0)))

        axes_delta = Axes(
            x_range=[float(t.min()), float(t.max()), 5.0],
            y_range=[y_min - 0.2, y_max + 0.2, 1.0],
            x_length=9.6,
            y_length=3.2,
            tips=False,
            axis_config={"stroke_width": 1.0, "font_size": 22},
        )
        axes_delta.to_edge(UP, buff=0.85)
        delta_labels = axes_delta.get_axis_labels(
            MathTex("t\\,(s)").scale(0.7),
            MathTex(r"\log_{10}\delta(t)").scale(0.7),
        )
        self.play(Create(axes_delta), FadeIn(delta_labels), run_time=0.7)

        reg_curve = axes_delta.plot_line_graph(
            x_values=t,
            y_values=y_reg,
            line_color="#1f77b4",
            add_vertex_dots=False,
        )
        cha_curve = axes_delta.plot_line_graph(
            x_values=t,
            y_values=y_cha,
            line_color="#ff7f0e",
            add_vertex_dots=False,
        )
        reg_curve.set_stroke(width=4.0)
        cha_curve.set_stroke(width=4.0)

        self.play(Create(reg_curve), run_time=0.6)
        self.play(Create(cha_curve), run_time=0.6)

        # Add a single marker at the final chaotic separation value.
        dot = Dot(axes_delta.c2p(float(t[-1]), float(y_cha[-1])), radius=0.04, color="#ff7f0e")
        self.play(FadeIn(dot), run_time=0.25)

        label_reg = Text(f"regular  λ={mle_regular:.2f}", color="#1f77b4").scale(0.42)
        label_cha = Text(f"chaotic  λ={mle_chaotic:.2f}", color="#ff7f0e").scale(0.42)
        label_reg.next_to(axes_delta, RIGHT, buff=0.35).shift(0.25 * UP)
        label_cha.next_to(label_reg, DOWN, buff=0.18)

        self.play(FadeIn(label_reg), FadeIn(label_cha), run_time=0.5)
        self.wait(0.4)

        # ---- Logistic threshold plot ----
        # ---- Logistic threshold plot ----
        axes_log = Axes(
            x_range=[float(theta1_grid.min()), float(theta1_grid.max()), (theta1_grid.max() - theta1_grid.min()) / 4],
            y_range=[0.0, 1.05, 0.25],
            x_length=9.6,
            y_length=3.0,
            tips=False,
            axis_config={"stroke_width": 1.0, "font_size": 22},
        )
        axes_log.to_edge(DOWN, buff=0.45)
        log_labels = axes_log.get_axis_labels(
            MathTex(r"\theta_{1,0}\,(rad)").scale(0.7),
            MathTex(r"P(\mathrm{chaotic}\mid\theta_{1,0})").scale(0.54),
        )

        self.play(FadeIn(axes_log), FadeIn(log_labels), run_time=0.7)

        logistic_curve = axes_log.plot_line_graph(
            x_values=theta1_grid,
            y_values=p_grid,
            line_color="#9467bd",
            add_vertex_dots=False,
        )
        self.play(Create(logistic_curve), run_time=0.7)

        # Target probability and threshold marking.
        x_left = float(theta1_grid.min())
        x_right = float(theta1_grid.max())
        hline = Line(
            axes_log.c2p(x_left, p_target),
            axes_log.c2p(x_right, p_target),
            color="#999999",
            stroke_width=2.0,
        )
        self.play(Create(hline), run_time=0.3)
        x0 = axes_log.c2p(threshold_angle, 0.0)[0]
        y0 = axes_log.c2p(threshold_angle, 0.0)[1]
        y1 = axes_log.c2p(threshold_angle, 1.0)[1]
        vert = Line(
            axes_log.c2p(threshold_angle, 0.0),
            axes_log.c2p(threshold_angle, 1.0),
            color=RED,
            stroke_width=3.0,
        )
        self.play(Create(vert), run_time=0.3)

        mark = Dot(axes_log.c2p(threshold_angle, p_target), radius=0.05, color=RED)
        self.play(FadeIn(mark), run_time=0.2)

        th_text = Text(f"threshold θ1* = {threshold_angle:.3f} rad", color="#d62728").scale(0.42)
        th_text.next_to(axes_log, RIGHT, buff=0.28).shift(0.15 * UP)
        self.play(FadeIn(th_text), run_time=0.4)

        # Hold for readability.
        self.wait(1.2)

