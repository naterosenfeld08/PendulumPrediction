from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from manim import (
    Axes,
    Dot,
    Line,
    Scene,
    Text,
    MathTex,
    FadeIn,
    Create,
    RED,
    VGroup,
    ORIGIN,
)


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

        eps = 1e-30
        y_reg = np.log10(np.clip(delta_regular, eps, None))
        y_cha = np.log10(np.clip(delta_chaotic, eps, None))

        y_min = float(min(y_reg.min(initial=0.0), y_cha.min(initial=0.0)))
        y_max = float(max(y_reg.max(initial=1.0), y_cha.max(initial=2.0)))
        t0, t1 = float(t.min()), float(t.max())

        axes_delta = Axes(
            x_range=[t0, t1, max(1.0, (t1 - t0) / 5.0)],
            y_range=[y_min - 0.2, y_max + 0.2, 1.0],
            x_length=9.2,
            y_length=2.85,
            tips=False,
            axis_config={"stroke_width": 1.0, "font_size": 22},
        )

        th_lo, th_hi = float(theta1_grid.min()), float(theta1_grid.max())
        axes_log = Axes(
            x_range=[th_lo, th_hi, (th_hi - th_lo) / 4],
            y_range=[0.0, 1.05, 0.25],
            x_length=9.2,
            y_length=2.75,
            tips=False,
            axis_config={"stroke_width": 1.0, "font_size": 22},
        )

        title_a = Text("A  Neighbor separation (log-scale)", font_size=34).scale(0.38)
        title_b = Text("B  Logistic P(chaotic | θ₁,₀)", font_size=34).scale(0.38)
        unit_a = MathTex(r"\log_{10}\delta\ \mathrm{(dimensionless)}").scale(0.38)
        unit_b = MathTex(r"P(\mathrm{chaotic}\mid\theta_{1,0})\ \mathrm{on}\ [0,1]").scale(0.32)

        top = VGroup(title_a, unit_a, axes_delta).arrange(DOWN, buff=0.1)
        bot = VGroup(title_b, unit_b, axes_log).arrange(DOWN, buff=0.1)
        panels = VGroup(top, bot).arrange(DOWN, buff=0.55)
        panels.move_to(ORIGIN)

        delta_labels = axes_delta.get_axis_labels(
            MathTex("t\\,(s)").scale(0.66),
            MathTex(r"\log_{10}\delta(t)").scale(0.66),
        )
        log_labels = axes_log.get_axis_labels(
            MathTex(r"\theta_{1,0}\,(rad)").scale(0.66),
            MathTex(r"P(\mathrm{chaotic}\mid\theta_{1,0})").scale(0.52),
        )

        self.play(FadeIn(title_a), FadeIn(unit_a), run_time=0.35)
        self.play(Create(axes_delta), FadeIn(delta_labels), run_time=0.65)

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

        self.play(Create(reg_curve), run_time=0.55)
        self.play(Create(cha_curve), run_time=0.55)

        dot = Dot(axes_delta.c2p(float(t[-1]), float(y_cha[-1])), radius=0.04, color="#ff7f0e")
        self.play(FadeIn(dot), run_time=0.25)

        tx = t0 + 0.03 * (t1 - t0)
        ty = y_min + 0.06 * (y_max - y_min)
        leg_r = Text(f"regular  λ={mle_regular:.2f}", color="#1f77b4").scale(0.32)
        leg_c = Text(f"chaotic  λ={mle_chaotic:.2f}", color="#ff7f0e").scale(0.32)
        leg_r.move_to(axes_delta.c2p(tx, ty + 0.12 * (y_max - y_min)))
        leg_c.move_to(axes_delta.c2p(tx, ty))
        self.play(FadeIn(leg_r), FadeIn(leg_c), run_time=0.45)
        self.wait(0.35)

        self.play(FadeIn(title_b), FadeIn(unit_b), run_time=0.35)
        self.play(FadeIn(axes_log), FadeIn(log_labels), run_time=0.65)

        logistic_curve = axes_log.plot_line_graph(
            x_values=theta1_grid,
            y_values=p_grid,
            line_color="#9467bd",
            add_vertex_dots=False,
        )
        self.play(Create(logistic_curve), run_time=0.65)

        x_left = th_lo
        x_right = th_hi
        hline = Line(
            axes_log.c2p(x_left, p_target),
            axes_log.c2p(x_right, p_target),
            color="#999999",
            stroke_width=2.0,
        )
        self.play(Create(hline), run_time=0.3)
        vert = Line(
            axes_log.c2p(threshold_angle, 0.0),
            axes_log.c2p(threshold_angle, 1.0),
            color=RED,
            stroke_width=3.0,
        )
        self.play(Create(vert), run_time=0.3)

        mark = Dot(axes_log.c2p(threshold_angle, p_target), radius=0.05, color=RED)
        self.play(FadeIn(mark), run_time=0.2)

        tx_note = th_lo + 0.52 * (th_hi - th_lo)
        ty_note = 0.22
        th_text = Text(f"θ₁* = {threshold_angle:.3f} rad", color="#d62728").scale(0.32)
        th_text.move_to(axes_log.c2p(tx_note, ty_note))
        self.play(FadeIn(th_text), run_time=0.4)

        self.wait(1.15)
