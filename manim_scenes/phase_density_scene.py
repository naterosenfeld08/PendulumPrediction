from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from manim import (
    Axes,
    ImageMobject,
    LEFT,
    RIGHT,
    DOWN,
    UP,
    Scene,
    Text,
    MathTex,
    FadeIn,
    Transform,
)
from manim import PI


def _session_dir() -> Path:
    sd = os.environ.get("MANIM_SESSION_DIR", "").strip()
    if not sd:
        raise RuntimeError("MANIM_SESSION_DIR is not set.")
    return Path(sd).expanduser().resolve()


class PhaseDensityAccumulationScene(Scene):
    """Animate accumulated phase-space density in (theta2, omega2)."""

    def construct(self) -> None:
        session_dir = _session_dir()
        data = np.load(session_dir / "ensemble_scene_data.npz")

        omega_edges = data["omega_edges"].astype(float)
        omega_min, omega_max = float(omega_edges[0]), float(omega_edges[-1])

        # Load precomputed density heatmap frames.
        reg_dir = session_dir / "density_frames_regular"
        cha_dir = session_dir / "density_frames_chaotic"
        reg_paths = sorted(reg_dir.glob("frame_*.png"))
        cha_paths = sorted(cha_dir.glob("frame_*.png"))
        if not reg_paths or not cha_paths:
            raise RuntimeError("Missing density frames. Run manim_export first.")

        n_frames = min(len(reg_paths), len(cha_paths))
        reg_paths = reg_paths[:n_frames]
        cha_paths = cha_paths[:n_frames]

        # Two-panel layout
        axes_kwargs = dict(
            x_range=[-PI, PI, PI / 2],
            y_range=[omega_min, omega_max, (omega_max - omega_min) / 4],
            x_length=5.2,
            y_length=3.2,
            tips=False,
            axis_config={"stroke_width": 1.0, "font_size": 22},
        )

        ax_reg = Axes(**axes_kwargs).to_edge(LEFT, buff=0.8)
        ax_cha = Axes(**axes_kwargs).to_edge(RIGHT, buff=0.8)

        reg_labels = ax_reg.get_axis_labels(
            MathTex(r"\theta_2\ (\mathrm{rad})").scale(0.62),
            MathTex(r"\omega_2\ (\mathrm{rad}/\mathrm{s})").scale(0.56),
        )
        cha_labels = ax_cha.get_axis_labels(
            MathTex(r"\theta_2\ (\mathrm{rad})").scale(0.62),
            MathTex(r"\omega_2\ (\mathrm{rad}/\mathrm{s})").scale(0.56),
        )
        reg_title = Text("Regular subset", weight="BOLD").scale(0.45).next_to(ax_reg, UP, buff=0.18)
        cha_title = Text("Chaotic subset", weight="BOLD").scale(0.45).next_to(ax_cha, UP, buff=0.18)

        # Create heatmap ImageMobjects for the first frame.
        reg_img = ImageMobject(str(reg_paths[0]))
        cha_img = ImageMobject(str(cha_paths[0]))

        reg_img.set_width(ax_reg.x_axis.get_length())
        reg_img.set_height(ax_reg.y_axis.get_length())
        reg_img.move_to(ax_reg.c2p(0, (omega_min + omega_max) / 2))

        cha_img.set_width(ax_cha.x_axis.get_length())
        cha_img.set_height(ax_cha.y_axis.get_length())
        cha_img.move_to(ax_cha.c2p(0, (omega_min + omega_max) / 2))

        self.play(FadeIn(ax_reg), FadeIn(ax_cha), FadeIn(reg_labels), FadeIn(cha_labels), run_time=0.8)
        self.play(FadeIn(reg_title), FadeIn(cha_title), run_time=0.4)
        self.play(FadeIn(reg_img), FadeIn(cha_img), run_time=0.6)

        # Step through frames with replacement transforms (no opacity flashing).
        for k in range(1, n_frames):
            next_reg = ImageMobject(str(reg_paths[k]))
            next_cha = ImageMobject(str(cha_paths[k]))

            next_reg.set_width(ax_reg.x_axis.get_length())
            next_reg.set_height(ax_reg.y_axis.get_length())
            next_reg.move_to(ax_reg.c2p(0, (omega_min + omega_max) / 2))

            next_cha.set_width(ax_cha.x_axis.get_length())
            next_cha.set_height(ax_cha.y_axis.get_length())
            next_cha.move_to(ax_cha.c2p(0, (omega_min + omega_max) / 2))

            self.play(
                Transform(reg_img, next_reg),
                Transform(cha_img, next_cha),
                run_time=0.28,
            )
            reg_img = next_reg
            cha_img = next_cha

        self.wait(1.0)

