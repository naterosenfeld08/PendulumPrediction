from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from manim import BLUE, ORANGE, Dot, Line, VGroup, ValueTracker
from manim import Scene


def _session_dir() -> Path:
    sd = os.environ.get("MANIM_SESSION_DIR", "").strip()
    if not sd:
        raise RuntimeError("MANIM_SESSION_DIR is not set.")
    return Path(sd).expanduser().resolve()


class EnsembleChaosScene(Scene):
    """Animate an ensemble of double pendulums (state evolution in time)."""

    def construct(self) -> None:
        session_dir = _session_dir()
        data = np.load(session_dir / "ensemble_scene_data.npz")

        t = data["t"]  # (n_frames,)
        theta1 = data["theta1"]  # (n_ensemble, n_frames)
        theta2 = data["theta2"]  # (n_ensemble, n_frames)
        L1 = data["L1"]  # (n_ensemble,)
        L2 = data["L2"]  # (n_ensemble,)
        is_chaotic = data["is_chaotic"].astype(bool)  # (n_ensemble,)

        n_ensemble, n_frames = theta1.shape

        # Arrange pendulums in a centered grid that fills the frame.
        cols = int(np.ceil(np.sqrt(n_ensemble)))
        rows = int(np.ceil(n_ensemble / cols))
        frame_w = 13.2
        frame_h = 7.2
        pad_x = frame_w / max(cols, 1)
        pad_y = frame_h / max(rows, 1)

        # Visual scale: keep each pendulum inside ~60% of a cell height.
        L_total_max = float(np.max(L1 + L2))
        if L_total_max <= 0:
            L_total_max = 1.0
        cell_target = min(0.60 * pad_y, 0.42 * pad_x)
        scale = cell_target / L_total_max

        # Place grid centers around the origin.
        centers = []
        for i in range(n_ensemble):
            r = i // cols
            c = i % cols
            x = (c - (cols - 1) / 2.0) * pad_x
            y = ((rows - 1) / 2.0 - r) * pad_y
            centers.append((x, y))

        # Time tracker for animation.
        k = ValueTracker(0)

        def coords_for_pendulum(i: int, kk: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            th1 = float(theta1[i, kk])
            th2 = float(theta2[i, kk])
            l1 = float(L1[i]) * scale
            l2 = float(L2[i]) * scale
            x1 = l1 * np.sin(th1)
            y1 = -l1 * np.cos(th1)
            x2 = x1 + l2 * np.sin(th2)
            y2 = y1 - l2 * np.cos(th2)
            x0, y0 = centers[i]
            return np.array([x0, y0, 0]), np.array([x0 + x1, y0 + y1, 0]), np.array([x0 + x2, y0 + y2, 0])

        pendulums = []
        for i in range(n_ensemble):
            center, bob1, bob2 = coords_for_pendulum(i, 0)
            color = ORANGE if is_chaotic[i] else BLUE
            line1 = Line(center, bob1, stroke_width=2.0, color=color)
            line2 = Line(bob1, bob2, stroke_width=2.0, color=color)
            d1 = Dot(bob1, radius=0.04, color=color)
            d2 = Dot(bob2, radius=0.04, color=color)

            def updater_line1(mob, idx=i):
                kk = int(round(k.get_value()))
                _, b1, _ = coords_for_pendulum(idx, kk)
                mob.put_start_and_end_on(coords_for_pendulum(idx, kk)[0], b1)

            def updater_line2(mob, idx=i):
                kk = int(round(k.get_value()))
                _, b1, b2 = coords_for_pendulum(idx, kk)
                mob.put_start_and_end_on(b1, b2)

            line1.add_updater(updater_line1)
            line2.add_updater(updater_line2)

            def updater_d1(mob, idx=i):
                kk = int(round(k.get_value()))
                _, b1, _ = coords_for_pendulum(idx, kk)
                mob.move_to(b1)

            def updater_d2(mob, idx=i):
                kk = int(round(k.get_value()))
                _, _, b2 = coords_for_pendulum(idx, kk)
                mob.move_to(b2)

            d1.add_updater(updater_d1)
            d2.add_updater(updater_d2)

            pendulums.append(VGroup(line1, line2, d1, d2))

        self.add(*pendulums)
        self.wait(0.2)
        self.play(k.animate.set_value(n_frames - 1), run_time=12.0, rate_func=lambda t: t)

        # Cleanup updaters so the scene can end cleanly.
        for group in pendulums:
            for mob in group:
                mob.clear_updaters()

        self.wait(0.3)

