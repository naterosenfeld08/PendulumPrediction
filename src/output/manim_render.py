"""Render Manim scenes using precomputed exports.

This module assumes Manim is installed and accessible on PATH.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Iterable


def _require_manim() -> str:
    manim = shutil.which("manim")
    if not manim:
        raise RuntimeError(
            "Manim executable not found on PATH. Install Manim via requirements-manim.txt "
            "and ensure `manim` is available."
        )
    return manim


def _quality_flag(quality: str) -> str:
    q = quality.strip().lower()
    mapping = {
        "low": "ql",
        "medium": "qm",
        "high": "qh",
        "2k": "qk",
        "4k": "qk",
    }
    if q not in mapping:
        raise ValueError(f"Unknown quality={quality!r}. Expected one of {sorted(mapping.keys())}.")
    return f"-{mapping[q]}"


def _find_mp4_recursive(root: Path) -> list[Path]:
    return sorted([p for p in root.rglob("*.mp4") if p.is_file()], key=lambda p: p.stat().st_mtime)


def render_scenes(
    session_dir: Path,
    scene_specs: Iterable[tuple[Path, str]],
    *,
    quality: str = "high",
    media_subdir: str = "manim_media",
) -> dict[str, Path]:
    """Render each `(scene_file, scene_class)` and return scene_class -> mp4 path."""
    manim_exe = _require_manim()

    session_dir.mkdir(parents=True, exist_ok=True)
    media_dir = session_dir / media_subdir
    media_dir.mkdir(parents=True, exist_ok=True)

    rendered: dict[str, Path] = {}
    qflag = _quality_flag(quality)

    for scene_file, scene_class in scene_specs:
        env = os.environ.copy()
        env["MANIM_SESSION_DIR"] = str(session_dir)
        cmd = [
            manim_exe,
            qflag,
            "--media_dir",
            str(media_dir),
            str(scene_file),
            scene_class,
        ]
        proc = subprocess.run(
            cmd,
            cwd=str(session_dir.parent),
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            stderr = (proc.stderr or "").strip()
            stdout = (proc.stdout or "").strip()
            msg = (
                f"Manim render failed for scene {scene_class}.\n"
                f"Command: {' '.join(cmd)}\n"
                f"Exit code: {proc.returncode}\n"
                f"STDERR:\n{stderr}\n\nSTDOUT:\n{stdout}"
            )
            raise RuntimeError(msg)

        # Locate the newest mp4 generated for this scene file.
        mp4s = _find_mp4_recursive(media_dir)
        if not mp4s:
            raise RuntimeError(f"No mp4 files found under {media_dir}")

        # Best-effort heuristic: match SceneClass.mp4 or last modified.
        expected_name = f"{scene_class}.mp4"
        matches = [p for p in mp4s if p.name == expected_name]
        if matches:
            rendered[scene_class] = matches[-1]
        else:
            rendered[scene_class] = mp4s[-1]

    return rendered

