#!/usr/bin/env python3
"""Build PNG summaries from ensemble_results.parquet into figures/."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
_PARQUET = _ROOT / "data" / "results" / "ensemble_results.parquet"
_OUT = Path(__file__).resolve().parent / "figures"


def main() -> None:
    if not _PARQUET.is_file():
        print(f"Missing {_PARQUET}; run main.py first.", file=sys.stderr)
        sys.exit(1)
    _OUT.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(_PARQUET)
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("ggplot")

    # 1) MLE histogram
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.hist(df["mle"], bins=40, color="#2c5282", edgecolor="white", alpha=0.9)
    ax.set_xlabel("Finite-time λ (MLE-style)")
    ax.set_ylabel("Count")
    ax.set_title("Ensemble: distribution of λ")
    fig.tight_layout()
    fig.savefig(_OUT / "mle_histogram.png", dpi=150)
    plt.close(fig)

    # 2) theta1 vs mle
    fig, ax = plt.subplots(figsize=(7, 4.2))
    sc = ax.scatter(
        df["theta1"],
        df["mle"],
        c=df["is_chaotic"].astype(int),
        cmap="coolwarm",
        s=12,
        alpha=0.65,
    )
    ax.set_xlabel(r"$\theta_{1,0}$ (rad)")
    ax.set_ylabel("λ")
    ax.set_title(r"Initial $\theta_1$ vs λ (color: is_chaotic)")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("is_chaotic")
    fig.tight_layout()
    fig.savefig(_OUT / "theta1_vs_mle.png", dpi=150)
    plt.close(fig)

    # 3) energy variance vs mle
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.scatter(df["mle"], df["energy_ratio_variance"], s=12, alpha=0.65, color="#276749")
    ax.set_xlabel("λ")
    ax.set_ylabel(r"var($KE_2/KE_{\rm total}$)")
    ax.set_title("Energy-ratio variance vs λ")
    fig.tight_layout()
    fig.savefig(_OUT / "energy_variance_vs_mle.png", dpi=150)
    plt.close(fig)

    # 4) Chaos label counts
    fig, ax = plt.subplots(figsize=(5, 4))
    vc = df["is_chaotic"].value_counts()
    labels = ["chaotic" if x else "regular" for x in vc.index]
    ax.bar(labels, vc.values, color=["#c05621", "#2b6cb0"])
    ax.set_ylabel("Count")
    ax.set_title("Chaos labels (λ > ε)")
    fig.tight_layout()
    fig.savefig(_OUT / "chaos_label_bar.png", dpi=150)
    plt.close(fig)

    print(f"Wrote figures to {_OUT}")


if __name__ == "__main__":
    main()
