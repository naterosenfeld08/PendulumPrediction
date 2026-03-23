"""Latin Hypercube parameter sampling."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from numpy.random import Generator
from scipy.stats import qmc


def sample_parameters(
    n: int,
    config: Mapping[str, Any],
    rng: Generator,
) -> list[dict[str, float]]:
    """Draw ``n`` parameter dicts via Latin Hypercube over configured ranges.

    Dimensions (in order): m1, m2, L1, L2, theta1, theta2, omega1, omega2.
    Each dict includes ``g`` = 9.81 (m/s²).
    """
    if n < 1:
        raise ValueError("n must be at least 1.")

    pcfg = config["parameters"]
    keys = ["m1", "m2", "L1", "L2", "theta1", "theta2", "omega1", "omega2"]
    lows = np.array([float(pcfg[k][0]) for k in keys], dtype=np.float64)
    highs = np.array([float(pcfg[k][1]) for k in keys], dtype=np.float64)

    lhs = qmc.LatinHypercube(d=len(keys), seed=rng)
    unit = lhs.random(n=n)
    scaled = qmc.scale(unit, lows, highs)

    g = 9.81
    out: list[dict[str, float]] = []
    for row in scaled:
        d = {keys[i]: float(row[i]) for i in range(len(keys))}
        d["g"] = g
        out.append(d)
    return out
