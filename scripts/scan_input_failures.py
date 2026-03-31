"""Large-scale scan of parameter combos that fail numerical validation.

This is a diagnostic tool: it samples many initial-condition parameter sets from
the configured LHS ranges, then attempts integration + energy conservation (and
optionally Lyapunov MLE). It records which error class occurs and the specific
parameter values that trigger it.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

_ROOT = Path(__file__).resolve().parents[1]

_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ensemble.sampler import sample_parameters
from ensemble.lyapunov import compute_mle
from physics.energy import compute_energy_timeseries
from physics.integrator import integrate
from physics.pendulum import dstate_dt


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config root must be a mapping.")
    return cfg


def _validate_param_dict(p: dict[str, Any], config: dict[str, Any]) -> list[str]:
    """Cheap sanity checks; does not attempt to guarantee numerical stability."""
    errs: list[str] = []
    pcfg = config["parameters"]
    keys = ["m1", "m2", "L1", "L2", "theta1", "theta2", "omega1", "omega2"]
    for k in keys:
        if k not in p:
            errs.append(f"missing:{k}")
            continue
        v = float(p[k])
        if not np.isfinite(v):
            errs.append(f"nonfinite:{k}")
            continue
        lo, hi = float(pcfg[k][0]), float(pcfg[k][1])
        if v < lo or v > hi:
            errs.append(f"out_of_bounds:{k}")
    if "g" in p and float(p["g"]) <= 0:
        errs.append("g_nonpositive")
    return errs


_ENERGY_DRIFT_RE = re.compile(
    r"scaled max energy drift\s*([0-9.eE+-]+)\s*exceeds allowed\s*([0-9.eE+-]+)",
)


def _parse_energy_drift_limits(msg: str) -> tuple[float | None, float | None]:
    m = _ENERGY_DRIFT_RE.search(msg)
    if not m:
        return None, None
    return float(m.group(1)), float(m.group(2))


@dataclass
class FailureRecord:
    idx: int
    error_type: str
    message: str
    max_rel_drift: float | None = None
    allowed: float | None = None
    # parameters (for quick debugging)
    m1: float | None = None
    m2: float | None = None
    L1: float | None = None
    L2: float | None = None
    theta1: float | None = None
    theta2: float | None = None
    omega1: float | None = None
    omega2: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "idx": self.idx,
            "error_type": self.error_type,
            "message": self.message,
            "max_rel_drift": self.max_rel_drift,
            "allowed": self.allowed,
            "m1": self.m1,
            "m2": self.m2,
            "L1": self.L1,
            "L2": self.L2,
            "theta1": self.theta1,
            "theta2": self.theta2,
            "omega1": self.omega1,
            "omega2": self.omega2,
        }


def main() -> None:
    p = argparse.ArgumentParser(description="Scan for integration / energy / Lyapunov failures.")
    p.add_argument("--config", type=Path, default=_ROOT / "config.yaml", help="YAML config path.")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for LHS sampling.")
    p.add_argument("--n-samples", type=int, default=300, help="Number of LHS samples to try.")
    p.add_argument(
        "--check-mle",
        action="store_true",
        help="If set, also compute Lyapunov MLE (slower).",
    )
    p.add_argument(
        "--scan-n-steps",
        type=int,
        default=None,
        help="Override integration.n_steps during scan (diagnostic speed knob).",
    )
    p.add_argument(
        "--scan-t-end",
        type=float,
        default=None,
        help="Override integration.t_span[1] during scan (diagnostic speed knob).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=_ROOT / "data" / "results",
        help="Output directory for JSON/CSV reports.",
    )
    args = p.parse_args()

    config = _load_config(args.config)
    integ_cfg = config["integration"]
    if args.scan_n_steps is not None:
        integ_cfg["n_steps"] = int(args.scan_n_steps)
    if args.scan_t_end is not None:
        t0 = float(integ_cfg["t_span"][0])
        integ_cfg["t_span"] = [t0, float(args.scan_t_end)]

    n = int(args.n_samples)
    rng = np.random.default_rng(int(args.seed))
    samples = sample_parameters(n, config, rng)

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_json = out_dir / f"input_failure_scan_{ts}.json"
    out_csv = out_dir / f"input_failure_scan_{ts}_failures.csv"

    counts: dict[str, int] = {"ok": 0}
    failures: list[FailureRecord] = []

    t_start = time.perf_counter()
    for idx, params in enumerate(samples):
        sanity_errs = _validate_param_dict(params, config)
        if sanity_errs:
            rec = FailureRecord(
                idx=idx,
                error_type="invalid_input",
                message=";".join(sanity_errs),
                m1=float(params.get("m1", np.nan)),
                m2=float(params.get("m2", np.nan)),
                L1=float(params.get("L1", np.nan)),
                L2=float(params.get("L2", np.nan)),
                theta1=float(params.get("theta1", np.nan)),
                theta2=float(params.get("theta2", np.nan)),
                omega1=float(params.get("omega1", np.nan)),
                omega2=float(params.get("omega2", np.nan)),
            )
            failures.append(rec)
            counts["invalid_input"] = counts.get("invalid_input", 0) + 1
            continue

        state0 = np.array(
            [params["theta1"], params["omega1"], params["theta2"], params["omega2"]],
            dtype=np.float64,
        )
        try:
            t, y = integrate(dstate_dt, state0, params, config)
            _ = compute_energy_timeseries(t, y, params, config)
            if args.check_mle:
                mle = float(compute_mle(t, y, params, config))
                if not np.isfinite(mle):
                    raise ValueError("Non-finite MLE.")
            counts["ok"] += 1
        except AssertionError as exc:
            msg = str(exc)
            max_rel, allowed = _parse_energy_drift_limits(msg)
            rec = FailureRecord(
                idx=idx,
                error_type="energy_drift",
                message=msg,
                max_rel_drift=max_rel,
                allowed=allowed,
                m1=float(params["m1"]),
                m2=float(params["m2"]),
                L1=float(params["L1"]),
                L2=float(params["L2"]),
                theta1=float(params["theta1"]),
                theta2=float(params["theta2"]),
                omega1=float(params["omega1"]),
                omega2=float(params["omega2"]),
            )
            failures.append(rec)
            counts["energy_drift"] = counts.get("energy_drift", 0) + 1
        except RuntimeError as exc:
            msg = str(exc)
            rec = FailureRecord(
                idx=idx,
                error_type="integration_failed",
                message=msg,
                m1=float(params["m1"]),
                m2=float(params["m2"]),
                L1=float(params["L1"]),
                L2=float(params["L2"]),
                theta1=float(params["theta1"]),
                theta2=float(params["theta2"]),
                omega1=float(params["omega1"]),
                omega2=float(params["omega2"]),
            )
            failures.append(rec)
            counts["integration_failed"] = counts.get("integration_failed", 0) + 1
        except ValueError as exc:
            msg = str(exc)
            rec = FailureRecord(
                idx=idx,
                error_type="lyapunov_value_error",
                message=msg,
                m1=float(params["m1"]),
                m2=float(params["m2"]),
                L1=float(params["L1"]),
                L2=float(params["L2"]),
                theta1=float(params["theta1"]),
                theta2=float(params["theta2"]),
                omega1=float(params["omega1"]),
                omega2=float(params["omega2"]),
            )
            failures.append(rec)
            counts["lyapunov_value_error"] = counts.get("lyapunov_value_error", 0) + 1
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            rec = FailureRecord(
                idx=idx,
                error_type="other_exception",
                message=msg,
                m1=float(params["m1"]),
                m2=float(params["m2"]),
                L1=float(params["L1"]),
                L2=float(params["L2"]),
                theta1=float(params["theta1"]),
                theta2=float(params["theta2"]),
                omega1=float(params["omega1"]),
                omega2=float(params["omega2"]),
            )
            failures.append(rec)
            counts["other_exception"] = counts.get("other_exception", 0) + 1

    elapsed_s = time.perf_counter() - t_start

    # Write JSON summary
    payload = {
        "config_path": str(args.config),
        "seed": args.seed,
        "n_samples": n,
        "check_mle": bool(args.check_mle),
        "scan_overrides": {
            "scan_n_steps": args.scan_n_steps,
            "scan_t_end": args.scan_t_end,
        },
        "integration_cfg": {
            "t_span": list(map(float, config["integration"]["t_span"])),
            "n_steps": int(config["integration"]["n_steps"]),
            "rtol": float(config["integration"]["rtol"]),
            "atol": float(config["integration"]["atol"]),
            "energy_drift_max_relative": float(config["integration"]["energy_drift_max_relative"]),
        },
        "counts": counts,
        "failure_rate": 1.0 - (counts.get("ok", 0) / max(1, n)),
        "elapsed_seconds": elapsed_s,
        "failures": [f.to_dict() for f in failures[:5000]],
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Write CSV with all failures (useful for quick filtering).
    if failures:
        # Keep header stable and human-readable.
        header = list(failures[0].to_dict().keys())
        lines = [",".join(header)]
        for f in failures:
            row = f.to_dict()
            lines.append(",".join(str(row.get(k, "")) for k in header))
        out_csv.write_text("\n".join(lines), encoding="utf-8")

    # Console summary (short)
    ok = counts.get("ok", 0)
    print(f"Scan finished in {elapsed_s:.2f}s: ok={ok}/{n} (failure_rate={1.0 - ok/n:.4%})")
    for k in sorted(counts.keys()):
        if k == "ok":
            continue
        print(f"  {k}: {counts[k]}")
    if failures:
        print(f"Failures written to:\n  {out_json}\n  {out_csv}")


if __name__ == "__main__":
    main()

