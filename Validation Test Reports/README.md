# Validation Test Reports

This folder holds **reproducible documentation** of automated tests and full-pipeline ensemble validation runs.

| Document | Purpose |
|----------|---------|
| [`validation_report_2026-03-31.md`](validation_report_2026-03-31.md) | Full write-up: pytest suite, `n=500` ensemble run, energy-check behavior, degenerate logistic outcome, timings |
| [`figures/`](figures/) | PNGs generated from `data/results/ensemble_results.parquet` (see script below) |
| [`generate_validation_figures.py`](generate_validation_figures.py) | Regenerates figures after a new ensemble run |

## Regenerate figures

From the `double_pendulum_sim` directory (with dependencies installed):

```bash
python "Validation Test Reports/generate_validation_figures.py"
```

Requires `ensemble_results.parquet` from a completed `main.py` run (default path: `data/results/ensemble_results.parquet`).

## Related code changes (same validation cycle)

Documented in the report and on `main` around the same date:

- **Scaled energy drift** — `src/physics/energy.py` (`scaled_max_energy_drift`) so tiny `|E(0)|` does not blow up relative drift.
- **Degenerate logistic** — `src/stats/threshold.py` when the ensemble has a single chaos class; `src/output/visualize.py` plots constant \(P\).
