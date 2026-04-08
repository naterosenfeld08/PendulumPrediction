# Pendulum Embedding-First Forecasting

This repository is now organized around one core goal:

1. simulate pendulum trajectories,
2. convert each trajectory window into a fixed embedding,
3. predict future **energy quantities** at multiple horizons.

The focus is short-horizon predictive structure in chaotic dynamics, not long-horizon deterministic certainty.

## v1 Scope

- Systems:
  - double pendulum
  - driven damped single pendulum
- Mandatory embedding:
  - physics-informed fixed vector (`physics_features_v1`)
- Task:
  - multi-horizon energy forecasting
  - targets per horizon: `[kinetic, potential, total]`
- Baselines:
  - persistence
  - linear regression
  - MLP regressor

## Repository Layout

```text
double_pendulum_sim/
├── main.py                       # CLI: generate / train / run-all
├── config.yaml                   # base physics configuration (double pendulum adapter)
├── embedding_config.yaml         # embedding-first dataset/task settings
├── src/
│   ├── systems/                  # system adapters and registry
│   ├── data/                     # trajectory schema, io, generation, splitting
│   ├── embeddings/               # fixed-length embedding implementations
│   ├── tasks/                    # supervised task builders
│   ├── models/                   # forecasting baselines
│   └── experiments/              # runner + metrics + plots
└── tests/                        # schema and evaluator contract tests (+ legacy tests)
```

## Quick Start

From `double_pendulum_sim/`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Generate trajectories (both systems) with leakage-safe trajectory-level splits:

```bash
python main.py generate
```

Train and evaluate all baseline models:

```bash
python main.py train
```

Run end-to-end:

```bash
python main.py run-all
```

## Configuration

### `embedding_config.yaml`

- `dataset.n_per_system`: number of simulated trajectories per system
- `dataset.duration_s`, `dataset.n_steps`: integration horizon/resolution
- `dataset.train_frac`, `dataset.val_frac`: split ratios (test is remainder)
- `task.window_size`: history length for each supervised sample
- `task.horizons`: forecast step offsets
- `task.stride`: window stride

### `config.yaml`

Used by the double pendulum adapter for physical/integration constants and tolerances.

## Artifacts

After running training:

- trajectories: `data/trajectories/{all,train,val,test}/`
- metrics per model: `data/embedding_artifacts/*_metrics.json`
- saved models: `data/embedding_artifacts/*.joblib`
- horizon error curves: `data/embedding_artifacts/horizon_errors.png`

## Tests

Run all tests:

```bash
python -m pytest tests/ -v
```

New contract tests include:

- schema + io roundtrip
- required energy channel validation
- supervised task shape guarantees
- per-horizon evaluation metric correctness

## Notes

- Existing legacy modules under `src/physics`, `src/ensemble`, `src/stats`, and `src/output` remain in the repo.
- The new embedding-first flow is driven by `main.py` and `src/{systems,data,embeddings,tasks,models,experiments}`.
