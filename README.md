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
- Embedding benchmark set:
  - `physics_features_v1`
  - `raw_window`
  - `fft_features`
  - `hybrid_physics_fft`
- Task:
  - multi-horizon energy forecasting
  - targets per horizon: `[kinetic, potential, total]`
- Model zoo:
  - persistence, linear, ridge
  - random forest, gradient boosting
  - MLP variants (`mlp_small`, `mlp_medium`)
  - optional XGBoost if installed

## Repository Layout

```text
double_pendulum_sim/
├── main.py                       # CLI: generate / sweep / select / final-eval / run-all
├── config.yaml                   # base physics configuration (double pendulum adapter)
├── embedding_config.yaml         # long-run training regimen configuration
├── src/
│   ├── systems/                  # system adapters and registry
│   ├── data/                     # trajectory schema, io, generation, splitting
│   ├── embeddings/               # fixed-length embedding implementations
│   ├── tasks/                    # supervised task builders
│   ├── models/                   # model zoo and hyperparameter sampling
│   └── experiments/              # sweep/select/final-eval + diagnostics
└── tests/                        # contract tests for schema, eval, and regimen behavior
```

## Quick Start

From `double_pendulum_sim/`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Stage 1: Generate data (70/20/10 split)

```bash
python3 main.py generate
```

### Stage 2: Sweep train/validation search

```bash
python3 main.py sweep
```

### Stage 3: Select top candidates by validation objective

```bash
python3 main.py select
```

### Stage 4: Locked test evaluation

```bash
python3 main.py final-eval
```

### End-to-end orchestration

```bash
python3 main.py run-all
```

### Quick compatibility run (small baseline path)

```bash
python3 main.py train
```

## Configuration

### `embedding_config.yaml`

- `dataset.n_per_system`: number of simulated trajectories per system
- `dataset.duration_s`, `dataset.n_steps`: integration horizon/resolution
- `dataset.train_frac`, `dataset.val_frac`: split ratios (`test = 1 - train - val`)
- `task.window_size`: history length for each supervised sample
- `task.horizons`: forecast step offsets
- `task.stride`: window stride
- `experiment.seed_grid`: repeated seeds for robustness checks
- `experiment.embedding_list`: embedding variants to evaluate
- `experiment.model_list`: model families to search
- `search.trials_per_model`: randomized trials per model per seed
- `selection.primary_metric`: model ranking objective (currently weighted RMSE)
- `selection.horizon_weights`: short-horizon-heavy weighting for objective
- `selection.top_k`: number of candidates to promote to locked test

### `config.yaml`

Used by the double pendulum adapter for physical/integration constants and tolerances.

## Artifacts

After running `sweep` + `select` + `final-eval`:

- trajectories: `data/trajectories/{all,train,val,test}/`
- split integrity summary: `data/trajectories/split_summary.json`
- trial log: `data/embedding_artifacts/trials.json`
- leaderboard: `data/embedding_artifacts/leaderboard.json` and `.csv`
- selected candidates: `data/embedding_artifacts/selected_candidates.json`
- locked final report: `data/embedding_artifacts/final_eval.json`
- saved finalist models: `data/embedding_artifacts/models/*.joblib`
- diagnostics per finalist: `data/embedding_artifacts/diagnostics/*/`

## Overfitting and Model Selection Policy

- Search and ranking use **validation only** (no test peeking).
- Locked test is run only after candidate selection.
- Overfitting checks include:
  - train/val/test RMSE gap report
  - per-horizon split gap curves
  - residual histogram on test
- A model is considered risky if:
  - train RMSE is much lower than val RMSE consistently
  - validation gains collapse on test
  - performance is unstable across seeds

## Tests

Run all tests:

```bash
python3 -m pytest tests/ -v
```

New contract tests include:

- schema + io roundtrip
- required energy channel validation
- supervised task shape guarantees
- per-horizon evaluation metric correctness
- split integrity (`70/20/10` + no overlap)
- validation-only selection behavior
- test-lock workflow behavior

## Notes

- Existing legacy modules under `src/physics`, `src/ensemble`, `src/stats`, and `src/output` remain in the repo.
- The new embedding-first flow is driven by `main.py` and `src/{systems,data,embeddings,tasks,models,experiments}`.
