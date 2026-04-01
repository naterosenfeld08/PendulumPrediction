# Double Pendulum Chaos and Predictability

Simulation and statistics for a **planar double pendulum**: full nonlinear dynamics, Lyapunov-style separation diagnostics, ensemble sampling, and optional Manim/Streamlit visualization.

**Central questions:** (1) *When does the motion behave like sensitive, hard-to-predict chaos, and how confidently can we state a boundary in initial conditions?* — via **δ(t)**, finite-time **λ**, and a **logistic threshold in θ₁**. (2) *For predicting the instantaneous **energy share** \(KE_2/KE_\mathrm{tot}\) from the **initial state**, where does uncertainty **widen** (sharpness), and at what **time** does an honest out-of-fold interval first **fail** to cover the simulation?* — via **`prediction` + `stats.breakdown`** (see below).

Broader pipeline: LHS ensembles, finite-time chaos labels, **GPR + bootstrap** on an energy-ratio variance proxy, **logistic regression** for a marginal θ₁ threshold, an inverse-style solve, and an optional **prediction-breakdown** pass over time-resolved energy share. See **Limitations and scope** for what these choices do *not* claim.

**Contents:** [Motivation](#why-this-system-matters) · [Model & integration](#physical-model) · [Chaos labels](#chaos-diagnosis-separation-of-nearby-trajectories) · [Statistics](#statistical-layer-ensemble-design) · [Limitations](#limitations-and-scope) · [Layout & config](#project-layout) · [CLI & tests](#installation-and-tests) · [GUI / Manim](#manim-animations-and-gui-optional) · [Roadmap](#roadmap)

### Visualization phases (GUI)

| Phase | Focus | Status |
|-------|--------|--------|
| **1** | **δ(t) + threshold** — primary tab; live matplotlib; versioned export manifest | **Current** |
| **2** | Ensemble + phase-space density (sidebar: experimental tabs) | After quality gates on Phase 1 |
| **3** | Broader polish / extra scenes | Future |

Phase 2 is gated on stable live playback, clear axes/legends, and optional MP4 matching the live δ(t) view.

---

## Why this system matters

A **simple pendulum** (one arm, small swings) is close to a perfect textbook oscillator: its period is nearly independent of amplitude, and if you know its state now, you can forecast it far into the future.

Add a **second arm**, allow **large angles**, and the story changes. The two angles are **coupled** through gravity and tension effects. For many initial conditions, two trajectories that start almost identical **diverge exponentially fast** in time. That behavior is the hallmark of **deterministic chaos**: the model is not random, yet long-term prediction is practically impossible because errors grow faster than any realistic measurement precision.

At the same time, not every initial condition is equally chaotic. Some regions of state space behave more **regularly**. This project explores that mix across an **ensemble** of simulations and uses **Lyapunov-type diagnostics** and **statistical models** to summarize what is predictable and what is not.

---

## Physical model

The simulation uses the **full nonlinear equations of motion** derived from a **Lagrangian** for two point masses at the ends of rigid, massless rods:

- **Generalized coordinates:** \(\theta_1\) and \(\theta_2\), each measured from the **downward vertical** (standard planar convention).
- **State vector:** \([\theta_1,\,\omega_1,\,\theta_2,\,\omega_2]\), with \(\omega_i = \mathrm{d}\theta_i/\mathrm{d}t\).
- **Gravity** \(g\) is included explicitly in the potential energy.

There is **no small-angle linearization**: the trigonometric couplings are retained throughout. That choice is essential; linearized models cannot reproduce the rich dynamics this project is meant to study.

**Mechanical energy** (kinetic plus gravitational potential, with the pivot as reference) should be **conserved** in the ideal mathematical model. In practice, numerical time integration introduces tiny errors. The code therefore checks that the **relative drift** of total energy over a run stays below a strict tolerance; if not, the run fails rather than silently continuing with unreliable dynamics.

---

## Time integration and energy accounting

Trajectories are advanced with **SciPy’s** `solve_ivp` using an **explicit Runge–Kutta pair (RK45)** and **tight relative and absolute tolerances**. For chaotic systems, loose tolerances can make numerical errors look like physical divergence, so conservative settings are deliberate.

For each stored time, the code computes:

- kinetic energy of each bob,
- potential energy of each bob,
- total kinetic energy and the **energy ratio** \(\mathrm{KE}_2 / \mathrm{KE}_{\mathrm{total}}\).

The statistical analyses focus on how the **variance of that ratio over time** behaves from run to run—an interpretable measure of how “unsteady” the partition of kinetic energy is during the simulation window.

---

## Chaos diagnosis: separation of nearby trajectories

One practical way to quantify sensitivity is the **maximal Lyapunov exponent (MLE)** in a **finite-time, numerical** sense. The implementation follows a **two-trajectory** procedure:

1. Integrate a **reference** trajectory from a chosen initial state.
2. Integrate a **nearby** trajectory whose initial state matches the reference except for a **tiny offset in \(\theta_1\) only** (as specified in configuration).
3. At each output time, compute the Euclidean distance \(\delta(t)\) between the two states in \([\theta_1,\omega_1,\theta_2,\omega_2]\)-space.
4. Estimate an exponent using \(\delta(t)\) at a configured horizon:  
   \[
   \lambda \approx \frac{1}{T}\ln\frac{\delta(T)}{\delta_0},
   \]
   where \(\delta_0\) is the initial separation scale and \(T\) is the Lyapunov horizon \(T\) from configuration.

A run is labeled **chaotic** when \(\lambda > \varepsilon\), with \(\varepsilon\) configurable (a strict choice is \(\varepsilon = 0\)).

**Caveats in brief:** finite-time \(\lambda\) is not an asymptotic exponent; separation can **saturate** (no renormalization here); perturbation is **only in \(\theta_1\)**. For a full discussion of what this implies—and how the chaos label and statistics relate—see **Limitations and scope**.

---

## Statistical layer (ensemble design)

Beyond individual trajectories, the project is designed to:

1. **Sample** many physically plausible parameter sets (masses, lengths, initial angles and angular velocities) using **Latin Hypercube Sampling** for efficient coverage of ranges defined in `config.yaml`.
2. **Run** simulations in parallel and store results incrementally (e.g., Parquet) so partial progress survives interruptions.
3. **Fit** a **Gaussian Process Regression** model mapping initial conditions to the **variance of \(\mathrm{KE}_2/\mathrm{KE}_{\mathrm{total}}\)** over time, with **bootstrap confidence intervals** for predictions.
4. **Estimate a marginal chaos threshold** in the initial angle \(\theta_{1,0}\) using **logistic regression** on the chaotic label, then read off the angle where the **predicted probability** matches the configured confidence level.
5. **Solve an inverse problem**: given a target variance cap at a confidence level, estimate a **maximum** initial angle consistent with that cap using the model’s **upper confidence bound** and a one-dimensional root finder.

Together, these steps connect **nonlinear mechanics**, **sensitivity analysis**, and **uncertainty-aware prediction**.

### Prediction breakdown (target, sharpness, breakdown time)

This is the path aligned with treating the project as **prediction of a dynamical quantity from initial data**, with **sharpness** (interval width) and a **time of first statistical failure**:

- **Target:** at sampled times \(t_k\) along the same integration grid, \(y_i(t_k)=\mathrm{KE}_2/\mathrm{KE}_\mathrm{tot}\) for ensemble member \(i\). Values are stored in `data/results/ensemble_energy_ratio_timeseries.npz` when `prediction.enabled` is true (see `config.yaml`).
- **Predictor:** \(K\)-fold **out-of-fold** Gaussian process regression: at each \(t_k\), train on initial features (the same eight as elsewhere) → \(y(t_k)\). Inner fits use **fast** GP mode (`optimize_hyperparameters=False`) so scanning many times stays practical; set `prediction.interval_method: bootstrap` to use bootstrap intervals instead of the default **analytic** GP predictive band.
- **Sharpness:** width of the OOF predictive interval vs. \(t\) (e.g. median width across held-out runs)—figure `prediction_breakdown_sharpness.png`.
- **Breakdown time \(t_i^\*\):** first sampled \(t_k\) where the **simulated** \(y_i(t_k)\) lies **outside** the OOF interval for that run. Runs that never leave the band have \(t^\*=\) NaN. Cross-check vs. \(\lambda\) in `prediction_breakdown_vs_mle.png`.

`main.py` runs this block after the ensemble when the NPZ exists; results are summarized in `summary_report.txt`.

---

## Limitations and scope

The pipeline is a **computational study** with explicit tradeoffs. The main limitations are:

1. **Lyapunov-style sensitivity is crude for general claims.** The estimate uses a **single perturbation direction** (\(\theta_1\) only), **no renormalization** when separation grows, and a **finite-time** horizon—so bias from transients, saturation, and direction choice is expected. Treat it as a **local sensitivity probe**, not a full Lyapunov spectrum or asymptotic exponent.

2. **The chaos label \(\lambda > \varepsilon\) is a working rule, not physics.** It is **arbitrary** in the sense that it encodes a study-specific cutoff. It is **sensitive** to horizon \(T\), initial scale \(\delta_0\), and numerical tolerances; changing those changes who gets labeled chaotic.

3. **Logistic regression on \(\theta_{1,0}\) alone is a deliberate simplification.** It reduces a **high-dimensional** initial condition (and parameter) space to a **single marginal** coordinate. **Interactions** with \(\theta_2\), angular velocities, masses, and lengths are **not** represented in that boundary—interpret it as an interpretable 1D summary, not a complete predictor.

4. **Energy-variance (and related) proxies are not standard chaos metrics.** Summaries based on how \(\mathrm{KE}_2/\mathrm{KE}_{\mathrm{total}}\) varies over time measure **irregularity of energy sharing** in the window you simulate. **Correlation** with a chaos label based on \(\delta(t)\) is **not** equivalence with a textbook definition of chaos.

**Despite this**, the workflow is **internally consistent**: the same definitions drive simulation, labeling, and statistics end to end, and the tradeoffs above are **explicit and reasonable** for a bounded numerical exploration—provided results are read as **conditional on those definitions**, not as universal statements about the double pendulum.

---

## Project layout

```text
double_pendulum_sim/
├── config.yaml
├── main.py
├── gui_app.py            # Streamlit: δ(t)+threshold primary; optional experimental scenes
├── requirements.txt
├── requirements-manim.txt
├── requirements-gui.txt
├── README.md
├── manim_scenes/         # Manim scenes (export-driven)
├── src/
│   ├── physics/
│   ├── ensemble/
│   ├── stats/
│   └── output/           # figures, report, manim_export / manim_render helpers
├── data/results/         # Parquet, figures, manim_sessions/
└── tests/
```

---

## Configuration

`config.yaml` groups settings for:

- **ensemble:** number of pendulums, random seed, checkpoint interval, `joblib` workers (`n_jobs`)  
- **parameters:** ranges for masses, lengths, initial angles and angular velocities  
- **integration:** time span, output count, RK45 tolerances, **hard** energy-drift limit  
- **lyapunov:** chaos threshold \(\varepsilon\), initial separation \(\delta_0\), horizon \(T\)  
- **statistics:** confidence level, bootstrap iteration count, optional GPR training cap, cheaper bootstrap counts for figures  
- **inverse:** target variance cap and grid/bootstrap settings for the inverse angle solve  

All simulation code is intended to **read these values** rather than hide tunables inside the source.

---

## Installation and tests

Create a virtual environment, install dependencies, and run the physics validation suite:

```bash
cd double_pendulum_sim
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install pytest          # if not already installed
python -m pytest tests/ -v
```

A chaotic benchmark test writes a diagnostic figure of \(\delta(t)\) to `data/results/delta_chaotic_validation.png` for visual inspection of separation growth.

**Validation reports:** documented ensemble validation runs, metrics, and reproducibility notes live under [Validation Test Reports](Validation%20Test%20Reports/README.md) (including generated figures after `main.py`).

### Full pipeline

From the project directory (after installing dependencies):

```bash
python main.py                    # use existing ensemble Parquet if present
python main.py --force            # re-run ensemble, then statistics, figures, report
python main.py --config path.yaml --n 1000
```

Outputs include `data/results/ensemble_results.parquet`, `ensemble_checkpoint.parquet`, five figures under `data/results/`, and `data/results/summary_report.txt`. The default ensemble size (500) is computationally heavy; use `--force` only when you want a fresh run.

---

## Manim animations and GUI (optional)

This repository also supports high-quality animated visualizations using ManimCE (Manim Community Edition).
The animations are driven by **precomputed physics exports**, so Manim does not re-run the ODE during animation.

### Setup

```bash
cd double_pendulum_sim
pip install -r requirements-manim.txt
pip install -r requirements-gui.txt
```

### GUI (interactive analysis)

Install core dependencies **and** the GUI extra (Streamlit), then start the app:

```bash
cd double_pendulum_sim
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt -r requirements-gui.txt
streamlit run gui_app.py
```

One-liner from the project directory (if dependencies are already installed):

```bash
python3 -m streamlit run gui_app.py
```

The app lets you set **LHS parameter ranges**, **ensemble size** (default **48**), **confidence level** (for the logistic threshold), **Lyapunov** settings (ε, δ₀, horizon T), and **integration** span. Quick **presets** (*Mild* / *Mixed* / *Wild*) adjust θ₁, θ₂, and angular-velocity boxes; *Mixed* matches `config.yaml`. Click **Run ensemble & statistics** to:

1. **Simulate** the full ensemble (same pipeline as `main.py` for that run).
2. **Fit** GPR on energy-ratio variance and **logistic** P(chaotic|θ₁) at your confidence.
3. **Visualize** (matplotlib, live): ensemble motion, **three phase portraits** (min / max λ and median-θ₁ member; optional arm 1 or 2 plane), accumulated **(θ₂, ω₂)** density, and **δ(t) + logistic threshold** — all driven by the **same** DataFrame as the statistics.

Sessions are written under `data/results/gui_sessions/<id>/` (NPZ + manifest, compatible with the Scene 3 contract).

### Manim / optional MP4

For **pre-rendered Manim** scenes, use `export_manim_session` / `manim_render` (CLI or scripts). Manim scenes:

- **`DeltaAndThresholdScene`** — δ(t) + logistic threshold.
- **`EnsembleChaosScene`** — ensemble motion.
- **`PhaseDensityAccumulationScene`** — accumulated phase-space density in $(\\theta_2,\\omega_2)$.

### Export contract (`session_manifest.json`)

Each session folder includes **`export_contract_version`** (currently `"1.0"`) and required keys for the Scene 3 NPZ. Manim export runs validation; GUI sessions perform a lightweight NPZ key check.

Rendered Manim assets default to `data/results/manim_sessions/<session_id>/`.

---

## Roadmap

Short list of directions that build on the current design (not an immediate commitment):

- **Sensitivity / Lyapunov:** multiple perturbation directions; optional Benettin-style or renormalized separation windows; sensitivity of \(\lambda\) to \(T\) and \(\delta_0\) (sweeps or plots).
- **Classification:** replace or complement \(\lambda > \varepsilon\) with a reported **ROC** or calibration curve; document dependence on \(T\).
- **Statistics:** multivariate logistic or GAM including \(\theta_2\), \(\omega\), or key parameters; interaction terms where data support them.
- **GUI / Manim:** finish Phase 2 scenes (smooth live playback, color semantics, no duplicate static panels); optional **Streamlit Cloud** or one-command demo.
- **Reproducibility:** pinned optional `requirements-lock`, small **fixture session** for Manim/GUI smoke tests, CI on `pytest`.
- **Write-up:** short methods note (PDF or `docs/`) mirroring **Limitations and scope** for a portfolio or preprint.

---

## Repository status

Implemented: nonlinear integrator with energy checks, LHS ensemble + Lyapunov-style labels, GPR/bootstrap, logistic threshold, inverse solve, matplotlib report, CLI, tests (`tests/`), and optional GUI/Manim export path.

---

## Further reading

- Coupled oscillators and **normal modes** (why a double pendulum is not two independent simple pendulums).  
- **Lyapunov exponents** and **trajectory separation** in nonlinear dynamics texts.  
- **Latin Hypercube Sampling** for parameter space exploration in computational experiments.  
- **Gaussian processes** for flexible regression with uncertainty; **bootstrap** methods for nonparametric confidence intervals.

---

## Author

[Nathaniel Rosenfeld](https://github.com/naterosenfeld08) — prospective physics and applied math student (Brooklyn, NY).

## License and citation

Choose a license (for example MIT) in the GitHub repository settings or add a `LICENSE` file when you publish. If you use this work in research, cite the repository URL and commit hash.
