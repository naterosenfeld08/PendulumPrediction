# Double Pendulum Chaos and Predictability

This project studies a **classic chaotic system**: the planar double pendulum. The **primary research thread** is: **when does the dynamics become predictably chaotic, and how confidently can we state a boundary in initial conditions?** The first-class diagnostic pair is **neighbor separation δ(t)** plus a **logistic threshold in θ₁,₀** at a configured confidence level—implemented end-to-end in simulation, statistics, and the optional Manim GUI.

More broadly, the code connects **physics** (how the arms actually move), **dynamical systems theory** (whether motion is predictable or sensitive to tiny changes), and **statistics** (what we can say when parameters vary). It simulates many pendulums with different masses, lengths, and starting angles, measures how “chaotic” each run is, and builds models that relate initial conditions to **how unevenly kinetic energy is shared between the two bobs** over time.

### Phased visualization roadmap

| Phase | Focus | Status |
|-------|--------|--------|
| **1** | **Scene 3** — δ(t) + logistic threshold; live matplotlib default; versioned export manifest | **Primary** |
| **2** | **Scenes 1–2** — ensemble motion and phase-space density (GUI: optional “experimental” tabs) | After Scene 3 quality gates |
| **3** | Broader scene library / polish | Future |

Quality gates before expanding Phase 2: Scene 3 runs smoothly in live mode without overlapping labels; axes and legend are unambiguous; optional MP4 export matches the live view.

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

**Important caveats (read once, remember later):**

- Any **finite-time** \(\lambda\) is an **estimate**, not an asymptotic invariant; transients and near-recurrences can distort short windows.
- In truly chaotic motion, \(\delta(t)\) eventually **saturates** at the scale of the attractor; very long horizons need more sophisticated **renormalization** schemes not used here.
- The method perturbs **only \(\theta_1\)** initially; that probes a **specific direction** in state space. The **maximal** exponent is a supremum over directions; a single direction can underestimate growth or behave intermittently.

These points do not invalidate the workflow—they are standard context for interpreting Lyapunov numbers from simulation.

---

## Statistical layer (ensemble design)

Beyond individual trajectories, the project is designed to:

1. **Sample** many physically plausible parameter sets (masses, lengths, initial angles and angular velocities) using **Latin Hypercube Sampling** for efficient coverage of ranges defined in `config.yaml`.
2. **Run** simulations in parallel and store results incrementally (e.g., Parquet) so partial progress survives interruptions.
3. **Fit** a **Gaussian Process Regression** model mapping initial conditions to the **variance of \(\mathrm{KE}_2/\mathrm{KE}_{\mathrm{total}}\)** over time, with **bootstrap confidence intervals** for predictions.
4. **Estimate a marginal chaos threshold** in the initial angle \(\theta_{1,0}\) using **logistic regression** on the chaotic label, then read off the angle where the **predicted probability** matches the configured confidence level.
5. **Solve an inverse problem**: given a target variance cap at a confidence level, estimate a **maximum** initial angle consistent with that cap using the model’s **upper confidence bound** and a one-dimensional root finder.

Together, these steps connect **nonlinear mechanics**, **sensitivity analysis**, and **uncertainty-aware prediction**.

---

## Project layout

```text
double_pendulum_sim/
├── config.yaml           # ensemble sizes, parameter ranges, integrator, Lyapunov, statistics
├── main.py               # full pipeline entry point (orchestration)
├── requirements.txt
├── README.md
├── src/
│   ├── physics/          # equations of motion, integration, energy + conservation check
│   ├── ensemble/         # LHS sampling, parallel runs, Lyapunov, incremental I/O
│   ├── stats/            # GPR, bootstrap CIs, logistic threshold, inverse problem
│   └── output/           # figures and text report
├── data/results/         # simulation outputs and generated plots
└── tests/                # physics, ensemble, and statistics tests
```

---

## Configuration

`config.yaml` groups settings for:

- **ensemble:** number of pendulums, random seed  
- **parameters:** ranges for masses, lengths, initial angles and angular velocities  
- **integration:** time span, output count, RK45 tolerances, **hard** energy-drift limit  
- **lyapunov:** chaos threshold \(\varepsilon\), initial separation \(\delta_0\), horizon \(T\)  
- **statistics:** confidence level, bootstrap iteration count, optional GPR training cap, cheaper bootstrap counts for figures  
- **inverse:** target variance cap and grid/bootstrap settings for the inverse angle solve  
- **ensemble:** checkpoint interval and `joblib` worker count (`n_jobs`)  

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

### Scenes

- **`DeltaAndThresholdScene` (primary)** — finite-time separation $\delta(t)$ together with the logistic chaos threshold in $\\theta_{1,0}$. This is the default GUI experience.
- **`EnsembleChaosScene` (experimental)** — ensemble motion (default GUI ensemble size is 32).
- **`PhaseDensityAccumulationScene` (experimental)** — accumulated phase-space density in $(\\theta_2,\\omega_2)$ over time.

Enable the experimental tabs in the sidebar when you want to revisit ensemble and phase-density views.

### GUI

Run the GUI:

```bash
streamlit run gui_app.py
```

**Default behavior:** **Live matplotlib** playback of the exported arrays (single shared frame clock; each tab loops with its own modulo). **Rendered MP4** is optional—opt in when you want shareable Manim files.

Click **Render scenes** to export trajectories and density frames. Optional MP4 rendering is controlled separately; with experimental scenes off, only `DeltaAndThresholdScene` is rendered to disk when MP4 is enabled.

### Export contract (`session_manifest.json`)

Each session folder includes `session_manifest.json` with **`export_contract_version`** (currently `"1.0"`) and a list of required keys for the Scene 3 NPZ. Export runs **validation** before finishing so the GUI and Manim always see a consistent schema.

Rendered videos and exported assets are written under `data/results/manim_sessions/<session_id>/`.

---

## Repository status

The **physics core**, **LHS ensemble** with checkpointed Parquet output, **GPR + bootstrap** variance model (with optional training subsampling), **logistic chaos threshold**, **inverse angle** solve, **matplotlib** diagnostics, **text report**, and **`main.py` CLI** are implemented. Tests in `tests/` cover physics, ensemble sampling/reproducibility, and statistics.

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
