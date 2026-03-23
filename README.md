# Double Pendulum Chaos and Predictability

This project studies a **classic chaotic system**: the planar double pendulum. The goal is to connect **physics** (how the arms actually move), **dynamical systems theory** (whether motion is predictable or sensitive to tiny changes), and **statistics** (what we can say with confidence when parameters vary). The code simulates many pendulums with different masses, lengths, and starting angles, measures how “chaotic” each run is, and builds models that relate initial conditions to **how unevenly kinetic energy is shared between the two bobs** over time.

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
- **statistics:** confidence level and bootstrap iteration count  

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
python -m pytest tests/test_physics.py -v
```

A chaotic benchmark test writes a diagnostic figure of \(\delta(t)\) to `data/results/delta_chaotic_validation.png` for visual inspection of separation growth.

---

## Repository status

The **nonlinear dynamics core** (equations of motion, high-accuracy integration, energy timeseries with a **strict** conservation check, and the **two-trajectory** Lyapunov helper used in tests) is implemented and covered by `tests/test_physics.py`. **Ensemble execution, Gaussian process fitting, logistic thresholding, reporting, and the full `main.py` pipeline** are specified in the project design and are completed incrementally on top of this validated physics layer.

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
