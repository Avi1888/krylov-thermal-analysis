# Workload-Aware Processor Thermal Simulation

A Python project studying two-dimensional processor thermal simulation with
sparse Krylov solvers, workload-state Markov chains, and repeated-solve
optimizations. The thermal PDE and sparse-system assembly are being developed
by Member 1; the solver and workload modules are under `src/`.

## Member 2 modules

- `src/krylov_solvers.py`: in-project CG and Jacobi-PCG implementations with
  residual histories, timing, symmetry checks, and SPD validation.
- `src/preconditioners.py`: Jacobi setup and matrix-fingerprint validation for
  safe preconditioner reuse.
- `src/workload_preprocessing.py`: dataset inspection, explicit metric
  selection, quantile state labels, and configurable relative heat-source
  intensity.
- `src/markov_model.py`: transition estimation and workload-sequence
  generation. Unobserved outgoing states are reported and default to explicit
  self-loops.
- `src/repeated_solve.py`: configurations A-D for cold/warm starts and rebuilt/
  reused Jacobi preconditioners. A changed matrix triggers a rebuild.
- `src/benchmark.py`: benchmark summaries and optional Matplotlib plots.

Benchmark summaries separate solver, Jacobi setup, and validation time; the
reported total is their sum. Timings are machine-dependent and the comparison
scripts use synthetic SPD systems, not measured processor performance.

## Setup

Use Python 3.10 or newer. From the repository root:

```bash
python -m pip install -r requirements.txt
```

Run the unit checks with Python's standard library test runner:

```bash
python -m unittest discover -s tests -v
```

## Dataset

The repository includes `data/raw/CPU_R7_4800U.pkl`. Its `input` array has
shape `(100, 100, 158)` and its `output` array has shape `(100, 223, 280)`.
The input metric names are not supplied with this file. The code therefore
requires a user-selected metric index and does not infer that an unlabeled
metric represents utilization, power, or another particular quantity.
The dataset research describes AMD uProf power/temperature telemetry and PMU
event counters, but the pickle does not provide a feature-name-to-index map.
See the [dataset documentation](https://sheldonucr.github.io/commercial_thermal_map_dataset/)
and its [research paper](https://doi.org/10.1145/3670474.3685963).

Inspect the arrays and write per-metric descriptive statistics:

```bash
python -m experiments.inspect_dataset
```

Pickle loading is appropriate here because this is the project's supplied
dataset, but Python pickle files can execute code. Only load trusted files.

## Member 2 experiments

Compare CG and PCG on a known symmetric positive-definite grid problem:

```bash
python -m experiments.run_solver_comparison
```

Compare all four repeated-solve configurations on a synthetic sequence of
gradually changing right-hand sides:

```bash
python -m experiments.run_reuse_comparison
```

Build a workload Markov model from a selected metric:

```bash
python -m experiments.run_workload_sequence --metric-index 0 --assume-chronological-sample-order
```

Choose `--metric-index` only after reviewing its feature statistics and
meaning. The chronology flag is required because the dataset documentation
does not establish the outer sample ordering. The generated heat values are
configurable relative intensities, not calibrated watts or processor power.

Experiment tables and figures are written under `results/tables/` and
`results/figures/`.

The workload-sequence command deliberately requires both an explicit feature
index and a confirmation of outer-sample chronology. Neither should be chosen
from value ranges alone.

## Integration boundary

Member 1's thermal assembler should provide a symmetric positive-definite
matrix `A` and right-hand side `b` for each time step. The solver API accepts
dense NumPy arrays or SciPy sparse matrices. `make_spatial_heat_source()` can
provide a normalized relative source array once the thermal grid shape and
spatial distribution are agreed between members.
