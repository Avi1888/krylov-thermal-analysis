# Workload-Aware Processor Thermal Simulation

A Python project for a reduced two-dimensional chip thermal model, sparse
Conjugate Gradient (CG) and Jacobi-preconditioned CG (PCG) solvers, workload
state modelling, and repeated-solve experiments. The integrated project keeps
Member 1's thermal discretization and Member 2's solver/workload modules as
separate, documented responsibilities.

## Model and finite-volume formulation (Member 1)

The die is represented by a uniform, cell-centred grid. For die dimensions
`W × H` and `cols × rows` control volumes,
`dx = W/cols`, `dy = H/rows`; cell centres are at
`x_c = (c + 1/2) dx`, `y_r = (r + 1/2) dy`. Temperature and power maps keep
shape `(rows, cols)`, with C-order/row-major flattening for solver vectors.

The reduced thermal model is

```text
rho*cp*dT/dt = k*laplacian(T) - (h_eff/t_die)*(T - T_sink) + p_surface/t_die
```

Backward Euler integrated over a cell of area `Ac = dx*dy` and die thickness
`t_die` gives

```text
(rho*cp*t_die*Ac/dt + h_eff*Ac + sum(Gij))*T_i(new)
    - sum(Gij*T_j(new))
  = rho*cp*t_die*Ac/dt*T_i(old) + h_eff*Ac*T_sink + p_i*Ac
```

Here `Gx = k*t_die*dy/dx` and `Gy = k*t_die*dx/dy`. The four lateral die
edges are insulated: zero normal flux is imposed at boundary faces by omitting
conductances to cells outside the die. The capacity term makes the assembled
matrix symmetric positive definite for finite positive density, heat capacity,
conductivity, thickness and time step, with nonnegative cooling coefficient.
The cooling term models an effective vertical package path. This is a reduced
educational model, not a calibrated full processor/package simulation.

The source interface is surface power density in `W/m^2`. A measured
temperature map is not a heat-source map and must not be passed as one.
Visualization converts internal Kelvin values to Celsius using
`T_C = T_K - 273.15`; physical plot extents span the full die dimensions.
See [README_member1.md](README_member1.md) for Member 1's detailed model
assumptions and validation notes.

## Solvers, workload model, and integration (Member 2)

- `src/krylov_solvers.py` implements CG and Jacobi-PCG, returning solver status,
  residual history, independently verified final true residual and timings.
- `src/preconditioners.py` builds Jacobi preconditioners and checks matrix
  fingerprints before reuse.
- `src/markov_model.py` estimates workload-state transitions and generates
  seeded sequences.
- `src/workload_preprocessing.py` loads and summarizes dataset arrays,
  supports explicit metric selection, categorizes scores and maps states to
  relative intensities. It does not infer undocumented metric meanings.
- `src/repeated_solve.py` runs cold/warm-start and rebuild/reuse configurations
  A–D; a changed matrix causes a preconditioner rebuild.
- `src/thermal_integration.py` adapts the solver callback to the thermal loop
  and converts relative workload intensity to explicit `W/m^2` maps.
- `src/benchmark.py` provides summaries and plotting helpers.

Member 1's `run_thermal_simulation(..., solver=...)` accepts a callback with
signature `(A, b) -> flat_solution`. Use
`ThermalSolverAdapter(method="cg")` or
`ThermalSolverAdapter(method="pcg", warm_start=True,
reuse_preconditioner=True)`. The latter reuses Jacobi only when the matrix is
unchanged. `workload_states_to_power_schedule()` requires a caller-selected
reference power density; it is an illustrative scale unless independently
calibrated.

## Dataset and interpretation limits

`data/raw/CPU_R7_4800U.pkl` is the supplied dataset. Its arrays have input shape
`(100, 100, 158)` and output shape `(100, 223, 280)`. The input metric names
and metric-to-meaning mapping are unavailable. Adjacent input windows overlap
by 99 telemetry frames, establishing relative stored order across 199 unique
frames, but no absolute timestamps or sampling interval are available. Code
must not infer utilization, power, or frequency from an unidentified metric
index. Dataset output maps are measured temperatures in Celsius, not power
maps; comparing their spatial distributions quantitatively with the thermal
model requires physical map registration and compatible measurement
conditions.
Synthetic workloads and heat intensities are not measured Ryzen 7 4800U
behavior or calibrated processor power.
The dataset source documentation and research paper are available from the
[dataset documentation](https://sheldonucr.github.io/commercial_thermal_map_dataset/)
and [associated paper](https://doi.org/10.1145/3670474.3685963).

To inspect the supplied dataset and produce per-metric descriptive statistics:

```bash
python -m experiments.inspect_dataset
```

The legacy workload-sequence script can be run with an explicitly selected
numeric column, but its column meaning and units are unavailable. Treat any
result as exploratory and do not label it as measured power, utilization, or a
named workload. The measured thermal-map analysis below does not use telemetry
columns. The script still requires `--assume-chronological-sample-order` as an
explicit acknowledgement of the relative order established by overlapping
windows; that flag does not validate the selected feature's meaning.

```bash
python -m experiments.run_workload_sequence --metric-index INDEX --assume-chronological-sample-order
```

Only load trusted pickle files: Python pickle deserialization can execute code.

## Ryzen 7 4800U Measured Thermal-Map Analysis

The measured-data experiment analyzes the actual `output` array in
`data/raw/CPU_R7_4800U.pkl`. Its input shape is `(100, 100, 158)`: 100
windows, each containing 100 frames and 158 telemetry columns. The output has
100 corresponding measured thermal maps of shape `(223, 280)`. This experiment
uses the measured output maps only; it does not feed them to the FVM or
interpret unidentified telemetry columns.

For every map, the experiment records minimum, maximum, mean, median, standard
deviation, range, 95th and 99th percentiles, and the hottest pixel's row and
column indices. It also summarizes pooled map temperatures and pixel-wise
differences between consecutive stored maps. Figures show a representative
map, maps with the highest maximum and mean temperatures, selected maps in
stored order, and mean/maximum temperatures across all maps. The stored-map
order is relative sequence order, not elapsed seconds; no heating rate is
calculated.

Temperatures are displayed in Celsius because that is the dataset's documented
output unit. Hotspot positions are array row and column indices because the
dataset does not include a verified pixel-to-die coordinate mapping. No
physical dimensions or orientation are inferred.

Run from the repository root:

```bash
python -m experiments.run_ryzen_thermal_analysis
```

The command validates the expected input and output shapes and writes
`results/ryzen_measured/tables/measured_map_statistics.csv`,
`results/ryzen_measured/tables/consecutive_map_changes.csv`, and PNG figures
under `results/ryzen_measured/figures/`.

The two experimental paths remain distinct:

- **Measured Ryzen dataset path:** analyzes actual measured Ryzen thermal maps
  and map-to-map differences. It does not drive the FVM with measured Ryzen
  power and does not provide quantitative physical validation of the FVM.
- **Synthetic simulation path:** uses generated heat-source inputs with the
  existing FVM, CG, PCG, and Markov-based experiments. Its simulated
  temperatures are not measured Ryzen temperatures.

## Installation and verification

Use Python 3.10 or newer from the repository root:

```bash
python -m pip install -r requirements.txt
python -m pytest -q
```

The unified pytest suite includes finite-volume mathematical checks, solver
and Markov tests, adapter/end-to-end integration checks, and experiment-data
consistency checks. Run a focused group with, for example,
`python -m pytest tests/test_finite_volume_assembly.py -q` or
`python -m pytest tests/test_solver_residuals.py tests/test_markov_probabilities.py -q`.

## Demonstrations and experiments

Member 1's illustrative chip demonstrations:

```bash
python -m experiments.member1_chip_demo
```

Member 2's synthetic CG/PCG and four-configuration repeated-solve comparisons:

```bash
python -m experiments.run_solver_comparison
python -m experiments.run_reuse_comparison
```

The approved Stage 3 scripts produce the seeded synthetic workload-aware
thermal run, solver comparison, and A–D repeated-solve measurements:

```bash
python -m experiments.run_solver_comparison --output-dir results/synthetic
python -m experiments.run_reuse_comparison --output-dir results/synthetic
python -m experiments.run_synthetic_thermal_simulation --output-dir results/synthetic
```

The Stage 3 methodology and artifact map are documented in
[`experiments/README_stage3.md`](experiments/README_stage3.md). Synthetic
outputs are kept under `results/synthetic/`; Member 2's earlier experiment
artifacts remain under `results/figures/` and `results/tables/`; Member 1's
demo outputs remain under `results/`. The legacy
`results/tables/dataset_feature_statistics.csv` contains descriptive
statistics for the supplied dataset only; it does not resolve feature
meanings. Relative chronology for the supplied Ryzen windows is established by
their adjacent-window overlap. The guarded real-dataset workload workflow
writes to `results/real_dataset/`, which has not been generated. Saved Stage 3
summary values are documented alongside their methodology in
`experiments/README_stage3.md`. Timing is machine-dependent, observed
small timing differences are not inferential evidence, and no optimization
claim should be made beyond the measurements and their variability. No real
dataset thermal simulation has been run as part of the synthetic experiments.

## Project structure

```text
src/                    thermal model, sparse assembly, solvers, workload, integration
tests/                  unified mathematical, solver, Markov, and integration tests
experiments/            member demos, dataset tools, and reproducible Stage 3 scripts
data/raw/               supplied CPU_R7_4800U dataset
results/                retained member artifacts and synthetic Stage 3 outputs
README_member1.md        detailed Member 1 thermal formulation notes
experiments/README_stage3.md  Stage 3 experiment methodology and outputs
```
