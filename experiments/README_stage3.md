# Stage 3 experiments

These experiment workflows keep synthetic software/performance checks separate
from real-dataset thermal-map analysis. Synthetic outputs are written under
`results/synthetic/`; measured Ryzen map-analysis outputs are written under
`results/ryzen_measured/`. The legacy telemetry-sequence workflow defaults to
`results/real_dataset/` and remains exploratory because its selected feature
meaning and units are unavailable. The root README integration is complete and
combines both members' project guidance; `README_member1.md` retains the detailed thermal
formulation, and this file retains the Stage 3 experiment methodology and
output details.

## Run the experiments

From the repository root:

```bash
python -m experiments.run_solver_comparison
python -m experiments.run_reuse_comparison
python -m experiments.run_synthetic_thermal_simulation
```

Each command accepts `--seed`, `--output-dir`, and workload-specific size,
step, repetition, or tolerance options. Defaults are serialized to JSON under
`results/synthetic/configs/`. The solver comparison and four-configuration
repeated-solve experiments use 20 timing repetitions by default and report
mean, median, and standard deviation. They warm both solver paths before
recording and alternate or randomize execution order using a saved seed.

## Methodology

`run_solver_comparison` builds a synthetic 2D Dirichlet diffusion SPD matrix
and applies a spatially varying diagonal congruence scaling. This produces a
nonconstant diagonal while preserving symmetry and positive definiteness. CG
and Jacobi-PCG receive the same matrix, right-hand side, zero initial guess,
tolerance, and iteration limit. Raw runs include iteration count, convergence,
true residual, residual threshold, solver time, PCG setup time, validation
time, and error against the known solution. This is a solver benchmark, not a
measured processor simulation.

For the default `grid_side=20`, this matrix has shape 400 × 400 and represents
400 unknowns (it is not a 400 × 400 thermal grid). Its five-point stencil has
1,920 nonzeros, or 1.2% density. The configuration records the Kronecker
construction and scaling formula.

`run_reuse_comparison` estimates a Markov chain from a fixed synthetic state
trace and generates one seeded workload sequence. Member 2 maps those states
to explicit illustrative W/m² source maps. Member 1's approved finite-volume
assembler builds a reference thermal RHS sequence by advancing a high-accuracy
PCG baseline. Configurations A-D then solve that same matrix and identical RHS
sequence. The coefficient matrix is constant under the fixed grid and
parameters; the reuse utility fingerprints it and rebuilds if a later matrix
does not match. Warm-start configurations pass the previous solution; cold
configurations pass no previous solution. Repetition summaries report solver,
preconditioner setup, validation, and combined time separately.
The A-D table's `cumulative_iterations_per_run_*` values count all iterations
across the 32 solves in one repetition and then summarize those 20 repetitions.
`iterations_per_solve_*` is the corresponding per-solve average. Timing spread
is reported in the summary CSVs as standard deviation; raw repetition records
are retained for alternate spread calculations.

`run_synthetic_thermal_simulation` advances the coupled workload/thermal path
over the generated sequence. The default run uses PCG with warm starts and
Jacobi reuse, validates every returned true residual against tolerance, checks
finite temperatures, and confirms that the matrix fingerprint remains
constant. It writes workload/time and transition plots, selected temperature
heatmaps, maximum/average temperature trends, raw per-step diagnostics, and a
NumPy temperature history.

The source scale and Low/Medium/High intensity levels are illustrative inputs,
not measured Ryzen 7 4800U power. No synthetic state sequence substitutes for
measured dataset telemetry. The thermal configuration JSON records grid and die
dimensions, cell spacing, duration, timestep, material properties, die
thickness, cooling coefficient, sink and initial temperatures, source scale,
and intensity levels. The model state and raw thermal history are in Kelvin;
temperature plots convert to Celsius using `T_C = T_K - 273.15`.

The optional legacy telemetry command is:


```bash
python -m experiments.run_workload_sequence --metric-index INDEX --assume-chronological-sample-order
```

Relative chronology is established by the stride-one overlap between adjacent
input windows. However, the 158 feature names and units remain unavailable, so
the command's selected column cannot currently be assigned a physical meaning
or treated as a named workload, power, frequency, or utilization measure. The
legacy chronology flag acknowledges the verified relative order; it does not
validate the selected feature.

The measured thermal-map analysis uses the actual Celsius-valued output maps
and does not use telemetry columns or drive the FVM. See the root README section
**Ryzen 7 4800U Measured Thermal-Map Analysis** for its command and outputs.

## Outputs

- `raw/`: per-run and per-step CSV files, per-iteration CG/PCG residual
  histories, per-cell thermal maps in K and °C, and the NumPy thermal history.
- `tables/`: aggregate CG/PCG, A-D, warm-start, reuse, workload, transition,
  and thermal summary CSV files.
- `figures/`: convergence, timing summaries, workload/transition plots,
  temperature trends, and selected cell-centred finite-volume heatmaps.
- `configs/`: seeds, solver settings, grid/source parameters, workload trace,
  and experiment assumptions.

Timing is host-dependent and is reported without asserting that an
optimization is faster. Solver time excludes setup and validation; combined
time includes solver, setup, and validation components as recorded by the
project's solver diagnostics.

## Preserved run results

The values below summarize the saved CSV files in this repository, not timing
samples from a later rerun. Solver times are medians across 20 repetitions.
The A-D iteration counts are cumulative over 32 solves in one repetition; the
per-solve figure is the mean across those 32 solves. A-D total time includes
solver, setup, and validation time.

| Solver | Iterations | Median solver time | Median setup time | Median total time |
| --- | ---: | ---: | ---: | ---: |
| CG | 120 | 10.054 ms | 0 ms | 18.9999 ms |
| Jacobi-PCG | 59 | 5.342 ms | 0.551 ms | 15.2742 ms |

Both methods converged in all 20 repetitions. For the four repeated-solve
configurations:

| Configuration | Iterations per run | Mean iterations per solve | Median setup | Median solver | Median total |
| --- | ---: | ---: | ---: | ---: | ---: |
| A: zero start, rebuild | 224 | 7.000 | 9.608 ms | 23.774 ms | 75.638 ms |
| B: zero start, reuse | 224 | 7.000 | 0.302 ms | 23.911 ms | 73.157 ms |
| C: warm start, rebuild | 100 | 3.125 | 9.582 ms | 13.012 ms | 64.990 ms |
| D: warm start, reuse | 100 | 3.125 | 0.307 ms | 13.354 ms | 64.249 ms |

Each configuration converged for all 32 steps in all 20 repetitions. Timing
variability is material relative to some differences between configurations;
these measurements do not establish a statistically reliable runtime
improvement. Warm starts reduced iteration counts for this workload. Jacobi
reuse reduced setup time when the matrix remained unchanged.

The saved synthetic thermal run completed 32/32 PCG steps (100 total
iterations). Its final temperature-map maximum, mean, and minimum were
300.276817 K, 300.098745 K, and 300.039918 K. These are outputs of the
illustrative synthetic workload and are not Ryzen 7 4800U measurements.
