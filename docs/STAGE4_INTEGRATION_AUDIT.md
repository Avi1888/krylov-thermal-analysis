# Stage 4: Project-Wide Integration and Consistency Audit

## Scope and Git state

The audit was performed on `feature/ryzen-measured-map-analysis`. At audit
start, `HEAD` was `60c4bc4e8ea7646d22042df087d4243a4c16c85d`, identical to
`codex/integration`; the merge base was the same commit. Thus the feature branch
contains the expected integration history and the Stage 3 work is still
uncommitted on top of it. The Stage 3 modules, tests, documentation, and
`results/ryzen_measured/` artifacts were present. No reset, rebase, merge,
commit, or push was performed.

The only audit edits were factual documentation wording: the stale statement
that Ryzen chronology needed metadata confirmation was corrected, and the
legacy telemetry command now describes its order flag as an acknowledgement of
known relative order. No numerical algorithm, experiment parameter, dataset,
or result artifact was changed or regenerated.

## Overall assessment

The repository contains two related but separate experiment paths. The
synthetic path couples generated workload states, illustrative heat-source
maps, the finite-volume model, and Krylov solvers. The measured Ryzen path
analyzes the dataset's measured Celsius thermal maps; it does not feed those
maps or unidentified telemetry columns to the finite-volume model. The
numerical methods and saved Stage 3 results are consistent with their scripts.

The current project is suitable for an IEEE report and presentation if these
paths and their limitations are stated explicitly. It is not a measured,
real-time Ryzen FVM simulation and does not establish quantitative physical
validation of the FVM against the measured maps.

Finding counts in this audit: **12 PASS, 9 WARNING, 0 ISSUE**. Warnings record
known limitations or incomplete dataset-specific coupling, not hidden
failures. The `ISSUE` count is zero after correcting the stale chronology text.

## 1. Architecture and module dependencies

### Dataset and measured-map path

`experiments.run_ryzen_thermal_analysis` calls
`src.ryzen_thermal_analysis.load_ryzen_dataset`, which validates the pickle's
`input` and `output` arrays and reads the trusted local dataset. The runner
passes only `output` to `analyze_thermal_maps`,
`analyze_consecutive_map_changes`, and `generate_thermal_figures`. It writes
per-map and consecutive-map CSVs and five figures under
`results/ryzen_measured/`.

This path has no import or call to `src.sparse_assembly`, `src.thermal_model`,
`src.thermal_integration`, `src.krylov_solvers`, or
`src.markov_model`. It is a measured-map analysis workflow, not an FVM input
pipeline.

`experiments.inspect_dataset` and `src.workload_preprocessing` provide general
dataset inspection and explicit-index feature statistics. The separate legacy
`experiments.run_workload_sequence` reads the telemetry input, averages the
selected column across each 100-frame window, bins the resulting 100 scores
into Low/Medium/High labels, estimates a Markov chain, and writes generated
states plus illustrative relative heat-intensity values. It does not use the
measured maps, run the FVM, or establish the selected column's meaning.

### Synthetic FVM and Krylov path

`experiments.stage3_utils` builds seeded synthetic state sequences from a
fixed synthetic state trace using `src.markov_model`; it maps states to
illustrative `W/m^2` maps through `src.thermal_integration`. For the synthetic
thermal run, `experiments.run_synthetic_thermal_simulation` passes the schedule
to `src.thermal_model.run_thermal_simulation` with a `ThermalSolverAdapter`
configured for PCG by default. The thermal model calls
`src.sparse_assembly.assemble_system`, then invokes the supplied `(A, b)`
solver callback and inserts the returned row-major solution into the grid.

The four-configuration experiment builds one finite-volume coefficient matrix
and a fixed sequence of right-hand sides from a high-accuracy PCG reference
path. `src.repeated_solve.compare_four_configurations` runs cold/rebuild,
cold/reuse, warm/rebuild, and warm/reuse PCG variants on those same systems.
Warm starts use the previous returned solution. Jacobi reuse is allowed only
when a content fingerprint of the complete matrix matches.

The separate CG/PCG comparison builds a synthetic scaled diffusion SPD system;
it is not the thermal-grid matrix. The `member1_chip_demo` calls the thermal
model without a solver callback and therefore uses the model's small-demo
SciPy direct-solver fallback. The unified Krylov integration tests explicitly
select CG and PCG and assert that this fallback is not called.

## 2. Mathematical implementation audit

The reduced equation in `README.md` and `README_member1.md` is

```text
rho*cp*dT/dt = k*laplacian(T)
               - (h_eff/t_die)*(T - T_sink)
               + p_surface/t_die
```

This is the requested diffusion equation with the model's source term
`Q = p_surface/t_die - (h_eff/t_die)*(T - T_sink)`. Here `p_surface` is a
surface heat input in `W/m^2`, so dividing by thickness gives a volumetric
source in `W/m^3`; the effective vertical cooling term is also volumetric.

`src.sparse_assembly.assemble_system` divides the backward-Euler finite-volume
cell equation by cell volume. For cell `i`, its implemented coefficients are

```text
(rho*cp/dt + h_eff/t_die + sum_j(w_ij))*T_i(new)
    - sum_j(w_ij*T_j(new))
  = (rho*cp/dt)*T_i(old) + (h_eff/t_die)*T_sink + p_i/t_die
```

where `w_ij = k/dx^2` for horizontal neighbors and `k/dy^2` for vertical
neighbors. This is algebraically equivalent to the integrated cell equation
documented in the README. At insulated die edges, the assembler omits
out-of-domain neighbors, implementing zero normal lateral flux. It stores a
symmetric CSR matrix; C-order flattening uses index `r*cols + c`. Under the
documented positive material, thickness, and timestep assumptions and
nonnegative cooling, the positive capacity term plus the symmetric diffusion
operator makes the matrix positive definite.

**PASS P1 — Discretization and units.** The source map remains the existing
surface-power-density interface in `W/m^2`; the RHS conversion and cooling
units agree with the reduced equation. The measured temperature array is not
used as a source.

**PASS P2 — Boundaries and matrix properties.** The no-neighbor boundary
handling and row-major indexing match the documented cell-centered FVM. Tests
cover interior, edge, and corner rows, zero-flux boundary behavior, symmetry,
positive definiteness, analytic uniform-source evolution, and refinement.

**PASS P3 — CG/PCG residual logic.** `src.krylov_solvers.solve_cg` and
`solve_pcg` independently recompute `b - A @ x` before accepting convergence
and on exit. Both use the documented threshold
`tol * max(norm(b), 1)`, and return the true residual and convergence status.
The tests verify residuals near tolerance and through thermal integration.

**PASS P4 — Jacobi and repeated-solve state.** `build_jacobi_preconditioner`
uses the inverse diagonal, validates strictly positive finite diagonal values,
and fingerprints the full matrix. Reuse is rejected/rebuilt for a changed
matrix. Warm-start state is copied from the preceding solution. A-D experiments
hold the same matrix and RHS sequence fixed.

**PASS P5 — Callback integration.** `thermal_time_step` accepts a callback
`(A, b) -> solution`; `ThermalSolverAdapter` explicitly dispatches CG or PCG.
The integration tests spy on the selected solver and fail if `spsolve` is
called, so the tested Krylov path does not silently use the direct fallback.

**WARNING W1 — Simplified physical model.** Material parameters are constant,
the die is a 2D conducting layer, side boundaries are insulated, and package
cooling is represented by a lumped effective vertical coefficient. These are
documented educational assumptions; current parameters are not Ryzen-specific
calibration.

**WARNING W2 — Finite but physically invalid Kelvin input is accepted.**
`create_chip_grid` and `ChipParameters.validate` check temperature values for
finiteness but do not reject negative initial or sink temperatures. Defaults
and saved experiments use 300 K; this does not affect current results, but a
future physical-input validation pass could require nonnegative Kelvin values.

## 3. Markov implementation audit

`src.markov_model.estimate_markov_model` counts adjacent observed state pairs
and normalizes each state's outgoing transition counts by that row's total.
An unobserved outgoing state receives an explicit self-loop by default, or can
be rejected with `unobserved_outgoing="raise"`. There is no pseudocount
smoothing. The initial distribution represents the first observed label;
`generate_workload_sequence` samples that distribution unless an initial state
is specified, then samples each next state from the current row.

**PASS P6 — Markov probabilities and generation.** Validation and tests check
row normalization, labels, self-loop behavior, explicit initial-state handling,
and deterministic seeded generation.

**PASS P7 — Synthetic Markov-to-thermal connection.** The Stage 3 synthetic
thermal runner actually passes generated Markov labels through the illustrative
heat schedule into the finite-volume solver. The integration tests separately
exercise synthetic labels with both CG and PCG.

**WARNING W3 — Labels are an abstraction.** Low/Medium/High are categorical
states from a hand-specified synthetic training trace in the synthetic
experiments. They are not named Ryzen workloads or states inferred from
verified Ryzen feature channels. The legacy telemetry runner's quantile labels
are exploratory and its heat intensities are illustrative only.

## 4. Ryzen measured-map integration audit

The local pickle contains input shape `(100, 100, 158)` and output shape
`(100, 223, 280)`. The Stage 3 runner validates these shapes and finite values,
then analyzes all 100 output maps in Celsius. It records per-map min/max/mean,
median, standard deviation, range, 95th/99th percentiles, and hottest-pixel row
and column indices. Its coordinate labels are pixel indices; it does not
claim physical locations. It computes 99 pixel-wise current-minus-previous
map differences and summary statistics, without time rates.

**PASS P8 — Measured-map use and separation.** Source inspection confirms the
runner loads the actual pickle and passes only `output` to the map analysis.
The output CSVs and figures are under `results/ryzen_measured/`, separate from
`results/synthetic/`. Neither measured maps nor unidentified channel columns
are used as FVM heat sources.

**PASS P9 — Saved output agreement.** The map-statistics CSV has 100 rows and
the consecutive-map CSV has 99 rows. Their values were checked against
recomputed statistics from the local pickle. The pooled pixel statistics are
min 44.0623 °C, max 55.0432 °C, mean 46.7455 °C, and standard deviation
1.0695 °C. Map 94 has both the highest map maximum (55.0432 °C) and highest
map mean (48.8079 °C). The script output and tables report changes per stored
map step, not per second.

**WARNING W4 — Physical map comparison is not calibrated.** No verified
pixel-to-die registration, physical dimensions, orientation, absolute
timestamps, or sampling interval are present. The outputs support analysis of
the measured images but do not support physical-distance claims or quantitative
FVM validation. The 158 telemetry channels still lack a verified ordered
name/unit manifest.

## 5. Experiment and result inventory

| Experiment | Input data | Method | Output location | Result type |
| --- | --- | --- | --- | --- |
| `run_solver_comparison` | Seeded synthetic RHS and scaled 2D diffusion SPD matrix (400 unknowns, not a 400×400 thermal grid) | CG and Jacobi-PCG; same A, b, zero guess, tolerance, 20 repetitions | `results/synthetic/{raw,tables,figures,configs}/` | Synthetic numerical benchmark |
| `run_reuse_comparison` | Fixed synthetic Markov trace, generated sequence, illustrative power maps; reference PCG-generated fixed RHS sequence | A-D PCG warm-start and Jacobi reuse variants; 32 solves, 20 repetitions | `results/synthetic/{raw,tables,figures,configs}/` | Synthetic FVM/solver experiment |
| `run_synthetic_thermal_simulation` | Synthetic Markov state trace and generated heat schedule | Markov → relative intensity/source map → FVM assembly → default PCG adapter | `results/synthetic/{raw,tables,figures,configs}/` | Synthetic temperatures |
| `member1_chip_demo` | Manually chosen illustrative heat maps and schedules | FVM with `thermal_time_step` direct-solver fallback | `results/` root PNG and NumPy histories | Illustrative simulation |
| `inspect_dataset` | Actual pickle input and output arrays | Shape and descriptive-statistics inspection; no feature-name inference | `results/tables/dataset_feature_statistics.csv` and console | Measured dataset, descriptive only |
| `run_workload_sequence` (legacy, not generated) | Actual telemetry windows, explicitly chosen but unidentified feature index | Window mean → tertile labels → Markov fit/generation → illustrative relative intensity | Defaults to `results/real_dataset/`; that directory is absent | Exploratory telemetry-derived labels, not named Ryzen workloads |
| `run_ryzen_thermal_analysis` | Actual measured output array (all 100 maps) | Per-map Celsius statistics, consecutive-map pixel differences, figures | `results/ryzen_measured/{tables,figures}/` | Measured Ryzen thermal maps |
| Retained early Member 2 solver artifacts | Earlier synthetic runs | Historical quick CG/PCG and reuse summaries | `results/tables/{cg_vs_pcg,reuse_comparison}.csv`, `results/figures/{cg_vs_pcg_residuals,reuse_comparison}.png` | Older synthetic artifacts; distinct from Stage 3 canonical outputs |

The saved Stage 3 solver summary reports 120 mean iterations for CG and 59 for
Jacobi-PCG; all 20 runs converged. Median solver times are 10.054 ms and
5.342 ms respectively, with 0.551 ms median PCG setup time. The four
configurations report 224 cumulative iterations per run for A/B and 100 for
C/D, with all 32 steps converged in every repetition. These values are
traceable to `results/synthetic/tables/*.csv`. The timing records have
host-dependent variation and are not treated as proof of statistically
reliable speedup.

**PASS P10 — Experiment provenance.** The saved JSON configurations distinguish
synthetic experiments and record seeds, matrix/grid settings, solver options,
thermal parameters, workload sequences, and assumptions. Stage 3 CSV values
and corresponding figures are present. Measured and synthetic outputs reside
in separate directories.

**WARNING W5 — Historical artifacts need careful labeling.** The earlier
Member 2 result tables and figures are retained, but current Stage 3 runners
write their canonical results under `results/synthetic/`. Keep historical
quick-run values separate from the repeated Stage 3 summaries when citing
results.

**WARNING W6 — Performance interpretation.** Measurements use finite repeats
on the recorded host and exhibit timing variability. The current documentation
correctly avoids asserting general PCG runtime superiority; report iteration
reductions and measured timings with their scope and variability.

## 6. Documentation consistency

The root README explains the reduced heat equation, cell-centered FVM, row
major array layout, units, insulated boundaries, callback interfaces, and the
separate measured and synthetic paths. `README_member1.md` retains the detailed
Member 1 formulation. `experiments/README_stage3.md` documents the seeded
solver and A-D methods, table labels, timing caveats, and saved results.

During this audit, the following stale chronology text was corrected:

- `README.md`: dataset order is now described as established by adjacent-window
  overlap; absolute time and cadence remain unavailable.
- `experiments/README_stage3.md`: the legacy telemetry workflow is separated
  from measured-map analysis, and its chronology flag is correctly described.
- `src/workload_preprocessing.py`, `inspect_thermal_dataset`: generic inspector
  text no longer says chronology must be established from metadata.
- `experiments/run_workload_sequence.py`, `main`: CLI help/error text now says
  the flag acknowledges relative order and does not validate the metric.

These source-file edits change only documentation strings; the guard and
experiment calculations are unchanged.

**PASS P11 — Measured/synthetic distinction.** Current documentation says the
measured path analyzes actual maps and does not drive the FVM, while the
synthetic path uses generated heat inputs and produces simulated temperatures.
It does not claim a real-time Ryzen simulator or that synthetic temperatures
are measured Ryzen values.

**WARNING W7 — Legacy telemetry runner retains an unnecessary flag.**
`experiments/run_workload_sequence.py`, `main`, still refuses to run unless
`--assume-chronological-sample-order` is supplied, despite the supplied
Ryzen windows' relative order being verified. The wording now explains the
flag accurately; removing the guard would be a behavioral change and is outside
this audit's documentation-only correction.

## 7. Confirmed limitations and future work

**WARNING W8 — Telemetry semantics.** There is no verified ordered mapping of
158 telemetry columns to names, units, or measurement semantics. Do not select
a column and call it power, utilization, frequency, temperature, or a named
workload without an authoritative manifest.

**WARNING W9 — Time, geometry, and operating conditions.** Absolute timestamps,
sampling interval, per-run/workload identifiers, and physical pixel-to-die
registration are unavailable. Do not report per-second thermal rates, physical
hotspot coordinates, or quantitative spatial FVM validation from the current
dataset alone.

Other limitations supported by the implementation:

- The stored Ryzen pickle is a 100-window representative sample; the full
  dataset is larger according to the dataset documentation.
- The FVM uses fixed material properties and an illustrative effective cooling
  coefficient; no processor/package calibration has been performed.
- The current measured-map analysis provides descriptive statistics and image
  differences, not a learned model, power estimate, or physics-based validation.
- Synthetic state levels and heat-source maps are illustrative experiment
  inputs. The Markov model represents the supplied categorical abstraction.
- The direct solver remains the default when the thermal callback is omitted;
  Krylov comparisons and the integrated CG/PCG tests pass explicit callbacks.

Future work, if metadata becomes available, should validate the manifest,
sampling interval, map geometry, and run conditions before adding any
named-feature analysis or power-driven FVM path. Any measured-map/FVM
comparison must first align coordinates and conditions; measured temperatures
must never be passed as a heat-source map.

## 8. Verification and readiness

**PASS P12 — Unified test suite.** `python -m pytest -q` completed with 66
passing tests and no failures. `git diff --check` passed. The integration test
covers the synthetic state-to-source-to-FVM-to-CG/PCG callback path, and the
measured-map tests cover validation, statistics, hotspot indices, and map
differences.

The tracked files under `results/synthetic/` have no diff. No Stage 3 result
was regenerated or overwritten during this audit. The measured outputs remain
in `results/ryzen_measured/`. The working tree is expected to remain uncommitted
on `feature/ryzen-measured-map-analysis` for review.

**Readiness:** appropriate for an IEEE report/PPT that presents (1) the
finite-volume and Krylov implementation, (2) synthetic numerical experiments,
and (3) a separate descriptive analysis of measured Ryzen thermal maps.
Not ready for claims of measured Ryzen-driven FVM simulation, measured power
mapping, or quantitative physical FVM validation.
