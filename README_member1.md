# Member 1 — Processor Chip Thermal Modelling and Numerical Formulation

This package implements Member 1's part of **Workload-Aware Processor Thermal
Simulation Using Krylov Subspace Methods and Markov Chain Modelling**.

## 1. Responsibilities covered

- 2D processor silicon-die grid and grid spacing.
- Initial temperature and material properties.
- Explicitly documented boundary assumptions.
- Transient heat-conduction model.
- Backward-Euler time discretisation and finite-difference spatial discretisation.
- Sparse matrix `A` and right-hand-side vector `b`.
- Matrix dimension, symmetry and positive-definiteness tests.
- CG residual test and analytic uniform-equilibrium validation.
- Chip heatmaps and temperature-versus-time plots.
- Multiple chip power-region and time-varying heat-input demonstrations.

Member 1 does not own dataset preprocessing, Markov modelling, CG or PCG.
Those are Member 2 responsibilities. Member 1 provides the thermal system
that receives a heat-source map from the combined workload pipeline.

## 2. Governing reduced-order chip equation

The model treats the silicon die as a 2D conducting layer with an effective
vertical path for heat removal:

`rho*cp*dT/dt = k*laplacian(T) - (h_eff/t_die)*(T-T_sink) + p_surface/t_die`

- `rho`: density, kg/m^3
- `cp`: specific heat, J/(kg K)
- `k`: thermal conductivity, W/(m K)
- `t_die`: die thickness, m
- `h_eff`: effective vertical heat-transfer coefficient, W/(m^2 K)
- `T_sink`: package/cooling reference temperature, K
- `p_surface`: generated power per die surface area, W/m^2

The four lateral die edges are insulated (zero normal heat flux). Heat
transfer into the package/cooling path is approximated using the effective
vertical term. This is a reduced 2D educational model, not a full 3D
transistor/die/package/heatsink simulation.

Backward Euler is used in time. The spatial Laplacian is discretised using
a symmetric nearest-neighbour finite-difference stencil. The resulting
matrix is symmetric positive definite for positive material properties,
positive time step and nonnegative effective cooling coefficient.

## 3. Interface for Member 2

The assembly function supports the project-guide interface:

```python
A, b = assemble_system(grid, heat_source, parameters)
```

It also supports:

```python
A, b = assemble_system(grid, parameters)
```

The heat-source map is a 2D array in W/m^2. The matrix is returned in CSR
format and `b` is a one-dimensional NumPy array.

Member 2 can connect a solver through:

```python
temperature, A, b = thermal_time_step(
    grid, parameters, solver=my_cg_or_pcg_function,
    power_density=heat_source
)
```

The solver callback must accept `(A, b)` and return a flattened vector with
`rows * cols` finite temperature values. The built-in sparse direct solver is
used only for small validation/demo cases; use the team's CG/PCG for final
solver comparisons.

## 4. Workload-dependent heat input

`make_power_map()` creates a spatial map with different chip regions.
`run_thermal_simulation(..., power_schedule=maps)` accepts a list of maps,
one per time step. This is the integration point for Member 2's Low,
Medium and High Markov workload states.

The demonstration power maps are manually chosen examples only. They do
not represent measured Ryzen 7 4800U core locations or power values.

## 5. Numerical validation

Tests check:
- Matrix shape and symmetry.
- Positive definiteness on a small grid.
- Compatibility with the documented Member 2 assembly call.
- CG residual.
- Agreement with a known uniform equilibrium solution.
- No-power equilibrium and heating under a positive heat source.
- Time-varying heat-source schedule.

## 6. Run on Windows / Anaconda Prompt

Extract the ZIP and change directory into `Member1_Chip_Thermal_Model_Final`:

```bat
conda create -n chipthermal python=3.11 -y
conda activate chipthermal
pip install -r requirements.txt
python -m pytest -q
python -m experiments.member1_chip_demo
```

The demo creates a `results` folder containing heatmaps, temperature trends
and NumPy temperature histories for the experiments.

## 7. Physical-parameter and dataset limitations

The example values are illustrative silicon/model parameters. They are not
verified Ryzen 7 4800U specifications, measured chip power maps or calibrated
cooling parameters. Dataset acquisition and preprocessing are assigned to
Member 2. For processor-specific quantitative claims, the team must inspect
the selected dataset, verify its units and temporal/spatial resolution, and
calibrate the model parameters against appropriate measurements.

The model's surface power-density input must be spatially and dimensionally
consistent with the model grid. Do not directly interpret a temperature map
as a power map.
