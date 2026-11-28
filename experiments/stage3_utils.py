"""Shared deterministic inputs and output helpers for synthetic Stage 3 runs."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from src.grid_and_boundary import ChipParameters, create_chip_grid
from src.krylov_solvers import solve_pcg
from src.markov_model import MarkovModel, estimate_markov_model, generate_workload_sequence
from src.sparse_assembly import assemble_system
from src.thermal_integration import workload_states_to_power_schedule


DEFAULT_SEED = 20261128
DEFAULT_TOLERANCE = 1e-9
DEFAULT_HEAT_LEVELS = {"Low": 0.25, "Medium": 0.60, "High": 1.0}
DEFAULT_REFERENCE_POWER_DENSITY = 40000.0


def write_csv(rows: list[dict[str, Any]], path: str | Path) -> Path:
    """Write structured rows to CSV, preserving a stable field order."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("Cannot write an empty CSV table.")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return destination


def write_json(value: dict[str, Any], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def thermal_parameter_config(
    parameters: ChipParameters,
    *,
    rows: int,
    cols: int,
    die_width_m: float,
    die_height_m: float,
    steps: int,
    initial_temperature_k: float,
) -> dict[str, Any]:
    """Return reconstructable settings for the approved finite-volume model."""
    material = asdict(parameters.material)
    return {
        "grid_shape_rows_cols": [rows, cols],
        "grid_spacing_m": {
            "dx": die_width_m / cols,
            "dy": die_height_m / rows,
        },
        "die_dimensions_m": {"width": die_width_m, "height": die_height_m},
        "die_thickness_m": parameters.die_thickness,
        "time_step_seconds": parameters.dt,
        "simulation_steps": steps,
        "simulation_duration_seconds": steps * parameters.dt,
        "material": {
            "density_kg_m3": material["density"],
            "specific_heat_j_kg_k": material["specific_heat"],
            "conductivity_w_m_k": material["conductivity"],
        },
        "effective_vertical_h_w_m2_k": parameters.effective_vertical_h,
        "sink_temperature_k": parameters.sink_temperature,
        "initial_temperature_k": initial_temperature_k,
        "lateral_boundary_condition": "insulated (zero normal heat flux)",
        "temperature_state_units": "K",
        "heat_source_units": "W/m^2 surface power density",
    }


def make_synthetic_markov_workload(
    steps: int,
    seed: int,
) -> tuple[MarkovModel, list[str], tuple[str, ...]]:
    """Generate explicitly synthetic workload states from a fixed state trace."""
    if steps < 1:
        raise ValueError("steps must be positive.")
    observed_trace = (
        "Low", "Low", "Medium", "Low", "High", "Medium", "High", "High",
        "Medium", "Low", "Low", "Medium", "High", "Medium", "Low", "High",
        "High", "Medium", "Medium", "Low", "High", "Low", "Medium", "High",
    )
    model = estimate_markov_model(observed_trace)
    generated = generate_workload_sequence(
        model, steps, initial_state="Low", rng=np.random.default_rng(seed)
    )
    return model, generated, observed_trace


def make_spatial_weights(rows: int, cols: int) -> np.ndarray:
    """Create a synthetic, non-uniform source shape with no processor claim."""
    yy, xx = np.mgrid[0:rows, 0:cols]
    x = (xx + 0.5) / cols
    y = (yy + 0.5) / rows
    first = np.exp(-((x - 0.32) ** 2 + (y - 0.42) ** 2) / 0.035)
    second = 0.7 * np.exp(-((x - 0.72) ** 2 + (y - 0.68) ** 2) / 0.055)
    return 0.15 + first + second


def make_power_schedule(
    states: Iterable[str],
    grid_shape: tuple[int, int],
    reference_power_density_w_m2: float = DEFAULT_REFERENCE_POWER_DENSITY,
) -> list[np.ndarray]:
    """Map synthetic workload labels to explicit surface-power-density maps."""
    return workload_states_to_power_schedule(
        list(states), grid_shape, reference_power_density_w_m2,
        heat_levels=DEFAULT_HEAT_LEVELS,
        spatial_weights=make_spatial_weights(*grid_shape),
    )


def build_finite_volume_rhs_sequence(
    states: list[str],
    *,
    rows: int = 20,
    cols: int = 20,
    dt: float = 1e-4,
    tolerance: float = DEFAULT_TOLERANCE,
    reference_power_density_w_m2: float = DEFAULT_REFERENCE_POWER_DENSITY,
) -> tuple[Any, list[np.ndarray], list[np.ndarray], list[np.ndarray], ChipParameters]:
    """Build identical repeated FVM systems from one deterministic baseline path.

    A high-accuracy PCG baseline advances the reference temperature once. Its
    resulting RHS sequence is then held fixed across A-D to ensure all four
    configurations solve exactly the same systems and workload schedule.
    """
    schedule = make_power_schedule(states, (rows, cols), reference_power_density_w_m2)
    parameters = ChipParameters(dt=dt)
    grid = create_chip_grid(rows, cols, die_width=0.01, die_height=0.01,
                            initial_temperature=parameters.sink_temperature)
    matrix = None
    matrices: list[Any] = []
    right_hand_sides: list[np.ndarray] = []
    baseline_temperatures: list[np.ndarray] = [grid.temperature.copy()]
    baseline_tolerance = min(tolerance * 0.01, 1e-12)

    for source in schedule:
        current_matrix, rhs = assemble_system(
            grid, parameters=parameters, power_density=source
        )
        if matrix is None:
            matrix = current_matrix.copy()
        elif (current_matrix != matrix).nnz != 0:
            raise RuntimeError("Finite-volume coefficient matrix changed unexpectedly.")
        reference_solve = solve_pcg(
            current_matrix, rhs, tol=baseline_tolerance,
            max_iter=10 * rows * cols,
        )
        true_residual = np.linalg.norm(rhs - current_matrix @ reference_solve.solution)
        if (not reference_solve.converged
                or true_residual > baseline_tolerance * max(np.linalg.norm(rhs), 1.0)):
            raise RuntimeError("High-accuracy reference solve failed residual validation.")
        matrices.append(matrix)
        right_hand_sides.append(rhs.copy())
        grid.temperature = reference_solve.solution.reshape(rows, cols).copy()
        baseline_temperatures.append(grid.temperature.copy())

    return matrix, matrices, right_hand_sides, baseline_temperatures, parameters


def summary_stats(values: Iterable[float], prefix: str) -> dict[str, float]:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        raise ValueError("Cannot summarize an empty sample.")
    return {
        f"{prefix}_mean": float(array.mean()),
        f"{prefix}_median": float(np.median(array)),
        f"{prefix}_std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
    }
