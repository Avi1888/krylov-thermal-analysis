"""Transient chip thermal simulation and solver integration."""
import numpy as np
from scipy.sparse.linalg import spsolve
from .sparse_assembly import assemble_system


def insert_solution(grid, solution):
    values = np.asarray(solution, dtype=float).ravel()
    if values.size != grid.rows * grid.cols or not np.all(np.isfinite(values)):
        raise ValueError("Solver must return a finite vector with rows*cols values.")
    grid.temperature = values.reshape(grid.rows, grid.cols).copy()
    return grid.temperature


def thermal_time_step(grid, parameters, solver=None, power_density=None):
    A, b = assemble_system(grid, parameters=parameters, power_density=power_density)
    solution = (spsolve if solver is None else solver)(A, b)
    return insert_solution(grid, solution).copy(), A, b


def run_thermal_simulation(grid, parameters, steps, solver=None, power_schedule=None):
    """Run steps using a constant map or one 2D map per step."""
    if not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer.")
    if power_schedule is not None:
        if isinstance(power_schedule, np.ndarray) and power_schedule.ndim == 2:
            if power_schedule.shape != (grid.rows, grid.cols):
                raise ValueError("Power map shape must match chip grid.")
        elif len(power_schedule) == 0:
            raise ValueError("power_schedule cannot be empty.")
        else:
            for p in power_schedule:
                if np.asarray(p).shape != (grid.rows, grid.cols):
                    raise ValueError("Every scheduled power map must match chip grid.")

    history = [grid.temperature.copy()]
    for step in range(steps):
        if power_schedule is None:
            power = None
        elif isinstance(power_schedule, np.ndarray) and power_schedule.ndim == 2:
            power = power_schedule
        else:
            power = power_schedule[min(step, len(power_schedule)-1)]
        temperature, _, _ = thermal_time_step(grid, parameters, solver, power)
        history.append(temperature)
    return np.stack(history, axis=0)
