"""Adapters between Member 2 workload/solver code and Member 1's thermal API.

Member 1's thermal loop calls a solver as ``solver(A, b)`` and expects a
temperature vector. Member 2's solvers return ``SolverResult`` objects. This
module bridges that callback and converts relative workload intensities to
the W/m^2 surface-power-density units required by the thermal assembler.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np

from .krylov_solvers import SolverResult, solve_cg, solve_pcg
from .preconditioners import JacobiPreconditioner, build_jacobi_preconditioner, matrix_fingerprint
from .workload_preprocessing import map_workload_to_heat_intensity


@dataclass
class ThermalSolveRecord:
    """Diagnostics for one callback invocation from a thermal time step."""

    step: int
    result: SolverResult
    preconditioner_rebuilt: bool
    matrix_changed: bool


class ThermalSolverAdapter:
    """Stateful ``(A, b) -> solution`` callback for Member 1's thermal loop.

    Set ``warm_start=True`` to use the previous temperature vector as ``x0``.
    For PCG, ``reuse_preconditioner=True`` reuses Jacobi only when the complete
    matrix fingerprint is unchanged; an altered matrix triggers a rebuild.
    """

    def __init__(
        self,
        method: str = "pcg",
        *,
        warm_start: bool = False,
        reuse_preconditioner: bool = False,
        tol: float = 1e-8,
        max_iter: int = 1000,
    ) -> None:
        normalized_method = method.lower()
        if normalized_method not in {"cg", "pcg"}:
            raise ValueError("method must be 'cg' or 'pcg'.")
        if normalized_method == "cg" and reuse_preconditioner:
            raise ValueError("Preconditioner reuse applies only to PCG.")
        if not np.isfinite(tol) or tol <= 0:
            raise ValueError("tol must be a finite positive number.")
        if isinstance(max_iter, bool) or int(max_iter) != max_iter or max_iter < 0:
            raise ValueError("max_iter must be a non-negative integer.")

        self.method = normalized_method
        self.warm_start = bool(warm_start)
        self.reuse_preconditioner = bool(reuse_preconditioner)
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.records: list[ThermalSolveRecord] = []
        self._previous_solution: np.ndarray | None = None
        self._preconditioner: JacobiPreconditioner | None = None
        self._matrix_signature: str | None = None
        self._spd_signature: str | None = None

    def __call__(self, A: Any, b: Any) -> np.ndarray:
        """Solve one thermal step and return only the flat solution vector."""
        preparation_started = perf_counter()
        signature = matrix_fingerprint(A)
        matrix_changed = self._matrix_signature is not None and signature != self._matrix_signature
        check_spd = signature != self._spd_signature
        x0 = self._previous_solution if self.warm_start else None
        preconditioner_rebuilt = False
        setup_time = 0.0

        if self.method == "cg":
            preparation_time = perf_counter() - preparation_started
            result = solve_cg(A, b, x0=x0, tol=self.tol, max_iter=self.max_iter,
                              check_spd=check_spd)
        else:
            if not self.reuse_preconditioner or self._preconditioner is None or matrix_changed:
                self._preconditioner = build_jacobi_preconditioner(A)
                preconditioner_rebuilt = True
                setup_time = self._preconditioner.setup_time
            preparation_time = perf_counter() - preparation_started
            result = solve_pcg(A, b, x0=x0, preconditioner=self._preconditioner,
                               tol=self.tol, max_iter=self.max_iter, check_spd=check_spd)
            if preconditioner_rebuilt:
                result.preconditioner_setup_time = setup_time

        # Include adapter-side hashing and cache checks in validation time;
        # keep Jacobi construction in its own setup-time field.
        result.validation_time += max(0.0, preparation_time - setup_time)
        self.records.append(ThermalSolveRecord(
            len(self.records), result, preconditioner_rebuilt, matrix_changed
        ))
        self._previous_solution = result.solution.copy()
        self._matrix_signature = signature
        if check_spd:
            self._spd_signature = signature
        return result.solution.copy()

    def reset(self) -> None:
        """Clear warm-start state, preconditioner, and collected diagnostics."""
        self.records.clear()
        self._previous_solution = None
        self._preconditioner = None
        self._matrix_signature = None
        self._spd_signature = None

    @property
    def summary(self) -> dict[str, float | int | str]:
        """Aggregate solver and setup metrics across callback invocations."""
        solver_time = sum(record.result.solve_time for record in self.records)
        setup_time = sum(record.result.preconditioner_setup_time for record in self.records)
        validation_time = sum(record.result.validation_time for record in self.records)
        return {
            "method": self.method,
            "steps": len(self.records),
            "converged_steps": sum(record.result.converged for record in self.records),
            "total_iterations": sum(record.result.iterations for record in self.records),
            "solver_time_seconds": solver_time,
            "preconditioner_setup_seconds": setup_time,
            "validation_seconds": validation_time,
            "total_time_seconds": solver_time + setup_time + validation_time,
        }


def workload_states_to_power_schedule(
    states: Any,
    grid_shape: tuple[int, int],
    reference_power_density_w_m2: float,
    *,
    heat_levels: dict[str, float] | None = None,
    spatial_weights: Any | None = None,
) -> list[np.ndarray]:
    """Convert workload labels into maps matching Member 1's W/m^2 interface.

    ``reference_power_density_w_m2`` is an explicit scale selected by the
    caller. It is not calibrated by the dataset. Optional spatial weights are
    relative shape factors normalized to mean one, preserving the chosen
    reference scale independently of grid resolution.
    """
    if len(grid_shape) != 2 or any(
        isinstance(size, bool) or int(size) != size or size <= 0 for size in grid_shape
    ):
        raise ValueError("grid_shape must contain two positive integers.")
    shape = tuple(int(size) for size in grid_shape)
    if (not np.isfinite(reference_power_density_w_m2)
            or reference_power_density_w_m2 < 0):
        raise ValueError("reference_power_density_w_m2 must be finite and non-negative.")
    if spatial_weights is None:
        factors = np.ones(shape, dtype=float)
    else:
        factors = np.asarray(spatial_weights, dtype=float)
        if factors.shape != shape or not np.all(np.isfinite(factors)) or np.any(factors < 0):
            raise ValueError("spatial_weights must be finite, non-negative, and match grid_shape.")
        average = float(factors.mean())
        if average <= 0:
            raise ValueError("spatial_weights must have a positive mean.")
        factors = factors / average

    power_schedule = []
    for state in states:
        relative_intensity = map_workload_to_heat_intensity(str(state), heat_levels)
        power_density = reference_power_density_w_m2 * relative_intensity * factors
        power_schedule.append(power_density.copy())
    return power_schedule
