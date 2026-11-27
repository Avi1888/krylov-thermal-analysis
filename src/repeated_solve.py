"""Warm-start and Jacobi-preconditioner reuse experiments."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Sequence

import numpy as np
from scipy import sparse

from .krylov_solvers import SolverResult, solve_pcg, validate_spd
from .preconditioners import (
    JacobiPreconditioner,
    build_jacobi_preconditioner,
    matrix_fingerprint,
)


@dataclass
class StepSolveRecord:
    step: int
    result: SolverResult
    preconditioner_rebuilt: bool
    matrix_changed: bool


@dataclass
class RepeatedSolveResult:
    configuration: str
    warm_start: bool
    reuse_preconditioner: bool
    records: list[StepSolveRecord]

    @property
    def total_iterations(self) -> int:
        return sum(record.result.iterations for record in self.records)

    @property
    def total_solver_time(self) -> float:
        return sum(record.result.solve_time for record in self.records)

    @property
    def total_preconditioner_setup_time(self) -> float:
        return sum(record.result.preconditioner_setup_time for record in self.records)

    @property
    def total_validation_time(self) -> float:
        return sum(record.result.validation_time for record in self.records)

    @property
    def total_time(self) -> float:
        return (self.total_solver_time + self.total_preconditioner_setup_time
                + self.total_validation_time)

    @property
    def final_residual(self) -> float:
        return self.records[-1].result.final_residual if self.records else 0.0


def _is_single_matrix(value: Any) -> bool:
    return sparse.issparse(value) or np.asarray(value).ndim == 2


def _expand_matrices(matrices: Any, n_steps: int) -> list[Any]:
    if _is_single_matrix(matrices):
        return [matrices] * n_steps
    result = list(matrices)
    if len(result) != n_steps:
        raise ValueError("Provide one matrix or exactly one matrix per RHS.")
    return result


def _expand_rhs(rhs_sequence: Any) -> list[np.ndarray]:
    if isinstance(rhs_sequence, np.ndarray):
        if rhs_sequence.ndim == 1:
            return [rhs_sequence.astype(float, copy=False)]
        if rhs_sequence.ndim == 2:
            return [row.astype(float, copy=False) for row in rhs_sequence]
        raise ValueError("RHS input must be a vector, a 2D array of rows, or a sequence of vectors.")
    values = list(rhs_sequence)
    if not values:
        return []
    if all(np.isscalar(value) for value in values):
        return [np.asarray(values, dtype=float)]
    result = [np.asarray(rhs, dtype=float) for rhs in values]
    if any(rhs.ndim != 1 for rhs in result):
        raise ValueError("Each RHS must be a one-dimensional vector.")
    return result


def run_repeated_solve(
    matrices: Any,
    rhs_sequence: Any,
    *,
    warm_start: bool = False,
    reuse_preconditioner: bool = False,
    tol: float = 1e-8,
    max_iter: int = 1000,
    configuration: str | None = None,
) -> RepeatedSolveResult:
    """Run consecutive PCG solves and record reuse/setup behavior.

    The matrix may be fixed or supplied once per RHS. A cached Jacobi
    preconditioner is reused only when its fingerprint exactly matches the
    current matrix; otherwise it is rebuilt and that step is marked.
    """
    right_hand_sides = _expand_rhs(rhs_sequence)
    if not right_hand_sides:
        raise ValueError("rhs_sequence must contain at least one RHS.")
    matrix_sequence = _expand_matrices(matrices, len(right_hand_sides))
    label = configuration or (
        ("D" if reuse_preconditioner else "C") if warm_start
        else ("B" if reuse_preconditioner else "A")
    )
    records: list[StepSolveRecord] = []
    previous_solution: np.ndarray | None = None
    cached_preconditioner: JacobiPreconditioner | None = None
    validated_matrix_signature: str | None = None

    for step, (matrix, rhs) in enumerate(zip(matrix_sequence, right_hand_sides)):
        preparation_started = perf_counter()
        signature = matrix_fingerprint(matrix)
        check_spd = signature != validated_matrix_signature
        guess = previous_solution if warm_start else None
        matrix_changed = False
        rebuilt = False
        if reuse_preconditioner:
            if cached_preconditioner is None:
                cached_preconditioner = build_jacobi_preconditioner(matrix)
                rebuilt = True
            else:
                try:
                    cached_preconditioner.validate_for(matrix)
                except ValueError:
                    cached_preconditioner = build_jacobi_preconditioner(matrix)
                    matrix_changed = True
                    rebuilt = True
        else:
            cached_preconditioner = build_jacobi_preconditioner(matrix)
            rebuilt = True

        setup_time = cached_preconditioner.setup_time if rebuilt else 0.0
        preparation_time = perf_counter() - preparation_started
        result = solve_pcg(matrix, rhs, x0=guess, preconditioner=cached_preconditioner,
                           tol=tol, max_iter=max_iter, check_spd=check_spd)
        if rebuilt:
            result.preconditioner_setup_time = setup_time
        result.validation_time += max(0.0, preparation_time - setup_time)

        records.append(StepSolveRecord(step, result, rebuilt, matrix_changed))
        validated_matrix_signature = signature
        previous_solution = result.solution.copy()

    return RepeatedSolveResult(label, warm_start, reuse_preconditioner, records)


def compare_four_configurations(
    matrices: Any,
    rhs_sequence: Any,
    *,
    tol: float = 1e-8,
    max_iter: int = 1000,
) -> dict[str, RepeatedSolveResult]:
    """Run A-D from the brief: cold/rebuilt, cold/reused, warm/rebuilt, warm/reused."""
    right_hand_sides = _expand_rhs(rhs_sequence)
    matrix_sequence = _expand_matrices(matrices, len(right_hand_sides))
    # Preflight every distinct matrix once before timing any configuration so
    # BLAS/eigensolver first-call initialization does not skew configuration A.
    checked_signatures: set[str] = set()
    for matrix in matrix_sequence:
        signature = matrix_fingerprint(matrix)
        if signature not in checked_signatures:
            validate_spd(matrix)
            checked_signatures.add(signature)
    return {
        "A": run_repeated_solve(matrix_sequence, right_hand_sides, warm_start=False,
                                reuse_preconditioner=False, tol=tol, max_iter=max_iter),
        "B": run_repeated_solve(matrix_sequence, right_hand_sides, warm_start=False,
                                reuse_preconditioner=True, tol=tol, max_iter=max_iter),
        "C": run_repeated_solve(matrix_sequence, right_hand_sides, warm_start=True,
                                reuse_preconditioner=False, tol=tol, max_iter=max_iter),
        "D": run_repeated_solve(matrix_sequence, right_hand_sides, warm_start=True,
                                reuse_preconditioner=True, tol=tol, max_iter=max_iter),
    }
