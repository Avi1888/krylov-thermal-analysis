"""Small, transparent implementations of CG and Jacobi-PCG.

The routines accept dense NumPy arrays or SciPy sparse matrices. They verify
that the input is square, finite, and symmetric; CG's positive-curvature
condition is checked during every iteration. A successful symmetry check
alone does not prove positive definiteness.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
from scipy import sparse

from .preconditioners import JacobiPreconditioner, build_jacobi_preconditioner


@dataclass
class SolverResult:
    """Outcome and diagnostics for one iterative solve."""

    solution: np.ndarray
    iterations: int
    converged: bool
    residual_history: list[float]
    final_residual: float
    solve_time: float
    preconditioner_setup_time: float = 0.0
    message: str = ""
    validation_time: float = 0.0


def _validate_system(A: Any, b: Any, tol: float) -> tuple[Any, np.ndarray]:
    if sparse.issparse(A):
        matrix = A.tocsr().astype(float)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
            raise ValueError("A must be a square matrix.")
        if not np.all(np.isfinite(matrix.data)):
            raise ValueError("A contains non-finite values.")
        difference = matrix - matrix.T
        asymmetry = float(sparse.linalg.norm(difference))
        scale = max(float(sparse.linalg.norm(matrix)), 1.0)
    else:
        matrix = np.asarray(A, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
            raise ValueError("A must be a square matrix.")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("A contains non-finite values.")
        asymmetry = float(np.linalg.norm(matrix - matrix.T, ord="fro"))
        scale = max(float(np.linalg.norm(matrix, ord="fro")), 1.0)

    rhs = np.asarray(b, dtype=float)
    if rhs.ndim != 1 or rhs.size != matrix.shape[0]:
        raise ValueError("b must be a one-dimensional vector matching A.")
    if not np.all(np.isfinite(rhs)):
        raise ValueError("b contains non-finite values.")
    if asymmetry > 1e-10 * scale:
        raise ValueError("CG and PCG require a symmetric matrix; A failed the symmetry check.")
    return matrix, rhs


def validate_spd(A: Any, symmetry_tol: float = 1e-10) -> None:
    """Raise ``ValueError`` unless A is numerically symmetric positive definite.

    Matrices up to 1024 rows use a Cholesky factorization. Larger sparse
    matrices use the smallest algebraic eigenvalue from ``eigsh``. For large
    matrices, failure of the iterative eigenvalue check is treated as
    inconclusive and reported as an error rather than silently accepted.
    """
    if not np.isfinite(symmetry_tol) or symmetry_tol <= 0:
        raise ValueError("symmetry_tol must be a finite positive number.")
    if sparse.issparse(A):
        matrix = A.tocsr().astype(float)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
            raise ValueError("A must be a non-empty square matrix.")
        if not np.all(np.isfinite(matrix.data)):
            raise ValueError("A contains non-finite values.")
        delta = matrix - matrix.T
        asymmetry = float(sparse.linalg.norm(delta))
        scale = max(float(sparse.linalg.norm(matrix)), 1.0)
        size = matrix.shape[0]
        dense_matrix = matrix.toarray() if size <= 1024 else None
    else:
        matrix = np.asarray(A, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
            raise ValueError("A must be a non-empty square matrix.")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("A contains non-finite values.")
        asymmetry = float(np.linalg.norm(matrix - matrix.T, ord="fro"))
        scale = max(float(np.linalg.norm(matrix, ord="fro")), 1.0)
        size = matrix.shape[0]
        dense_matrix = matrix
    if asymmetry > symmetry_tol * scale:
        raise ValueError("A is not symmetric within the requested tolerance.")
    try:
        if dense_matrix is not None:
            np.linalg.cholesky(dense_matrix)
        else:
            smallest = float(sparse.linalg.eigsh(matrix, k=1, which="SA",
                                                  return_eigenvectors=False)[0])
            if not np.isfinite(smallest) or smallest <= 0:
                raise ValueError("A is not positive definite.")
    except np.linalg.LinAlgError as exc:
        raise ValueError("A is not positive definite (Cholesky factorization failed).") from exc
    except (RuntimeError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("A is not positive definite"):
            raise
        raise ValueError("Could not establish positive definiteness for A.") from exc


def _validate_options(tol: float, max_iter: int) -> None:
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be a finite positive number.")
    if isinstance(max_iter, bool) or int(max_iter) != max_iter or max_iter < 0:
        raise ValueError("max_iter must be a non-negative integer.")


def _initial_solution(size: int, x0: Any | None) -> np.ndarray:
    if x0 is None:
        return np.zeros(size, dtype=float)
    guess = np.asarray(x0, dtype=float)
    if guess.shape != (size,) or not np.all(np.isfinite(guess)):
        raise ValueError("x0 must be a finite vector matching the system size.")
    return guess.copy()


def _matvec(A: Any, vector: np.ndarray) -> np.ndarray:
    return np.asarray(A @ vector, dtype=float).reshape(-1)


def _true_residual(A: Any, b: np.ndarray, solution: np.ndarray) -> np.ndarray:
    """Recompute ``b - A @ x`` from the current solution, not CG recurrence."""
    residual = b - _matvec(A, solution)
    if not np.all(np.isfinite(residual)):
        raise FloatingPointError("True residual became non-finite.")
    return residual


def solve_cg(
    A: Any,
    b: Any,
    x0: Any | None = None,
    tol: float = 1e-8,
    max_iter: int = 1000,
    check_spd: bool = True,
) -> SolverResult:
    """Solve ``A x = b`` with the Conjugate Gradient method.

    Convergence uses ``||r|| <= tol * max(||b||, 1)``. A non-positive
    ``p.T @ A @ p`` raises ``ValueError`` because it violates CG's SPD
    requirement (or indicates severe numerical breakdown).
    """
    _validate_options(tol, max_iter)
    validation_started = perf_counter()
    matrix, rhs = _validate_system(A, b, tol)
    if check_spd:
        validate_spd(matrix, symmetry_tol=1e-10)
    solution = _initial_solution(rhs.size, x0)
    validation_time = perf_counter() - validation_started
    solve_started = perf_counter()
    residual = _true_residual(matrix, rhs, solution)
    rhs_scale = max(float(np.linalg.norm(rhs)), 1.0)
    threshold = tol * rhs_scale
    residual_norm = float(np.linalg.norm(residual))
    history = [residual_norm]
    if residual_norm <= threshold:
        return SolverResult(solution, 0, True, history, residual_norm,
                            perf_counter() - solve_started,
                            message="Initial guess satisfies tolerance.",
                            validation_time=validation_time)

    direction = residual.copy()
    residual_sq = float(residual @ residual)
    for iteration in range(1, int(max_iter) + 1):
        product = _matvec(matrix, direction)
        curvature = float(direction @ product)
        if not np.isfinite(curvature) or curvature <= 0:
            raise ValueError("CG encountered non-positive curvature; A may not be positive definite.")
        alpha = residual_sq / curvature
        solution += alpha * direction
        residual -= alpha * product
        residual_norm = float(np.linalg.norm(residual))
        history.append(residual_norm)
        if not np.isfinite(residual_norm):
            raise FloatingPointError("CG residual became non-finite.")
        if residual_norm <= threshold:
            # Recursive residuals can drift from b - A*x in floating point.
            # Only report convergence after an independent residual check.
            residual = _true_residual(matrix, rhs, solution)
            residual_norm = float(np.linalg.norm(residual))
            history[-1] = residual_norm
            if residual_norm <= threshold:
                return SolverResult(solution, iteration, True, history, residual_norm,
                                    perf_counter() - solve_started, message="Converged.",
                                    validation_time=validation_time)
            # Residual replacement restarts CG if the recurrence was optimistic.
            direction = residual.copy()
            residual_sq = float(residual @ residual)
            continue
        next_residual_sq = float(residual @ residual)
        if residual_sq == 0 or not np.isfinite(next_residual_sq):
            raise FloatingPointError("CG encountered numerical breakdown.")
        direction = residual + (next_residual_sq / residual_sq) * direction
        residual_sq = next_residual_sq

    residual = _true_residual(matrix, rhs, solution)
    residual_norm = float(np.linalg.norm(residual))
    history[-1] = residual_norm
    converged = residual_norm <= threshold
    return SolverResult(solution, int(max_iter), converged, history, residual_norm,
                        perf_counter() - solve_started,
                        message="Converged after true-residual check." if converged else "Maximum iterations reached.",
                        validation_time=validation_time)


def solve_pcg(
    A: Any,
    b: Any,
    x0: Any | None = None,
    preconditioner: JacobiPreconditioner | None = None,
    tol: float = 1e-8,
    max_iter: int = 1000,
    check_spd: bool = True,
) -> SolverResult:
    """Solve ``A x = b`` using PCG and a Jacobi preconditioner.

    If ``preconditioner`` is omitted, Jacobi setup is performed and timed.
    A supplied preconditioner is checked against the current matrix before
    use; its original build time is not charged again for this solve.
    """
    _validate_options(tol, max_iter)
    validation_started = perf_counter()
    matrix, rhs = _validate_system(A, b, tol)
    if check_spd:
        validate_spd(matrix, symmetry_tol=1e-10)
    if preconditioner is None:
        preconditioner = build_jacobi_preconditioner(matrix)
        setup_time = preconditioner.setup_time
    else:
        preconditioner.validate_for(matrix)
        setup_time = 0.0

    solution = _initial_solution(rhs.size, x0)
    validation_time = max(0.0, perf_counter() - validation_started - setup_time)
    solve_started = perf_counter()
    residual = _true_residual(matrix, rhs, solution)
    rhs_scale = max(float(np.linalg.norm(rhs)), 1.0)
    threshold = tol * rhs_scale
    residual_norm = float(np.linalg.norm(residual))
    history = [residual_norm]
    if residual_norm <= threshold:
        return SolverResult(solution, 0, True, history, residual_norm,
                            perf_counter() - solve_started, setup_time,
                            "Initial guess satisfies tolerance.", validation_time)

    z = preconditioner.apply(residual)
    residual_preconditioned = float(residual @ z)
    if not np.isfinite(residual_preconditioned) or residual_preconditioned <= 0:
        raise ValueError("Preconditioner must be positive definite for PCG.")
    direction = z.copy()

    for iteration in range(1, int(max_iter) + 1):
        product = _matvec(matrix, direction)
        curvature = float(direction @ product)
        if not np.isfinite(curvature) or curvature <= 0:
            raise ValueError("PCG encountered non-positive curvature; A may not be positive definite.")
        alpha = residual_preconditioned / curvature
        solution += alpha * direction
        residual -= alpha * product
        residual_norm = float(np.linalg.norm(residual))
        history.append(residual_norm)
        if not np.isfinite(residual_norm):
            raise FloatingPointError("PCG residual became non-finite.")
        if residual_norm <= threshold:
            # Re-evaluate with the returned x before accepting convergence.
            residual = _true_residual(matrix, rhs, solution)
            residual_norm = float(np.linalg.norm(residual))
            history[-1] = residual_norm
            if residual_norm <= threshold:
                return SolverResult(solution, iteration, True, history, residual_norm,
                                    perf_counter() - solve_started, setup_time, "Converged.",
                                    validation_time)
            # Restart PCG from the recomputed residual after residual replacement.
            z = preconditioner.apply(residual)
            residual_preconditioned = float(residual @ z)
            if not np.isfinite(residual_preconditioned) or residual_preconditioned <= 0:
                raise ValueError("PCG encountered non-positive preconditioned residual norm.")
            direction = z.copy()
            continue

        z = preconditioner.apply(residual)
        next_residual_preconditioned = float(residual @ z)
        if not np.isfinite(next_residual_preconditioned) or next_residual_preconditioned <= 0:
            raise ValueError("PCG encountered non-positive preconditioned residual norm.")
        beta = next_residual_preconditioned / residual_preconditioned
        direction = z + beta * direction
        residual_preconditioned = next_residual_preconditioned

    residual = _true_residual(matrix, rhs, solution)
    residual_norm = float(np.linalg.norm(residual))
    history[-1] = residual_norm
    converged = residual_norm <= threshold
    return SolverResult(solution, int(max_iter), converged, history, residual_norm,
                        perf_counter() - solve_started, setup_time,
                        "Converged after true-residual check." if converged else "Maximum iterations reached.",
                        validation_time)
