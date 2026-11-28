"""True-residual accuracy and convergence checks for the Member 2 solvers."""
import numpy as np
import pytest
from scipy import sparse

from src.krylov_solvers import solve_cg, solve_pcg


@pytest.mark.parametrize("solver", [solve_cg, solve_pcg], ids=["cg", "pcg"])
def test_final_residual_matches_direct_residual_of_returned_solution(solver):
    matrix = sparse.csr_matrix([
        [4.0, 1.0, 0.0],
        [1.0, 3.0, 0.5],
        [0.0, 0.5, 2.0],
    ])
    expected_solution = np.array([0.25, -1.5, 2.0])
    rhs = matrix @ expected_solution

    result = solver(matrix, rhs, tol=1e-12, max_iter=20)
    direct_residual = np.linalg.norm(rhs - matrix @ result.solution)

    assert result.converged
    assert result.final_residual == pytest.approx(direct_residual, rel=1e-14, abs=1e-16)
    assert result.residual_history[-1] == pytest.approx(direct_residual, rel=1e-14, abs=1e-16)


@pytest.mark.parametrize("solver", [solve_cg, solve_pcg], ids=["cg", "pcg"])
@pytest.mark.parametrize(
    "residual_fraction, expected_converged",
    [(0.999, True), (1.001, False)],
    ids=["just-within-tolerance", "just-outside-tolerance"],
)
def test_convergence_status_uses_true_residual_near_tolerance(
    solver, residual_fraction, expected_converged
):
    matrix = sparse.csr_matrix([[1.0]])
    rhs = np.array([1.0])
    tolerance = 1e-8
    initial_residual = residual_fraction * tolerance
    x0 = np.array([1.0 - initial_residual])

    result = solver(matrix, rhs, x0=x0, tol=tolerance, max_iter=0)
    direct_residual = np.linalg.norm(rhs - matrix @ result.solution)
    threshold = tolerance * max(np.linalg.norm(rhs), 1.0)

    assert result.final_residual == pytest.approx(direct_residual, rel=0.0, abs=1e-16)
    assert result.converged == (direct_residual <= threshold)
    assert result.converged == expected_converged


@pytest.mark.parametrize("solver", [solve_cg, solve_pcg], ids=["cg", "pcg"])
def test_max_iteration_result_reports_true_residual(solver):
    matrix = np.array([[3.0, 1.0], [1.0, 2.0]])
    rhs = np.array([1.0, 4.0])
    result = solver(matrix, rhs, tol=1e-15, max_iter=1)
    direct_residual = np.linalg.norm(rhs - matrix @ result.solution)

    assert not result.converged
    assert result.final_residual == pytest.approx(direct_residual, rel=1e-14, abs=1e-16)
