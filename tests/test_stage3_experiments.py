"""Input and matrix checks for the reproducible Stage 3 experiments."""
import numpy as np
from scipy.sparse.linalg import eigsh

from experiments.run_solver_comparison import build_nonconstant_diagonal_spd_matrix
from experiments.stage3_utils import (
    build_finite_volume_rhs_sequence,
    make_synthetic_markov_workload,
    make_power_schedule,
)
from src.preconditioners import matrix_fingerprint


def test_solver_comparison_matrix_is_spd_and_has_nonconstant_diagonal():
    matrix = build_nonconstant_diagonal_spd_matrix(6)
    diagonal = matrix.diagonal()
    assert np.std(diagonal) > 0.1
    assert np.allclose((matrix - matrix.T).toarray(), 0.0)
    assert eigsh(matrix, k=1, which="SA", return_eigenvectors=False)[0] > 0.0


def test_synthetic_markov_workload_is_seed_reproducible_and_separate():
    model_a, states_a, trace_a = make_synthetic_markov_workload(12, 9876)
    model_b, states_b, trace_b = make_synthetic_markov_workload(12, 9876)
    assert states_a == states_b
    assert trace_a == trace_b
    np.testing.assert_array_equal(model_a.transition_matrix, model_b.transition_matrix)
    assert len(states_a) == 12
    assert set(states_a) <= {"Low", "Medium", "High"}


def test_repeated_solve_inputs_use_identical_finite_volume_matrix_and_schedule():
    states = ["Low", "High", "Medium"]
    schedule = make_power_schedule(states, (4, 5))
    matrix, matrices, rhs, baseline, _ = build_finite_volume_rhs_sequence(
        states, rows=4, cols=5, tolerance=1e-8
    )
    assert len(schedule) == len(rhs) == len(matrices) == 3
    assert baseline[0].shape == baseline[-1].shape == (4, 5)
    assert len({matrix_fingerprint(item) for item in matrices}) == 1
    assert matrix_fingerprint(matrix) == matrix_fingerprint(matrices[0])
    assert all(source.shape == (4, 5) for source in schedule)
    assert all(vector.shape == (20,) for vector in rhs)
