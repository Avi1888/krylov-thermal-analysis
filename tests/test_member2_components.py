"""Regression tests for member 2 solver and data-processing components."""

import unittest

import numpy as np
from scipy import sparse

from src.krylov_solvers import solve_cg, solve_pcg, validate_spd
from src.markov_model import estimate_markov_model
from src.preconditioners import build_jacobi_preconditioner
from src.repeated_solve import compare_four_configurations, run_repeated_solve
from src.workload_preprocessing import (
    categorize_workload,
    inspect_thermal_dataset,
    make_spatial_heat_source,
    workload_scores,
)


class SolverTests(unittest.TestCase):
    def setUp(self):
        self.matrix = np.array([[4.0, 1.0], [1.0, 3.0]])
        self.expected = np.array([1.5, -2.0])
        self.rhs = self.matrix @ self.expected

    def test_cg_recovers_solution(self):
        result = solve_cg(self.matrix, self.rhs, tol=1e-12)
        self.assertTrue(result.converged)
        np.testing.assert_allclose(result.solution, self.expected, atol=1e-11)
        self.assertEqual(len(result.residual_history), result.iterations + 1)
        self.assertGreaterEqual(result.validation_time, 0.0)

    def test_sparse_pcg_recovers_solution(self):
        result = solve_pcg(sparse.csr_matrix(self.matrix), self.rhs, tol=1e-12)
        self.assertTrue(result.converged)
        np.testing.assert_allclose(result.solution, self.expected, atol=1e-11)
        self.assertGreaterEqual(result.preconditioner_setup_time, 0.0)

    def test_indefinite_and_asymmetric_matrices_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "positive definite"):
            validate_spd(np.array([[1.0, 2.0], [2.0, 1.0]]))
        with self.assertRaisesRegex(ValueError, "symmetric"):
            solve_cg(np.array([[2.0, 1.0], [0.0, 2.0]]), np.ones(2))

    def test_jacobi_preconditioner_rejects_changed_matrix(self):
        preconditioner = build_jacobi_preconditioner(self.matrix)
        with self.assertRaisesRegex(ValueError, "different matrix"):
            preconditioner.validate_for(2 * self.matrix)


class RepeatedSolveTests(unittest.TestCase):
    def test_four_configurations_and_setup_accounting(self):
        matrix = sparse.csr_matrix([[4.0, 1.0], [1.0, 3.0]])
        expected = [np.array([1.0, -1.0]), np.array([1.1, -0.9]), np.array([1.2, -0.8])]
        rhs_sequence = [matrix @ solution for solution in expected]
        results = compare_four_configurations(matrix, rhs_sequence, tol=1e-11)
        self.assertEqual(set(results), {"A", "B", "C", "D"})
        for result in results.values():
            self.assertTrue(all(record.result.converged for record in result.records))
        self.assertEqual(results["A"].total_preconditioner_setup_time > 0, True)
        self.assertAlmostEqual(
            results["A"].total_time,
            results["A"].total_solver_time
            + results["A"].total_preconditioner_setup_time
            + results["A"].total_validation_time,
        )
        self.assertEqual(sum(record.preconditioner_rebuilt for record in results["A"].records), 3)
        self.assertEqual(sum(record.preconditioner_rebuilt for record in results["B"].records), 1)

    def test_reuse_rebuilds_when_matrix_changes(self):
        first = np.array([[4.0, 1.0], [1.0, 3.0]])
        second = 2.0 * first
        rhs_values = [first @ np.ones(2), second @ np.ones(2)]
        result = run_repeated_solve(
            [first, second], rhs_values, reuse_preconditioner=True, tol=1e-11
        )
        self.assertFalse(result.records[0].matrix_changed)
        self.assertTrue(result.records[1].matrix_changed)
        self.assertTrue(result.records[1].preconditioner_rebuilt)


class WorkloadPreprocessingTests(unittest.TestCase):
    def test_dataset_report_and_explicit_feature_selection(self):
        inputs = np.arange(24, dtype=float).reshape(3, 4, 2)
        outputs = np.ones((3, 5, 6), dtype=float)
        report = inspect_thermal_dataset({"input": inputs, "output": outputs})
        self.assertEqual(report["input_layout"]["sequence_length"], 4)
        self.assertEqual(report["input_layout"]["feature_count"], 2)
        np.testing.assert_allclose(workload_scores(inputs, metric_index=1), [4, 12, 20])

    def test_quantile_states_and_configurable_heat_source(self):
        states, thresholds = categorize_workload([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertEqual(set(states), {"Low", "Medium", "High"})
        self.assertLess(thresholds[0], thresholds[1])
        weights = np.array([[1.0, 0.0], [1.0, 2.0]])
        source = make_spatial_heat_source(
            "High", (2, 2), heat_levels={"Low": 0.1, "Medium": 0.4, "High": 0.9},
            spatial_weights=weights,
        )
        self.assertAlmostEqual(float(source.sum()), 0.9)
        self.assertEqual(source[0, 1], 0.0)

    def test_missing_dataset_alignment_is_reported(self):
        data = {"input": np.ones((2, 3)), "output": np.ones((1, 2, 2))}
        self.assertFalse(inspect_thermal_dataset(data)["sample_counts_match"])


if __name__ == "__main__":
    unittest.main()
