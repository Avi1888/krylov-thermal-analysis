"""Unit tests for estimated Markov transition probabilities."""

import unittest

import numpy as np

from src.markov_model import estimate_markov_model, generate_workload_sequence


class MarkovProbabilityTests(unittest.TestCase):
    def test_transition_rows_are_valid_probabilities(self):
        model = estimate_markov_model(
            ["Low", "Low", "Medium", "High", "Medium", "Low"]
        )
        self.assertTrue(np.all(model.transition_matrix >= 0.0))
        self.assertTrue(np.all(model.transition_matrix <= 1.0))
        np.testing.assert_allclose(model.transition_matrix.sum(axis=1), 1.0)
        np.testing.assert_array_equal(
            model.transition_counts,
            np.array([[1, 1, 0], [1, 0, 1], [0, 1, 0]]),
        )

    def test_unobserved_outgoing_state_is_explicit_self_loop(self):
        model = estimate_markov_model(["Low", "Medium", "Low"])
        self.assertEqual(model.unobserved_outgoing_states, ("High",))
        np.testing.assert_array_equal(model.transition_matrix[2], [0.0, 0.0, 1.0])

    def test_unobserved_outgoing_state_can_be_rejected(self):
        with self.assertRaisesRegex(ValueError, "no observed outgoing transitions"):
            estimate_markov_model(["Low", "Medium"], unobserved_outgoing="raise")

    def test_unknown_states_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "outside the model"):
            estimate_markov_model(["Low", "Busy"])

    def test_generated_sequence_has_requested_length_and_valid_states(self):
        model = estimate_markov_model(["Low", "Medium", "High", "High"])
        generated = generate_workload_sequence(
            model, 50, initial_state="Medium", rng=np.random.default_rng(12)
        )
        self.assertEqual(len(generated), 50)
        self.assertEqual(generated[0], "Medium")
        self.assertTrue(set(generated).issubset(set(model.states)))

    def test_zero_length_generation_is_empty(self):
        model = estimate_markov_model(["Low"])
        self.assertEqual(generate_workload_sequence(model, 0), [])


if __name__ == "__main__":
    unittest.main()
