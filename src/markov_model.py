"""First-order Markov workload transitions and sequence generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass
class MarkovModel:
    """Estimated transition probabilities over a fixed ordered state set."""

    states: tuple[str, ...]
    transition_matrix: np.ndarray
    transition_counts: np.ndarray
    initial_probabilities: np.ndarray
    unobserved_outgoing_states: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        n = len(self.states)
        matrix = np.asarray(self.transition_matrix, dtype=float)
        counts = np.asarray(self.transition_counts, dtype=np.int64)
        initial = np.asarray(self.initial_probabilities, dtype=float)
        if len(set(self.states)) != n or n == 0:
            raise ValueError("states must be a non-empty sequence of unique labels.")
        if matrix.shape != (n, n) or counts.shape != (n, n) or initial.shape != (n,):
            raise ValueError("Transition and initial probability dimensions must match states.")
        if not np.all(np.isfinite(matrix)) or np.any(matrix < 0) or np.any(matrix > 1):
            raise ValueError("Transition probabilities must be finite and in [0, 1].")
        if not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-12):
            raise ValueError("Every transition row must sum to one.")
        if np.any(counts < 0) or not np.all(np.isfinite(initial)) or np.any(initial < 0):
            raise ValueError("Counts and initial probabilities must be non-negative.")
        if not np.isclose(initial.sum(), 1.0):
            raise ValueError("Initial probabilities must sum to one.")
        self.transition_matrix = matrix
        self.transition_counts = counts
        self.initial_probabilities = initial

    def index(self, state: str) -> int:
        try:
            return self.states.index(state)
        except ValueError as exc:
            raise ValueError(f"Unknown workload state: {state!r}.") from exc


def estimate_markov_model(
    sequence: Sequence[str],
    states: Sequence[str] = ("Low", "Medium", "High"),
    unobserved_outgoing: str = "self_loop",
) -> MarkovModel:
    """Estimate transition probabilities from adjacent observed states.

    States with no observed outgoing transitions either become explicit
    self-loops (the default) or cause an error when ``unobserved_outgoing`` is
    ``raise``. No pseudo-count smoothing is applied.
    """
    ordered_states = tuple(states)
    if not ordered_states or len(set(ordered_states)) != len(ordered_states):
        raise ValueError("states must contain unique labels.")
    if unobserved_outgoing not in ("self_loop", "raise"):
        raise ValueError("unobserved_outgoing must be 'self_loop' or 'raise'.")
    observations = tuple(sequence)
    if not observations:
        raise ValueError("sequence must contain at least one state.")
    lookup = {state: index for index, state in enumerate(ordered_states)}
    unknown = sorted(set(observations) - set(ordered_states))
    if unknown:
        raise ValueError(f"Sequence contains states outside the model: {unknown}.")

    counts = np.zeros((len(ordered_states), len(ordered_states)), dtype=np.int64)
    initial_counts = np.zeros(len(ordered_states), dtype=np.int64)
    initial_counts[lookup[observations[0]]] += 1
    for current, following in zip(observations[:-1], observations[1:]):
        counts[lookup[current], lookup[following]] += 1

    probabilities = np.zeros_like(counts, dtype=float)
    unobserved: list[str] = []
    for index, state in enumerate(ordered_states):
        total = int(counts[index].sum())
        if total:
            probabilities[index] = counts[index] / total
        elif unobserved_outgoing == "self_loop":
            probabilities[index, index] = 1.0
            unobserved.append(state)
        else:
            raise ValueError(f"State {state!r} has no observed outgoing transitions.")
    initial = initial_counts.astype(float) / initial_counts.sum()
    return MarkovModel(ordered_states, probabilities, counts, initial, tuple(unobserved))


def generate_workload_sequence(
    markov_model: MarkovModel,
    n_steps: int,
    initial_state: str | None = None,
    rng: np.random.Generator | None = None,
) -> list[str]:
    """Generate ``n_steps`` states, optionally conditioned on a first state."""
    if isinstance(n_steps, bool) or int(n_steps) != n_steps or n_steps < 0:
        raise ValueError("n_steps must be a non-negative integer.")
    if n_steps == 0:
        return []
    random = np.random.default_rng() if rng is None else rng
    if initial_state is None:
        current_index = int(random.choice(len(markov_model.states), p=markov_model.initial_probabilities))
    else:
        current_index = markov_model.index(initial_state)
    result = [markov_model.states[current_index]]
    for _ in range(1, int(n_steps)):
        current_index = int(random.choice(
            len(markov_model.states), p=markov_model.transition_matrix[current_index]
        ))
        result.append(markov_model.states[current_index])
    return result
