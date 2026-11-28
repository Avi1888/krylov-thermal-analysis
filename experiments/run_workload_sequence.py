"""Build a Markov workload model from one explicitly selected metric."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.benchmark import (
    plot_transition_matrix,
    plot_workload_sequence,
    save_summary_csv,
)
from src.markov_model import estimate_markov_model, generate_workload_sequence
from src.workload_preprocessing import (
    categorize_workload,
    load_thermal_dataset,
    map_workload_to_heat_intensity,
    workload_scores,
)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path,
        default=project_root / "data" / "raw" / "CPU_R7_4800U.pkl",
    )
    parser.add_argument(
        "--metric-index", type=int, required=True,
        help="Feature index selected from the dataset's unlabeled input metrics.",
    )
    parser.add_argument(
        "--assume-chronological-sample-order", action="store_true",
        help="Confirm that the outer sample dimension is chronological before estimating transitions.",
    )
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--output-dir", type=Path,
        default=project_root / "results" / "real_dataset",
        help="Output directory for this explicitly selected real-dataset experiment.",
    )
    args = parser.parse_args()
    if not args.assume_chronological_sample_order:
        parser.error(
            "The dataset documentation does not establish outer sample chronology; "
            "confirm it with --assume-chronological-sample-order before estimating transitions."
        )
    if args.steps < 0:
        parser.error("--steps must be non-negative")

    dataset = load_thermal_dataset(args.dataset)
    scores = workload_scores(dataset["input"], metric_index=args.metric_index)
    observed_states, thresholds = categorize_workload(scores)
    model = estimate_markov_model(observed_states)
    generated_states = generate_workload_sequence(
        model, args.steps, rng=np.random.default_rng(args.seed)
    )

    observed_rows = [
        {
            "sample_index": index,
            "selected_metric_mean": float(score),
            "workload_state": str(state),
            "relative_heat_intensity": map_workload_to_heat_intensity(str(state)),
        }
        for index, (score, state) in enumerate(zip(scores, observed_states))
    ]
    generated_rows = [
        {"step": index, "generated_state": state,
         "relative_heat_intensity": map_workload_to_heat_intensity(state)}
        for index, state in enumerate(generated_states)
    ]
    transition_rows = [
        {"current_state": state,
         **{f"next_{next_state.lower()}": float(model.transition_matrix[row, column])
            for column, next_state in enumerate(model.states)}}
        for row, state in enumerate(model.states)
    ]

    table_dir = args.output_dir / "tables"
    figure_dir = args.output_dir / "figures"
    save_summary_csv(observed_rows, table_dir / "observed_workload_states.csv")
    if generated_rows:
        save_summary_csv(generated_rows, table_dir / "generated_workload_states.csv")
    save_summary_csv(transition_rows, table_dir / "markov_transition_matrix.csv")
    plot_transition_matrix(model, figure_dir / "markov_transition_matrix.png")
    plot_workload_sequence(observed_states, figure_dir / "observed_workload_sequence.png")
    if generated_states:
        plot_workload_sequence(generated_states, figure_dir / "generated_workload_sequence.png")

    print(f"Selected metric index: {args.metric_index} (feature name unavailable)")
    print(f"Tertile thresholds: {thresholds}")
    print("Observed transition counts:\n", model.transition_counts)
    print("Estimated transition probabilities:\n", model.transition_matrix)
    if model.unobserved_outgoing_states:
        print("States assigned explicit self-loops:", model.unobserved_outgoing_states)
    print(f"Results written below {args.output_dir}")


if __name__ == "__main__":
    main()
