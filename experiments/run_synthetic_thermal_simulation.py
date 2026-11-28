"""Run a reproducible, synthetic Markov-workload finite-volume simulation."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.benchmark import plot_transition_matrix
from src.grid_and_boundary import ChipParameters, create_chip_grid
from src.preconditioners import matrix_fingerprint
from src.thermal_integration import ThermalSolverAdapter
from src.thermal_model import run_thermal_simulation
from src.visualization import plot_chip_heatmap, plot_temperature_trends
from .stage3_utils import (
    DEFAULT_HEAT_LEVELS,
    DEFAULT_REFERENCE_POWER_DENSITY,
    DEFAULT_SEED,
    DEFAULT_TOLERANCE,
    make_power_schedule,
    make_synthetic_markov_workload,
    thermal_parameter_config,
    write_csv,
    write_json,
)


def plot_workload_over_time(states, dt: float, path: Path) -> None:
    """Plot synthetic categorical states over physical simulation time."""
    lookup = {"Low": 0, "Medium": 1, "High": 2}
    if not states or any(state not in lookup for state in states):
        raise ValueError("states must be a non-empty Low/Medium/High sequence.")
    time_edges = np.arange(len(states) + 1) * dt
    values = [lookup[state] for state in states] + [lookup[states[-1]]]
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.step(time_edges, values, where="post")
    ax.set_yticks([0, 1, 2], labels=["Low", "Medium", "High"])
    ax.set(xlabel="Simulation time (s)", ylabel="Synthetic workload state",
           title="Markov-generated synthetic workload")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=20)
    parser.add_argument("--cols", type=int, default=20)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--tol", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument("--method", choices=("cg", "pcg"), default="pcg")
    parser.add_argument("--output-dir", type=Path, default=Path("results/synthetic"))
    args = parser.parse_args()
    if args.rows < 2 or args.cols < 2 or args.steps < 1:
        parser.error("--rows and --cols must be >= 2 and --steps must be >= 1")

    model, states, observed_trace = make_synthetic_markov_workload(args.steps, args.seed)
    parameters = ChipParameters()
    grid = create_chip_grid(args.rows, args.cols, die_width=0.01, die_height=0.01,
                            initial_temperature=parameters.sink_temperature)
    schedule = make_power_schedule(states, (args.rows, args.cols))
    adapter = ThermalSolverAdapter(
        method=args.method,
        warm_start=True,
        reuse_preconditioner=(args.method == "pcg"),
        tol=args.tol,
        max_iter=10 * args.rows * args.cols,
    )

    history = run_thermal_simulation(
        grid, parameters, args.steps, solver=adapter, power_schedule=schedule
    )
    if not np.all(np.isfinite(history)):
        raise RuntimeError("Thermal simulation produced non-finite temperatures.")
    if len(adapter.records) != args.steps or adapter.summary["converged_steps"] != args.steps:
        raise RuntimeError("Not every requested thermal step converged.")

    # Reassemble each system from the recorded old temperature and source, then
    # validate the residual independently against the stored solver solution.
    from src.sparse_assembly import assemble_system
    diagnostic_rows = []
    signatures = []
    for step, (state, source, record) in enumerate(zip(states, schedule, adapter.records)):
        grid.temperature = history[step].copy()
        matrix, rhs = assemble_system(grid, parameters=parameters, power_density=source)
        signatures.append(matrix_fingerprint(matrix))
        true_residual = float(np.linalg.norm(rhs - matrix @ record.result.solution))
        threshold = args.tol * max(float(np.linalg.norm(rhs)), 1.0)
        if true_residual > threshold or not record.result.converged:
            raise RuntimeError(f"Thermal solver residual check failed at step {step}.")
        if not np.allclose(record.result.solution.reshape(args.rows, args.cols), history[step + 1]):
            raise RuntimeError(f"Solver callback output does not match temperature history at step {step}.")
        diagnostic_rows.append({
            "step": step,
            "time_seconds": (step + 1) * parameters.dt,
            "workload_state": state,
            "power_density_mean_w_m2": float(source.mean()),
            "power_density_max_w_m2": float(source.max()),
            "solver_method": args.method.upper(),
            "iterations": record.result.iterations,
            "converged": record.result.converged,
            "final_true_residual": true_residual,
            "residual_threshold": threshold,
            "solver_time_seconds": record.result.solve_time,
            "preconditioner_setup_seconds": record.result.preconditioner_setup_time,
            "validation_time_seconds": record.result.validation_time,
            "step_total_time_seconds": record.result.solve_time
            + record.result.preconditioner_setup_time + record.result.validation_time,
            "preconditioner_rebuilt": record.preconditioner_rebuilt,
            "matrix_changed": record.matrix_changed,
            "maximum_temperature_k": float(history[step + 1].max()),
            "average_temperature_k": float(history[step + 1].mean()),
        })

    if len(set(signatures)) != 1:
        raise RuntimeError("The model matrix changed across steps; cached-preconditioner assumption failed.")
    if args.method == "pcg":
        expected_rebuilt = [True] + [False] * (args.steps - 1)
        if [record.preconditioner_rebuilt for record in adapter.records] != expected_rebuilt:
            raise RuntimeError("PCG preconditioner reuse diagnostics were unexpected.")

    output = args.output_dir
    raw_dir, table_dir, figure_dir = output / "raw", output / "tables", output / "figures"
    raw_dir.mkdir(parents=True, exist_ok=True)
    np.save(raw_dir / "synthetic_temperature_history_k.npy", history)
    write_csv(diagnostic_rows, raw_dir / "synthetic_thermal_steps.csv")
    map_rows = []
    x_centres = grid.x_centers
    y_centres = grid.y_centers
    for step, temperature_map in enumerate(history):
        time_seconds = step * parameters.dt
        for row in range(args.rows):
            for column in range(args.cols):
                temperature_k = float(temperature_map[row, column])
                map_rows.append({
                    "step": step,
                    "time_seconds": time_seconds,
                    "row": row,
                    "column": column,
                    "x_cell_centre_m": float(x_centres[column]),
                    "y_cell_centre_m": float(y_centres[row]),
                    "temperature_k": temperature_k,
                    "temperature_c": temperature_k - 273.15,
                })
    write_csv(map_rows, raw_dir / "synthetic_temperature_maps.csv")
    write_csv([
        {"step": i, "time_seconds": i * parameters.dt, "state": state,
         "relative_heat_intensity": DEFAULT_HEAT_LEVELS[state],
         "mean_power_density_w_m2": float(schedule[i].mean())}
        for i, state in enumerate(states)
    ], table_dir / "synthetic_workload_sequence.csv")
    write_csv([
        {"current_state": state,
         **{f"next_{next_state.lower()}": float(model.transition_matrix[i, j])
            for j, next_state in enumerate(model.states)},
         **{f"count_{next_state.lower()}": int(model.transition_counts[i, j])
            for j, next_state in enumerate(model.states)}}
        for i, state in enumerate(model.states)
    ], table_dir / "synthetic_markov_transition_matrix.csv")
    thermal_summary = [{
        "steps": args.steps,
        "solver": args.method.upper(),
        "converged_steps": adapter.summary["converged_steps"],
        "total_iterations": adapter.summary["total_iterations"],
        "solver_time_seconds": adapter.summary["solver_time_seconds"],
        "preconditioner_setup_seconds": adapter.summary["preconditioner_setup_seconds"],
        "validation_time_seconds": adapter.summary["validation_seconds"],
        "total_time_seconds": adapter.summary["total_time_seconds"],
        "maximum_true_residual": max(row["final_true_residual"] for row in diagnostic_rows),
        "maximum_temperature_k": float(history[-1].max()),
        "average_temperature_k": float(history[-1].mean()),
        "minimum_temperature_k": float(history[-1].min()),
    }]
    write_csv(thermal_summary, table_dir / "synthetic_thermal_summary.csv")
    write_csv([{"index": i, "observed_synthetic_state": state}
               for i, state in enumerate(observed_trace)], raw_dir / "markov_training_trace.csv")

    write_json({
        "experiment": "synthetic Markov workload coupled to Member 1 finite-volume model",
        "synthetic": True,
        "dataset_used": False,
        "ryzen_behavior_claimed": False,
        "seed": args.seed,
        "rows": args.rows,
        "cols": args.cols,
        "steps": args.steps,
        "simulation_duration_seconds": args.steps * parameters.dt,
        "solver": args.method.upper(),
        "warm_start": True,
        "reuse_preconditioner": args.method == "pcg",
        "tolerance": args.tol,
        "maximum_iterations": adapter.max_iter,
        "thermal_model": thermal_parameter_config(
            parameters, rows=args.rows, cols=args.cols,
            die_width_m=0.01, die_height_m=0.01, steps=args.steps,
            initial_temperature_k=parameters.sink_temperature,
        ),
        "reference_power_density_w_m2": DEFAULT_REFERENCE_POWER_DENSITY,
        "heat_levels": DEFAULT_HEAT_LEVELS,
        "mean_source_power_density_by_state_w_m2": {
            state: DEFAULT_REFERENCE_POWER_DENSITY * intensity
            for state, intensity in DEFAULT_HEAT_LEVELS.items()
        },
        "spatial_weight_formula": (
            "0.15 + exp(-((x-0.32)^2+(y-0.42)^2)/0.035) "
            "+ 0.7*exp(-((x-0.72)^2+(y-0.68)^2)/0.055); normalized to mean one"
        ),
        "synthetic_training_state_trace": observed_trace,
        "generated_state_sequence": states,
        "matrix_fingerprint": signatures[0],
        "matrix_constant_across_steps": True,
        "temperature_history_units": "K",
        "heat_source_units": "W/m^2",
        "temperature_figure_units": "degrees Celsius; T_C = T_K - 273.15",
    }, output / "configs" / "synthetic_thermal_simulation.json")

    transition_figure = plot_transition_matrix(model, figure_dir / "synthetic_markov_transition_matrix.png")
    plt.close(transition_figure)
    plot_workload_over_time(states, parameters.dt, figure_dir / "synthetic_workload_vs_time.png")
    trend_figure = plot_temperature_trends(history, parameters.dt, figure_dir / "synthetic_temperature_trends.png")
    plt.close(trend_figure)
    selected_steps = sorted(set([0, args.steps // 3, (2 * args.steps) // 3, args.steps]))
    for step in selected_steps:
        heatmap_figure = plot_chip_heatmap(
            history[step], figure_dir / f"synthetic_temperature_step_{step:04d}.png",
            f"Synthetic thermal simulation: step {step}", die_width=0.01, die_height=0.01,
        )
        plt.close(heatmap_figure)

    print("Synthetic workload thermal summary:")
    print(thermal_summary[0])
    print(f"Generated states: {', '.join(states)}")
    print(f"Synthetic outputs saved below {output.resolve()}")


if __name__ == "__main__":
    main()
