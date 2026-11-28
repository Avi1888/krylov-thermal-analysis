"""Compare warm starts and safe Jacobi reuse on a synthetic thermal RHS sequence."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.preconditioners import matrix_fingerprint
from src.repeated_solve import run_repeated_solve
from .stage3_utils import (
    DEFAULT_HEAT_LEVELS,
    DEFAULT_REFERENCE_POWER_DENSITY,
    DEFAULT_SEED,
    DEFAULT_TOLERANCE,
    build_finite_volume_rhs_sequence,
    make_synthetic_markov_workload,
    summary_stats,
    thermal_parameter_config,
    write_csv,
    write_json,
)


CONFIGURATIONS = {
    "A": (False, False),
    "B": (False, True),
    "C": (True, False),
    "D": (True, True),
}


def _run_configuration(label, matrices, rhs_sequence, warm, reuse, tolerance, max_iter):
    result = run_repeated_solve(
        matrices,
        rhs_sequence,
        warm_start=warm,
        reuse_preconditioner=reuse,
        tol=tolerance,
        max_iter=max_iter,
        configuration=label,
    )
    for record, rhs in zip(result.records, rhs_sequence):
        threshold = tolerance * max(float(np.linalg.norm(rhs)), 1.0)
        if not record.result.converged or record.result.final_residual > threshold:
            raise RuntimeError(f"Configuration {label}, step {record.step} failed residual validation.")
    return result


def _aggregate(rows, key):
    return summary_stats([float(row[key]) for row in rows], key)


def _paired_effect(rows_by_rep, cold, warm, metric):
    differences = [rows_by_rep[rep][cold][metric] - rows_by_rep[rep][warm][metric]
                   for rep in sorted(rows_by_rep)]
    stats = summary_stats(differences, f"{metric}_cold_minus_warm")
    return stats


def _paired_reuse_effect(rows_by_rep, rebuild, reuse, metric):
    differences = [rows_by_rep[rep][rebuild][metric] - rows_by_rep[rep][reuse][metric]
                   for rep in sorted(rows_by_rep)]
    return summary_stats(differences, f"{metric}_rebuild_minus_reuse")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-side", type=int, default=20)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--tol", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument("--output-dir", type=Path, default=Path("results/synthetic"))
    args = parser.parse_args()
    if args.grid_side < 2 or args.steps < 2 or args.repetitions < 2:
        parser.error("--grid-side >= 2, --steps >= 2, and --repetitions >= 2 are required")

    _, states, observed_trace = make_synthetic_markov_workload(args.steps, args.seed)
    matrix, matrices, rhs_sequence, _, parameters = build_finite_volume_rhs_sequence(
        states,
        rows=args.grid_side,
        cols=args.grid_side,
        tolerance=args.tol,
        reference_power_density_w_m2=DEFAULT_REFERENCE_POWER_DENSITY,
    )
    signatures = {matrix_fingerprint(item) for item in matrices}
    if len(signatures) != 1:
        raise RuntimeError("The repeated-solve experiment expects a fixed finite-volume matrix.")

    max_iter = 10 * matrix.shape[0]
    rng = np.random.default_rng(args.seed + 1)
    # Warm each distinct code path before recording timing samples.
    for label, (warm, reuse) in CONFIGURATIONS.items():
        _run_configuration(label, matrices, rhs_sequence, warm, reuse, args.tol, max_iter)

    raw_rows = []
    repetition_rows = []
    grouped = {label: [] for label in CONFIGURATIONS}
    for repetition in range(args.repetitions):
        order = list(CONFIGURATIONS)
        rng.shuffle(order)
        by_config = {}
        for label in order:
            warm, reuse = CONFIGURATIONS[label]
            result = _run_configuration(
                label, matrices, rhs_sequence, warm, reuse, args.tol, max_iter
            )
            by_config[label] = result
            step_rows = []
            for step, (record, rhs, state) in enumerate(zip(result.records, rhs_sequence, states)):
                item = {
                    "repetition": repetition,
                    "configuration": label,
                    "warm_start": warm,
                    "reuse_preconditioner": reuse,
                    "step": step,
                    "workload_state": state,
                    "power_density_mean_w_m2": float(np.mean(
                        DEFAULT_REFERENCE_POWER_DENSITY * DEFAULT_HEAT_LEVELS[state]
                    )),
                    "initial_guess_policy": "previous_solution" if warm else "zero",
                    "preconditioner_policy": "reuse_if_matrix_unchanged" if reuse else "rebuild_each_step",
                    "matrix_changed": record.matrix_changed,
                    "preconditioner_rebuilt": record.preconditioner_rebuilt,
                    "iterations": record.result.iterations,
                    "converged": record.result.converged,
                    "final_true_residual": record.result.final_residual,
                    "residual_threshold": args.tol * max(float(np.linalg.norm(rhs)), 1.0),
                    "true_residual_ratio": record.result.final_residual / (
                        args.tol * max(float(np.linalg.norm(rhs)), 1.0)
                    ),
                    "solver_time_seconds": record.result.solve_time,
                    "preconditioner_setup_seconds": record.result.preconditioner_setup_time,
                    "validation_time_seconds": record.result.validation_time,
                    "step_total_time_seconds": record.result.solve_time
                    + record.result.preconditioner_setup_time + record.result.validation_time,
                }
                raw_rows.append(item)
                step_rows.append(item)

            run_summary = {
                "repetition": repetition,
                "configuration": label,
                "cumulative_iterations": result.total_iterations,
                "mean_iterations_per_solve": float(np.mean([row["iterations"] for row in step_rows])),
                "solver_time_seconds": result.total_solver_time,
                "preconditioner_setup_seconds": result.total_preconditioner_setup_time,
                "validation_time_seconds": result.total_validation_time,
                "total_time_seconds": result.total_time,
                "max_final_residual": max(row["final_true_residual"] for row in step_rows),
                "max_true_residual_ratio": max(row["true_residual_ratio"] for row in step_rows),
                "converged_steps": sum(row["converged"] for row in step_rows),
            }
            repetition_rows.append(run_summary)
            grouped[label].append((result, run_summary))

            # Reuse must occur only on the first step for the fixed matrix;
            # rebuild configurations must rebuild at every step.
            rebuilt = [record.preconditioner_rebuilt for record in result.records]
            expected = [True] * args.steps if not reuse else [True] + [False] * (args.steps - 1)
            if rebuilt != expected or any(record.matrix_changed for record in result.records):
                raise RuntimeError(f"Unexpected preconditioner lifecycle in configuration {label}.")

    summary_rows = []
    per_rep_lookup = {rep: {} for rep in range(args.repetitions)}
    for label, (warm, reuse) in CONFIGURATIONS.items():
        rep_rows = [row for row in repetition_rows if row["configuration"] == label]
        for row in rep_rows:
            per_rep_lookup[row["repetition"]][label] = row
        result_rows = [item for _, item in grouped[label]]
        summary = {
            "configuration": label,
            "warm_start": warm,
            "reuse_preconditioner": reuse,
            "steps": args.steps,
            "repetitions": args.repetitions,
            "convergence_rate": float(np.mean([row["converged_steps"] == args.steps for row in result_rows])),
        }
        summary.update(summary_stats([row["cumulative_iterations"] for row in result_rows],
                                     "cumulative_iterations_per_run"))
        summary.update(summary_stats([row["mean_iterations_per_solve"] for row in result_rows],
                                     "iterations_per_solve"))
        for metric in ("solver_time_seconds", "preconditioner_setup_seconds",
                       "validation_time_seconds", "total_time_seconds",
                       "max_final_residual", "max_true_residual_ratio"):
            summary.update(_aggregate(result_rows, metric))
        summary_rows.append(summary)

    warm_rows = []
    for name, cold, warm in (("rebuild: cold vs warm", "A", "C"),
                             ("reuse: cold vs warm", "B", "D")):
        row = {"comparison": name, "cold_configuration": cold, "warm_configuration": warm}
        for metric in ("cumulative_iterations", "solver_time_seconds", "total_time_seconds"):
            row.update(_paired_effect(per_rep_lookup, cold, warm, metric))
        warm_rows.append(row)

    reuse_rows = []
    for name, rebuild, reuse in (("zero start: rebuild vs reuse", "A", "B"),
                                 ("warm start: rebuild vs reuse", "C", "D")):
        row = {"comparison": name, "rebuild_configuration": rebuild, "reuse_configuration": reuse}
        for metric in ("preconditioner_setup_seconds", "solver_time_seconds", "total_time_seconds"):
            row.update(_paired_reuse_effect(per_rep_lookup, rebuild, reuse, metric))
        reuse_rows.append(row)

    output = args.output_dir
    write_csv(raw_rows, output / "raw" / "repeated_solve_steps.csv")
    write_csv(repetition_rows, output / "raw" / "repeated_solve_repetitions.csv")
    write_csv(summary_rows, output / "tables" / "four_configuration_summary.csv")
    write_csv(warm_rows, output / "tables" / "warm_start_comparison.csv")
    write_csv(reuse_rows, output / "tables" / "preconditioner_reuse_comparison.csv")
    write_csv([{"step": i, "synthetic_workload_state": state} for i, state in enumerate(states)],
              output / "raw" / "repeated_solve_workload_sequence.csv")
    write_json({
        "experiment": "synthetic repeated finite-volume systems",
        "synthetic": True,
        "dataset_used": False,
        "seed": args.seed,
        "order_randomization_seed": args.seed + 1,
        "grid_shape": [args.grid_side, args.grid_side],
        "steps": args.steps,
        "repetitions": args.repetitions,
        "tolerance": args.tol,
        "max_iterations": max_iter,
        "workload_sequence": states,
        "observed_synthetic_trace_used_to_estimate_markov_model": observed_trace,
        "reference_power_density_w_m2": DEFAULT_REFERENCE_POWER_DENSITY,
        "heat_levels": DEFAULT_HEAT_LEVELS,
        "spatial_weight_formula": (
            "0.15 + exp(-((x-0.32)^2+(y-0.42)^2)/0.035) "
            "+ 0.7*exp(-((x-0.72)^2+(y-0.68)^2)/0.055); normalized to mean one"
        ),
        "matrix_fingerprint": next(iter(signatures)),
        "matrix_constant_across_steps": True,
        "reference_rhs_path": "high-accuracy PCG baseline; fixed RHS sequence reused across A-D",
        "reference_baseline_tolerance": min(args.tol * 0.01, 1e-12),
        "thermal_model": thermal_parameter_config(
            parameters, rows=args.grid_side, cols=args.grid_side,
            die_width_m=0.01, die_height_m=0.01, steps=args.steps,
            initial_temperature_k=300.0,
        ),
        "configurations": {
            label: {"warm_start": options[0], "reuse_preconditioner": options[1]}
            for label, options in CONFIGURATIONS.items()
        },
    }, output / "configs" / "repeated_solve.json")
    _plot_configuration_summary(
        summary_rows, output / "figures" / "four_configuration_comparison.png", args.steps
    )

    print("Four-configuration summary (means/medians/stds across repetitions):")
    for row in summary_rows:
        print(row)
    print(f"Synthetic repeated-solve outputs saved below {output.resolve()}")


def _plot_configuration_summary(rows, path: Path, steps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [row["configuration"] for row in rows]
    metrics = [
        ("cumulative_iterations_per_run_mean", f"Mean cumulative iterations across {steps} solves"),
        ("solver_time_seconds_median", "Median solver time (s)"),
        ("preconditioner_setup_seconds_median", "Median setup time (s)"),
        ("total_time_seconds_median", "Median solver + setup + validation (s)"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 4))
    for axis, (key, title) in zip(axes, metrics):
        base = key.removesuffix("_mean").removesuffix("_median")
        error_key = f"{base}_std"
        values = [row[key] for row in rows]
        errors = [row.get(error_key, 0.0) for row in rows]
        axis.bar(labels, values, yerr=errors if "mean" in key else None, capsize=3)
        axis.set_title(title)
        axis.grid(True, axis="y", alpha=0.25)
    fig.suptitle(f"A-D repeated PCG runs on identical synthetic thermal systems ({steps} solves)")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
