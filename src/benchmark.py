"""Solver comparisons, result summaries, and optional plots."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from .krylov_solvers import SolverResult, solve_cg, solve_pcg
from .repeated_solve import RepeatedSolveResult, compare_four_configurations


def benchmark_cg_vs_pcg(
    A: Any,
    b: Any,
    *,
    x0: Any | None = None,
    tol: float = 1e-8,
    max_iter: int = 1000,
    reference_solution: Any | None = None,
) -> dict[str, Any]:
    """Run CG and Jacobi-PCG and return comparable diagnostics."""
    cg = solve_cg(A, b, x0=x0, tol=tol, max_iter=max_iter)
    pcg = solve_pcg(A, b, x0=x0, tol=tol, max_iter=max_iter)
    reference = None if reference_solution is None else np.asarray(reference_solution, dtype=float)
    rows: dict[str, Any] = {}
    for name, result in (("CG", cg), ("PCG", pcg)):
        row: dict[str, Any] = {
            "iterations": result.iterations,
            "converged": result.converged,
            "final_residual": result.final_residual,
            "solver_time": result.solve_time,
            "preconditioner_setup_time": result.preconditioner_setup_time,
            "total_time": result.solve_time + result.preconditioner_setup_time,
        }
        if reference is not None:
            if reference.shape != result.solution.shape:
                raise ValueError("reference_solution must match the solution shape.")
            row["solution_error_norm"] = float(np.linalg.norm(result.solution - reference))
        rows[name] = row
    return {"results": {"CG": cg, "PCG": pcg}, "summary": rows}


def benchmark_repeated_configurations(
    matrices: Any,
    rhs_sequence: Any,
    *,
    tol: float = 1e-8,
    max_iter: int = 1000,
) -> dict[str, RepeatedSolveResult]:
    """Convenience wrapper for all four repeated-solve configurations."""
    return compare_four_configurations(matrices, rhs_sequence, tol=tol, max_iter=max_iter)


def repeated_solve_summary(results: dict[str, RepeatedSolveResult]) -> list[dict[str, Any]]:
    """Return one flat row per repeated-solve configuration."""
    summary = []
    for name, result in results.items():
        summary.append({
            "configuration": name,
            "warm_start": result.warm_start,
            "reuse_preconditioner": result.reuse_preconditioner,
            "steps": len(result.records),
            "converged_steps": sum(record.result.converged for record in result.records),
            "total_iterations": result.total_iterations,
            "solver_time_seconds": result.total_solver_time,
            "preconditioner_setup_seconds": result.total_preconditioner_setup_time,
            "total_time_seconds": result.total_time,
            "final_residual": result.final_residual,
        })
    return summary


def save_summary_csv(rows: list[dict[str, Any]], path: str | Path) -> Path:
    """Write benchmark summary rows to CSV, creating parent directories."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("rows must not be empty.")
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return destination


def plot_residual_histories(
    results: dict[str, SolverResult],
    path: str | Path | None = None,
):
    """Plot residual norm by iteration for CG/PCG; return the Matplotlib figure."""
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots()
    for label, result in results.items():
        axis.semilogy(range(len(result.residual_history)), result.residual_history, label=label)
    axis.set(xlabel="Iteration", ylabel="Residual norm", title="Krylov solver convergence")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    if path is not None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=160)
    return figure


def plot_transition_matrix(model: Any, path: str | Path | None = None):
    """Plot a Markov model's transition matrix and return the figure."""
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots()
    image = axis.imshow(model.transition_matrix, vmin=0, vmax=1, cmap="Blues")
    axis.set_xticks(range(len(model.states)), labels=model.states)
    axis.set_yticks(range(len(model.states)), labels=model.states)
    axis.set(xlabel="Next state", ylabel="Current state", title="Workload transition probabilities")
    for row in range(len(model.states)):
        for column in range(len(model.states)):
            axis.text(column, row, f"{model.transition_matrix[row, column]:.2f}",
                      ha="center", va="center")
    figure.colorbar(image, ax=axis, label="Probability")
    figure.tight_layout()
    if path is not None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=160)
    return figure


def plot_workload_sequence(sequence: Any, path: str | Path | None = None):
    """Plot categorical workload states in their supplied order."""
    import matplotlib.pyplot as plt

    states = ("Low", "Medium", "High")
    lookup = {state: index for index, state in enumerate(states)}
    values = list(sequence)
    if any(value not in lookup for value in values):
        raise ValueError("Sequence must contain only Low, Medium, and High states.")
    figure, axis = plt.subplots()
    axis.step(range(len(values)), [lookup[value] for value in values], where="mid")
    axis.set_yticks(range(3), labels=states)
    axis.set(xlabel="Observation index", ylabel="Workload state", title="Workload-state sequence")
    axis.grid(True, axis="x", alpha=0.25)
    figure.tight_layout()
    if path is not None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=160)
    return figure
