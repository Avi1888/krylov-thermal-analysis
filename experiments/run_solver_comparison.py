"""Repeated CG versus Jacobi-PCG benchmark on a synthetic heterogeneous SPD system."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse

from src.krylov_solvers import solve_cg, solve_pcg, validate_spd
from .stage3_utils import DEFAULT_SEED, DEFAULT_TOLERANCE, summary_stats, write_csv, write_json


def build_nonconstant_diagonal_spd_matrix(side: int) -> sparse.csr_matrix:
    """Build a scaled 2D Dirichlet diffusion matrix with spatially varying scale.

    The congruence scaling preserves symmetry and positive definiteness while
    producing a nonconstant diagonal for a meaningful Jacobi-PCG comparison.
    This is a synthetic solver test problem, not a calibrated chip model.
    """
    if side < 2:
        raise ValueError("grid side must be at least 2.")
    ones = np.ones(side)
    one_dimensional = sparse.diags(
        [-ones[:-1], 2.0 * ones, -ones[:-1]], [-1, 0, 1], format="csr"
    )
    identity = sparse.eye(side, format="csr")
    diffusion = (
        sparse.kron(identity, one_dimensional, format="csr")
        + sparse.kron(one_dimensional, identity, format="csr")
        + 0.05 * sparse.eye(side * side, format="csr")
    )
    coordinate = np.linspace(0.0, 1.0, side)
    xx, yy = np.meshgrid(coordinate, coordinate)
    scaling = (0.5 + 3.5 * (0.65 * xx + 0.35 * yy)).ravel()
    diagonal_scale = sparse.diags(scaling, format="csr")
    matrix = diagonal_scale @ diffusion @ diagonal_scale
    return matrix.tocsr()


def _write_figures(rows: list[dict], representative_results: dict, output: Path) -> None:
    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for solver, result in representative_results.items():
        ax.semilogy(result.residual_history, label=solver)
    ax.set(xlabel="Iteration", ylabel="Residual norm",
           title="CG and Jacobi-PCG convergence (representative run)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "cg_pcg_convergence.png", dpi=170)
    plt.close(fig)

    summaries = {row["solver"]: row for row in rows}
    labels = ["CG", "PCG"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    axes[0].bar(labels, [summaries[x]["iterations_mean"] for x in labels],
                yerr=[summaries[x]["iterations_std"] for x in labels], capsize=4)
    axes[0].set(title="Mean iterations", ylabel="Iterations")
    axes[1].bar(labels, [summaries[x]["solver_time_median"] for x in labels])
    axes[1].set(title="Median solver time", ylabel="Seconds")
    axes[2].bar(labels, [summaries[x]["preconditioner_setup_time_median"] for x in labels])
    axes[2].set(title="Median Jacobi setup", ylabel="Seconds")
    for axis in axes:
        axis.grid(True, axis="y", alpha=0.25)
    fig.suptitle("Synthetic heterogeneous SPD system; timing distributions in CSV")
    fig.tight_layout()
    fig.savefig(figure_dir / "cg_pcg_comparison.png", dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-side", type=int, default=20)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--tol", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument("--output-dir", type=Path, default=Path("results/synthetic"))
    args = parser.parse_args()
    if args.repetitions < 2:
        parser.error("--repetitions must be at least 2 to report timing spread")

    matrix = build_nonconstant_diagonal_spd_matrix(args.grid_side)
    validate_spd(matrix)
    diagonal = matrix.diagonal()
    rng = np.random.default_rng(args.seed)
    exact_solution = rng.normal(size=matrix.shape[0])
    rhs = np.asarray(matrix @ exact_solution)
    x0 = np.zeros_like(rhs)
    max_iter = 10 * matrix.shape[0]
    threshold = args.tol * max(float(np.linalg.norm(rhs)), 1.0)

    # Warm the two code paths before recording timing samples.
    solve_cg(matrix, rhs, x0=x0, tol=args.tol, max_iter=max_iter)
    solve_pcg(matrix, rhs, x0=x0, tol=args.tol, max_iter=max_iter)

    raw_rows = []
    residual_rows = []
    results_by_solver = {"CG": [], "PCG": []}
    for repetition in range(args.repetitions):
        order = ["CG", "PCG"]
        if repetition % 2:
            order.reverse()
        for name in order:
            solve = solve_cg if name == "CG" else solve_pcg
            result = solve(matrix, rhs, x0=x0, tol=args.tol, max_iter=max_iter)
            actual_residual = float(np.linalg.norm(rhs - matrix @ result.solution))
            if not result.converged or actual_residual > threshold:
                raise RuntimeError(f"{name} did not meet the requested true-residual tolerance.")
            results_by_solver[name].append(result)
            residual_rows.extend({
                "repetition": repetition,
                "solver": name,
                "iteration": iteration,
                "residual_norm": residual,
            } for iteration, residual in enumerate(result.residual_history))
            raw_rows.append({
                "repetition": repetition,
                "solver": name,
                "matrix_dimension": matrix.shape[0],
                "diagonal_min": float(diagonal.min()),
                "diagonal_max": float(diagonal.max()),
                "diagonal_std": float(diagonal.std()),
                "initial_guess": "zero",
                "tolerance": args.tol,
                "max_iterations": max_iter,
                "iterations": result.iterations,
                "converged": result.converged,
                "final_true_residual": actual_residual,
                "residual_threshold": threshold,
                "true_residual_ratio": actual_residual / threshold,
                "solver_time_seconds": result.solve_time,
                "preconditioner_setup_seconds": result.preconditioner_setup_time,
                "validation_time_seconds": result.validation_time,
                "total_time_seconds": result.solve_time + result.preconditioner_setup_time + result.validation_time,
                "solution_error_norm": float(np.linalg.norm(result.solution - exact_solution)),
            })

    summary_rows = []
    representative = {}
    representative_repetitions = {}
    for name, results in results_by_solver.items():
        raw = [row for row in raw_rows if row["solver"] == name]
        row = {
            "solver": name,
            "repetitions": args.repetitions,
            "convergence_rate": float(np.mean([item.converged for item in results])),
            "final_residual_max": max(item.final_residual for item in results),
            "max_true_residual_ratio": max(
                row["true_residual_ratio"] for row in raw if row["solver"] == name
            ),
            "preconditioner_setup_time_median": float(np.median([item.preconditioner_setup_time for item in results])),
        }
        row.update(summary_stats([item.iterations for item in results], "iterations"))
        row.update(summary_stats([item.final_residual for item in results], "true_residual"))
        row.update(summary_stats([item.solve_time for item in results], "solver_time"))
        row.update(summary_stats([item.preconditioner_setup_time for item in results], "setup_time"))
        row.update(summary_stats([item.solve_time + item.preconditioner_setup_time + item.validation_time
                                  for item in results], "total_time"))
        summary_rows.append(row)
        median_iterations = np.median([item.iterations for item in results])
        representative_index = min(
            range(len(results)), key=lambda index: abs(results[index].iterations - median_iterations)
        )
        representative[name] = results[representative_index]
        representative_repetitions[name] = representative_index
    for row in raw_rows:
        row["representative_convergence_plot"] = (
            row["repetition"] == representative_repetitions[row["solver"]]
        )
    for row in residual_rows:
        row["representative_convergence_plot"] = (
            row["repetition"] == representative_repetitions[row["solver"]]
        )

    output = args.output_dir
    write_csv(raw_rows, output / "raw" / "cg_pcg_runs.csv")
    write_csv(residual_rows, output / "raw" / "cg_pcg_residual_histories.csv")
    write_csv(summary_rows, output / "tables" / "cg_pcg_summary.csv")
    write_json({
        "experiment": "synthetic CG vs Jacobi-PCG",
        "synthetic": True,
        "dataset_used": False,
        "seed": args.seed,
        "grid_side": args.grid_side,
        "grid_is_thermal_grid": False,
        "matrix_dimension": matrix.shape[0],
        "matrix_shape": list(matrix.shape),
        "matrix_nonzero_count": int(matrix.nnz),
        "matrix_density_fraction": float(matrix.nnz / (matrix.shape[0] * matrix.shape[1])),
        "matrix": (
            "A = S K S, where K = kron(I,T) + kron(T,I) + 0.05 I; "
            "T is the 1D Dirichlet tridiagonal matrix with diagonal 2 and off-diagonal -1; "
            "S=diag(s), s=0.5 + 3.5*(0.65*x + 0.35*y), with x,y coordinates on linspace(0,1,grid_side)"
        ),
        "diagonal_min": float(diagonal.min()),
        "diagonal_max": float(diagonal.max()),
        "rhs_generation": "seeded standard-normal exact solution x_star; b=A@x_star",
        "initial_guess": "zero vector for both solvers",
        "execution_order": "CG then PCG on even repetitions; PCG then CG on odd repetitions",
        "repetitions": args.repetitions,
        "tolerance": args.tol,
        "max_iterations": max_iter,
        "same_matrix_rhs_initial_guess_and_tolerance": True,
    }, output / "configs" / "cg_pcg.json")
    _write_figures(summary_rows, representative, output)
    for row in summary_rows:
        print(row)
    print(f"Synthetic CG/PCG outputs saved below {output.resolve()}")


if __name__ == "__main__":
    main()
