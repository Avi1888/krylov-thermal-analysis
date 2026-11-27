"""Compare the four warm-start and Jacobi-reuse configurations."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy import sparse

from src.benchmark import (
    plot_repeated_solve_comparison,
    repeated_solve_summary,
    save_summary_csv,
)
from src.repeated_solve import compare_four_configurations


def build_spd_grid_matrix(side: int) -> sparse.csr_matrix:
    if side < 2:
        raise ValueError("grid side must be at least 2.")
    ones = np.ones(side)
    one_dimensional = sparse.diags(
        [-ones[:-1], 2.0 * ones, -ones[:-1]], [-1, 0, 1], format="csr"
    )
    identity = sparse.eye(side, format="csr")
    return sparse.kron(identity, one_dimensional) + sparse.kron(one_dimensional, identity)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-side", type=int, default=12)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be at least 1")

    matrix = build_spd_grid_matrix(args.grid_side)
    rng = np.random.default_rng(args.seed)
    base = rng.normal(size=matrix.shape[0])
    rhs_sequence = []
    for step in range(args.steps):
        # Smoothly varying right-hand sides make the warm-start experiment
        # reproducible without implying that this synthetic workload is real.
        exact_solution = base + (step / max(args.steps - 1, 1)) * 0.1
        rhs_sequence.append(matrix @ exact_solution)

    results = compare_four_configurations(
        matrix, rhs_sequence, tol=1e-10, max_iter=10 * matrix.shape[0]
    )
    rows = repeated_solve_summary(results)
    table_path = args.output_dir / "tables" / "reuse_comparison.csv"
    figure_path = args.output_dir / "figures" / "reuse_comparison.png"
    save_summary_csv(rows, table_path)
    plot_repeated_solve_comparison(results, figure_path)
    for row in rows:
        print(row)
    print(f"Saved table to {table_path}")
    print(f"Saved comparison plot to {figure_path}")


if __name__ == "__main__":
    main()
