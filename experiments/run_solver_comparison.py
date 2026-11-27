"""Compare the implemented CG and Jacobi-PCG methods on a Dirichlet grid."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy import sparse

from src.benchmark import benchmark_cg_vs_pcg, plot_residual_histories, save_summary_csv


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
    parser.add_argument("--grid-side", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    matrix = build_spd_grid_matrix(args.grid_side)
    rng = np.random.default_rng(args.seed)
    exact_solution = rng.normal(size=matrix.shape[0])
    rhs = matrix @ exact_solution
    benchmark = benchmark_cg_vs_pcg(
        matrix, rhs, tol=1e-10, max_iter=10 * matrix.shape[0],
        reference_solution=exact_solution,
    )
    rows = [{"solver": name, **summary}
            for name, summary in benchmark["summary"].items()]
    table_path = args.output_dir / "tables" / "cg_vs_pcg.csv"
    figure_path = args.output_dir / "figures" / "cg_vs_pcg_residuals.png"
    save_summary_csv(rows, table_path)
    plot_residual_histories(benchmark["results"], figure_path)
    for row in rows:
        print(row)
    print(f"Saved table to {table_path}")
    print(f"Saved residual plot to {figure_path}")


if __name__ == "__main__":
    main()
