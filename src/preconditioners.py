"""Preconditioners compatible with the project's CG methods."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import blake2b
from time import perf_counter
from typing import Any

import numpy as np
from scipy import sparse


def matrix_fingerprint(A: Any) -> str:
    """Return a content fingerprint used to guard preconditioner reuse."""
    digest = blake2b(digest_size=20)
    if sparse.issparse(A):
        matrix = A.tocsr().astype(float, copy=True)
        matrix.sum_duplicates()
        matrix.eliminate_zeros()
        matrix.sort_indices()
        digest.update(np.asarray(matrix.shape, dtype=np.int64).tobytes())
        digest.update(matrix.indptr.tobytes())
        digest.update(matrix.indices.tobytes())
        digest.update(matrix.data.tobytes())
    else:
        matrix = np.ascontiguousarray(np.asarray(A, dtype=float))
        digest.update(np.asarray(matrix.shape, dtype=np.int64).tobytes())
        digest.update(matrix.tobytes())
    return digest.hexdigest()


@dataclass
class JacobiPreconditioner:
    """Diagonal inverse action ``M^-1 r`` for ``M = diag(A)``."""

    inverse_diagonal: np.ndarray
    matrix_signature: str
    setup_time: float

    def apply(self, residual: Any) -> np.ndarray:
        vector = np.asarray(residual, dtype=float)
        if vector.shape != self.inverse_diagonal.shape:
            raise ValueError("Residual size does not match the Jacobi preconditioner.")
        return self.inverse_diagonal * vector

    def validate_for(self, A: Any) -> None:
        """Reject reuse unless the complete coefficient matrix is unchanged."""
        if matrix_fingerprint(A) != self.matrix_signature:
            raise ValueError("Jacobi preconditioner was built for a different matrix.")


def build_jacobi_preconditioner(A: Any) -> JacobiPreconditioner:
    """Build and time a Jacobi preconditioner without forming a matrix inverse."""
    started = perf_counter()
    if sparse.issparse(A):
        matrix = A.tocsr()
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("A must be square to build a Jacobi preconditioner.")
        diagonal = np.asarray(matrix.diagonal(), dtype=float)
    else:
        matrix = np.asarray(A, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("A must be square to build a Jacobi preconditioner.")
        diagonal = np.diag(matrix).copy()
    if not np.all(np.isfinite(diagonal)) or np.any(diagonal <= 0):
        raise ValueError("Jacobi PCG requires finite, strictly positive diagonal entries.")
    inverse_diagonal = 1.0 / diagonal
    signature = matrix_fingerprint(matrix)
    return JacobiPreconditioner(inverse_diagonal, signature, perf_counter() - started)
