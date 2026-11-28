import numpy as np
from scipy.sparse.linalg import eigsh
from src.grid_and_boundary import create_chip_grid, ChipParameters
from src.sparse_assembly import assemble_system


def test_matrix_shape_symmetry_and_positive_definiteness():
    grid = create_chip_grid(5, 6, power_density=50000)
    A, b = assemble_system(grid, parameters=ChipParameters())
    assert A.shape == (30, 30)
    assert b.shape == (30,)
    assert np.allclose((A - A.T).toarray(), 0.0)
    assert eigsh(A, k=1, which="SA", return_eigenvectors=False)[0] > 0


def test_documented_member2_interface():
    grid = create_chip_grid(4, 4)
    heat = np.full((4, 4), 20000.0)
    A, b = assemble_system(grid, heat, ChipParameters())
    assert A.shape == (16, 16)
    assert np.all(b > 0)
