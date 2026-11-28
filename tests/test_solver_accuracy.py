import numpy as np
from scipy.sparse.linalg import cg, spsolve
from src.grid_and_boundary import create_chip_grid, ChipParameters
from src.sparse_assembly import assemble_system


def test_cg_residual_is_small():
    A, b = assemble_system(create_chip_grid(8, 8, power_density=60000), ChipParameters())
    x, info = cg(A, b, rtol=1e-10, atol=0.0, maxiter=2000)
    assert info == 0
    assert np.linalg.norm(A @ x - b) / np.linalg.norm(b) < 1e-8


def test_uniform_equilibrium_matches_analytic_solution():
    """At equilibrium with uniform power, T = T_sink + p_surface/h_eff."""
    power = 40000.0
    params = ChipParameters()
    expected = params.sink_temperature + power / params.effective_vertical_h
    grid = create_chip_grid(5, 5, initial_temperature=expected, power_density=power)
    A, b = assemble_system(grid, params)
    computed = spsolve(A, b)
    assert np.allclose(computed, expected, rtol=1e-10, atol=1e-10)
