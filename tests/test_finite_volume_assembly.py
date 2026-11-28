"""Mathematical checks for the cell-centred finite-volume thermal system."""
import numpy as np
import pytest
from scipy.sparse.linalg import eigsh

from src.grid_and_boundary import (
    ChipMaterial,
    ChipParameters,
    create_chip_grid,
    make_power_map,
)
from src.sparse_assembly import assemble_system
from src.thermal_model import run_thermal_simulation
from src.visualization import plot_chip_heatmap


def test_grid_spacing_and_cell_centres_cover_die():
    grid = create_chip_grid(3, 4, die_width=8.0, die_height=6.0)
    assert grid.dx == 2.0
    assert grid.dy == 2.0
    np.testing.assert_allclose(grid.x_centers, [1.0, 3.0, 5.0, 7.0])
    np.testing.assert_allclose(grid.y_centers, [1.0, 3.0, 5.0])
    assert grid.temperature.shape == (3, 4)
    assert np.all((grid.x_centers > 0) & (grid.x_centers < 8.0))
    assert np.all((grid.y_centers > 0) & (grid.y_centers < 6.0))


def test_heatmap_extent_represents_full_die():
    fig = plot_chip_heatmap(np.full((3, 4), 300.0), die_width=8.0, die_height=6.0)
    try:
        assert tuple(fig.axes[0].images[0].get_extent()) == (0.0, 8.0, 0.0, 6.0)
        assert fig.axes[0].get_xlabel() == "Chip x position (m)"
        assert fig.axes[0].get_ylabel() == "Chip y position (m)"
    finally:
        import matplotlib.pyplot as plt
        plt.close(fig)


def test_interior_edge_and_corner_rows_match_finite_volume_stencil():
    grid = create_chip_grid(3, 4, die_width=8.0, die_height=6.0)
    params = ChipParameters(
        material=ChipMaterial(density=2.0, specific_heat=3.0, conductivity=5.0),
        die_thickness=0.5,
        dt=0.25,
        sink_temperature=10.0,
        effective_vertical_h=4.0,
    )
    A, b = assemble_system(grid, parameters=params)
    matrix = A.toarray()
    capacity = 2.0 * 3.0 / 0.25
    vertical = 4.0 / 0.5
    wx = 5.0 / grid.dx**2
    wy = 5.0 / grid.dy**2
    # Row-major indices: top-left corner 0, top edge 1, interior 5.
    for index, neighbours in ((0, 2), (1, 3), (5, 4)):
        assert matrix[index, index] == pytest.approx(capacity + vertical + neighbours * wx)
        assert np.count_nonzero(matrix[index]) == neighbours + 1
    assert matrix[0, 1] == pytest.approx(-wx)
    assert matrix[0, 4] == pytest.approx(-wy)
    assert b[0] == pytest.approx(capacity * 300.0 + vertical * 10.0)


def test_zero_flux_boundary_has_no_outside_neighbour_conductance():
    grid = create_chip_grid(3, 4, die_width=8.0, die_height=6.0)
    params = ChipParameters(
        material=ChipMaterial(density=2.0, specific_heat=3.0, conductivity=5.0),
        die_thickness=0.5,
        dt=0.25,
        effective_vertical_h=0.0,
    )
    A, _ = assemble_system(grid, parameters=params)
    capacity = 2.0 * 3.0 / 0.25
    # With h=0, subtracting the capacity leaves the discrete lateral
    # Laplacian. Its row sums vanish, including at edges and corners: the
    # omitted exterior faces contribute zero normal heat flux.
    lateral = A.toarray() - capacity * np.eye(grid.rows * grid.cols)
    np.testing.assert_allclose(lateral.sum(axis=1), 0.0, atol=1e-12)
    assert np.count_nonzero(A.toarray()[0]) == 3  # corner: itself + 2 neighbours
    assert np.count_nonzero(A.toarray()[1]) == 4  # edge: itself + 3 neighbours
    assert np.count_nonzero(A.toarray()[5]) == 5  # interior: itself + 4 neighbours


def test_matrix_is_symmetric_positive_definite():
    grid = create_chip_grid(5, 6, die_width=0.012, die_height=0.01)
    A, _ = assemble_system(grid, parameters=ChipParameters())
    assert np.allclose((A - A.T).toarray(), 0.0, atol=1e-12)
    assert eigsh(A, k=1, which="SA", return_eigenvectors=False)[0] > 0.0


@pytest.mark.parametrize("shape", [(2, 2), (4, 5), (8, 9)])
def test_uniform_source_matches_analytical_transient_for_grid_refinement(shape):
    rows, cols = shape
    rho, cp, thickness, dt, power = 1200.0, 800.0, 0.001, 0.02, 24000.0
    steps = 4
    params = ChipParameters(
        material=ChipMaterial(density=rho, specific_heat=cp, conductivity=7.0),
        die_thickness=thickness,
        dt=dt,
        sink_temperature=300.0,
        effective_vertical_h=0.0,
    )
    grid = create_chip_grid(
        rows, cols, die_width=0.02, die_height=0.01,
        initial_temperature=300.0, power_density=power,
    )
    history = run_thermal_simulation(grid, params, steps)
    # Uniform T has zero lateral gradient. With insulated edges and h=0,
    # the exact solution is T(t)=T0 + p*t/(rho*cp*thickness).
    expected = 300.0 + np.arange(steps + 1) * dt * power / (rho * cp * thickness)
    np.testing.assert_allclose(
        history, np.broadcast_to(expected[:, None, None], history.shape),
        rtol=1e-12, atol=1e-12,
    )


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_material_parameters_are_rejected(value):
    with pytest.raises(ValueError):
        ChipMaterial(density=value).validate()


@pytest.mark.parametrize("field", ["dt", "die_thickness", "sink_temperature", "effective_vertical_h"])
def test_nonfinite_chip_parameters_are_rejected(field):
    params = ChipParameters(**{field: np.nan})
    with pytest.raises(ValueError):
        params.validate()


@pytest.mark.parametrize("bad_power", [np.nan, np.inf, -1.0])
def test_nonfinite_or_negative_power_map_values_are_rejected(bad_power):
    with pytest.raises(ValueError):
        create_chip_grid(2, 2, power_density=bad_power)


def test_make_power_map_rejects_nonfinite_background():
    with pytest.raises(ValueError):
        make_power_map(2, 2, background=np.inf)
