"""Backward-Euler cell-centred finite-volume assembly for a 2D silicon die."""
import numpy as np
from scipy.sparse import lil_matrix


def assemble_system(grid, heat_source=None, parameters=None,
                    previous_temperature=None, power_density=None):
    """Return CSR matrix A and vector b for A*T_new=b.

    Supported calls:
      assemble_system(grid, parameters)
      assemble_system(grid, heat_source, parameters)

    heat_source/power_density is surface power density in W/m^2.
    The grid entries are cell-centred, equal-area control volumes with
    ``dx = die_width / cols`` and ``dy = die_height / rows``. The equation is
    the backward-Euler discretization of

      rho*cp*dT/dt = k*laplacian(T) - (h_eff/t_die)*(T-T_sink) + p/t_die.

    Lateral faces on the die boundary have zero normal flux, so no conductance
    is added to cells outside the die. The vertical cooling path is represented
    by ``effective_vertical_h``. Rows and flattened vectors use C/row-major order.
    """
    # Backward compatibility with the earlier two-argument interface.
    if parameters is None and heat_source is not None and hasattr(heat_source, "material"):
        parameters, heat_source = heat_source, None
    if parameters is None:
        raise TypeError("Pass ChipParameters as the third argument or second argument.")

    grid.validate()
    parameters.validate()
    rows, cols = grid.rows, grid.cols
    material = parameters.material
    # Divide the integrated finite-volume equation by each equal cell volume.
    capacity_rate = material.density * material.specific_heat / parameters.dt
    k = material.conductivity
    vertical_g = parameters.effective_vertical_h / parameters.die_thickness

    old = grid.temperature if previous_temperature is None else np.asarray(previous_temperature, dtype=float)
    chosen_power = power_density if power_density is not None else heat_source
    p = grid.power_density if chosen_power is None else np.asarray(chosen_power, dtype=float)
    if old.shape != (rows, cols) or not np.all(np.isfinite(old)):
        raise ValueError("Previous temperature must be finite and match the chip grid.")
    if p.shape != (rows, cols) or not np.all(np.isfinite(p)) or np.any(p < 0):
        raise ValueError("Heat-source map must be finite, nonnegative and match the chip grid.")

    size = rows * cols
    A = lil_matrix((size, size), dtype=float)
    b = (capacity_rate * old.ravel()
         + vertical_g * parameters.sink_temperature
         + (p / parameters.die_thickness).ravel())
    # Cell-face conductance divided by cell volume reduces to these weights.
    wx, wy = k / grid.dx**2, k / grid.dy**2

    for r in range(rows):
        for c in range(cols):
            i = r * cols + c
            diagonal = capacity_rate + vertical_g
            for rr, cc, weight in (
                (r-1, c, wy), (r+1, c, wy),
                (r, c-1, wx), (r, c+1, wx)
            ):
                if 0 <= rr < rows and 0 <= cc < cols:
                    A[i, rr*cols + cc] = -weight
                    diagonal += weight
            A[i, i] = diagonal
    return A.tocsr(), np.asarray(b, dtype=float)
