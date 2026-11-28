"""Grid, material properties and boundary assumptions for a 2D silicon die."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ChipMaterial:
    density: float = 2330.0
    specific_heat: float = 700.0
    conductivity: float = 130.0

    def validate(self):
        if min(self.density, self.specific_heat, self.conductivity) <= 0:
            raise ValueError("Material properties must be positive.")


@dataclass(frozen=True)
class ChipParameters:
    material: ChipMaterial = ChipMaterial()
    die_thickness: float = 0.0005
    dt: float = 1e-4
    sink_temperature: float = 300.0
    effective_vertical_h: float = 10000.0

    def validate(self):
        self.material.validate()
        if self.die_thickness <= 0 or self.dt <= 0 or self.effective_vertical_h < 0:
            raise ValueError("Die thickness and time step must be positive; cooling coefficient cannot be negative.")


@dataclass
class ChipGrid:
    rows: int
    cols: int
    dx: float
    dy: float
    temperature: np.ndarray
    power_density: np.ndarray

    def validate(self):
        shape = (self.rows, self.cols)
        if self.rows < 2 or self.cols < 2:
            raise ValueError("Chip grid must contain at least 2 rows and 2 columns.")
        if self.dx <= 0 or self.dy <= 0:
            raise ValueError("Grid spacing must be positive.")
        if self.temperature.shape != shape or self.power_density.shape != shape:
            raise ValueError("Temperature and power maps must match the chip grid.")
        if not np.all(np.isfinite(self.temperature)) or not np.all(np.isfinite(self.power_density)):
            raise ValueError("Grid arrays must contain finite values.")
        if np.any(self.power_density < 0):
            raise ValueError("Power density cannot be negative.")


def create_chip_grid(rows=21, cols=21, die_width=0.01, die_height=0.01,
                     initial_temperature=300.0, power_density=0.0):
    if rows < 2 or cols < 2 or die_width <= 0 or die_height <= 0:
        raise ValueError("Grid and die dimensions must be positive.")
    p = np.asarray(power_density, dtype=float)
    if p.ndim == 0:
        p = np.full((rows, cols), float(p))
    if p.shape != (rows, cols) or np.any(p < 0) or not np.all(np.isfinite(p)):
        raise ValueError("Power density must be finite, nonnegative and match grid shape.")
    if not np.isfinite(initial_temperature):
        raise ValueError("Initial temperature must be finite.")
    grid = ChipGrid(rows, cols, die_width/(cols-1), die_height/(rows-1),
                    np.full((rows, cols), float(initial_temperature)), p.copy())
    grid.validate()
    return grid


def make_power_map(rows, cols, background=0.0, hot_regions=None):
    """Create spatial power density in W/m^2; region end indices are exclusive."""
    if background < 0:
        raise ValueError("Background power density cannot be negative.")
    p = np.full((rows, cols), float(background))
    for z in hot_regions or []:
        r0, r1 = z["row_start"], z["row_end"]
        c0, c1 = z["col_start"], z["col_end"]
        value = float(z["power_density"])
        if not (0 <= r0 < r1 <= rows and 0 <= c0 < c1 <= cols) or value < 0:
            raise ValueError("Invalid hot-region coordinates or power density.")
        p[r0:r1, c0:c1] = value
    return p
