import numpy as np
from src.grid_and_boundary import create_chip_grid, ChipParameters, make_power_map
from src.thermal_model import run_thermal_simulation


def test_no_power_remains_at_sink_temperature():
    grid = create_chip_grid(5, 5, initial_temperature=300.0)
    history = run_thermal_simulation(grid, ChipParameters(sink_temperature=300.0), 2)
    assert np.allclose(history, 300.0)


def test_powered_chip_heats_up():
    power = make_power_map(7, 7, hot_regions=[
        {"row_start": 2, "row_end": 5, "col_start": 2, "col_end": 5, "power_density": 80000}
    ])
    grid = create_chip_grid(7, 7, power_density=power)
    history = run_thermal_simulation(grid, ChipParameters(), 5)
    assert history[-1].max() > history[0].max()
    assert history[-1].max() > history[-1].min()


def test_time_varying_power_schedule_is_accepted():
    grid = create_chip_grid(5, 5)
    params = ChipParameters()
    low = np.full((5, 5), 10000.0)
    high = np.full((5, 5), 50000.0)
    history = run_thermal_simulation(grid, params, 2, power_schedule=[low, high])
    assert history.shape == (3, 5, 5)
    assert history[-1].mean() > history[1].mean()
