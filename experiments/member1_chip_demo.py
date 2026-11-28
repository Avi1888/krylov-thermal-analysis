"""Demonstrations of chip thermal behaviour; all power values are illustrative."""
from pathlib import Path
import numpy as np
from src.grid_and_boundary import ChipParameters, ChipMaterial, create_chip_grid, make_power_map
from src.thermal_model import run_thermal_simulation
from src.visualization import plot_chip_heatmap, plot_temperature_trends


def main():
    output = Path("results")
    output.mkdir(exist_ok=True)
    rows = cols = 31
    parameters = ChipParameters(
        material=ChipMaterial(density=2330, specific_heat=700, conductivity=130),
        die_thickness=0.0005, dt=1e-4, sink_temperature=300,
        effective_vertical_h=10000
    )

    # Experiment 1: one active region.
    one_region = make_power_map(rows, cols, background=8000, hot_regions=[
        {"row_start": 9, "row_end": 20, "col_start": 10, "col_end": 21, "power_density": 70000}
    ])
    grid = create_chip_grid(rows, cols, initial_temperature=300, power_density=one_region)
    history = run_thermal_simulation(grid, parameters, 100)
    plot_chip_heatmap(history[-1], output / "one_hot_region.png",
                      "Chip temperature: one active region")
    plot_temperature_trends(history, parameters.dt, output / "one_region_trend.png")
    np.save(output / "one_region_history.npy", history)

    # Experiment 2: two regions with different activity levels.
    multiple = make_power_map(rows, cols, background=8000, hot_regions=[
        {"row_start": 5, "row_end": 13, "col_start": 5, "col_end": 14, "power_density": 80000},
        {"row_start": 18, "row_end": 26, "col_start": 18, "col_end": 27, "power_density": 50000}
    ])
    grid = create_chip_grid(rows, cols, initial_temperature=300, power_density=multiple)
    history = run_thermal_simulation(grid, parameters, 100)
    plot_chip_heatmap(history[-1], output / "multiple_hot_regions.png",
                      "Chip temperature: multiple active regions")
    plot_temperature_trends(history, parameters.dt, output / "multiple_regions_trend.png")
    np.save(output / "multiple_regions_history.npy", history)

    # Experiment 3: workload-like power change during the simulation.
    low = np.full((rows, cols), 10000.0)
    medium = np.full((rows, cols), 30000.0)
    high = np.full((rows, cols), 60000.0)
    grid = create_chip_grid(rows, cols, initial_temperature=300)
    history = run_thermal_simulation(
        grid, parameters, 90,
        power_schedule=[low] * 30 + [medium] * 30 + [high] * 30
    )
    plot_chip_heatmap(history[-1], output / "changing_workload_final.png",
                      "Chip temperature after changing workload")
    plot_temperature_trends(history, parameters.dt, output / "changing_workload_trend.png")
    np.save(output / "changing_workload_history.npy", history)

    print("Member 1 chip thermal experiments completed.")
    print(f"Final maximum temperature (changing workload): {history[-1].max()-273.15:.3f} °C")
    print(f"Final average temperature (changing workload): {history[-1].mean()-273.15:.3f} °C")
    print(f"Results saved in: {output.resolve()}")


if __name__ == "__main__":
    main()
