"""Chip temperature visualisation."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def plot_chip_heatmap(temperature, output_path=None, title="Processor chip temperature",
                      die_width=None, die_height=None):
    """Plot cell-centred temperatures, optionally spanning physical die extents.

    When physical dimensions are omitted, axes show cell-centre indices.
    """
    fig, ax = plt.subplots()
    values = np.asarray(temperature) - 273.15
    extent = None
    if die_width is not None or die_height is not None:
        if (die_width is None or die_height is None
                or not np.isfinite(die_width) or not np.isfinite(die_height)
                or die_width <= 0 or die_height <= 0):
            raise ValueError("die_width and die_height must both be finite and positive.")
        extent = (0.0, float(die_width), 0.0, float(die_height))
    image = ax.imshow(values, origin="lower", extent=extent,
                      cmap="inferno", aspect="equal")
    if extent is None:
        ax.set(title=title, xlabel="Chip grid column", ylabel="Chip grid row")
    else:
        ax.set(title=title, xlabel="Chip x position (m)", ylabel="Chip y position (m)")
    fig.colorbar(image, ax=ax, label="Temperature (°C)")
    fig.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=180)
    return fig


def plot_temperature_trends(history, dt, output_path=None):
    values = np.asarray(history) - 273.15
    times = np.arange(values.shape[0]) * dt
    fig, ax = plt.subplots()
    ax.plot(times, values.max(axis=(1, 2)), label="Maximum")
    ax.plot(times, values.mean(axis=(1, 2)), label="Average")
    ax.set(title="Chip temperature versus time", xlabel="Time (s)", ylabel="Temperature (°C)")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=180)
    return fig
