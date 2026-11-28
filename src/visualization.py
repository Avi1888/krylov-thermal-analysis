"""Chip temperature visualisation."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def plot_chip_heatmap(temperature, output_path=None, title="Processor chip temperature"):
    fig, ax = plt.subplots()
    image = ax.imshow(np.asarray(temperature) - 273.15, origin="lower",
                      cmap="inferno", aspect="equal")
    ax.set(title=title, xlabel="Chip grid column", ylabel="Chip grid row")
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
