"""Analysis utilities for the measured Ryzen 7 4800U thermal maps.

The dataset's ``output`` array contains measured temperatures in degrees
Celsius. Spatial locations in this module are array row/column indices only;
the pickle does not provide a pixel-to-physical-coordinate registration.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_INPUT_SHAPE = (100, 100, 158)
EXPECTED_OUTPUT_SHAPE = (100, 223, 280)


def validate_ryzen_dataset(
    data: Any,
    *,
    expected_input_shape: tuple[int, ...] | None = EXPECTED_INPUT_SHAPE,
    expected_output_shape: tuple[int, ...] | None = EXPECTED_OUTPUT_SHAPE,
) -> dict[str, np.ndarray]:
    """Validate the pickle structure and return its input/output arrays.

    Optional shape expectations allow unit tests to use small artificial
    datasets. Production loading uses the documented local dataset shapes.
    No telemetry column names, meanings, or units are inferred here.
    """
    if not isinstance(data, dict) or not {"input", "output"}.issubset(data):
        raise ValueError("Dataset must be a dictionary containing 'input' and 'output'.")

    inputs = np.asarray(data["input"])
    maps = np.asarray(data["output"])
    if inputs.ndim != 3:
        raise ValueError("input must have shape (samples, window, telemetry_features).")
    if maps.ndim != 3:
        raise ValueError("output must have shape (maps, pixel_rows, pixel_columns).")
    if inputs.shape[0] != maps.shape[0]:
        raise ValueError("input sample count and output map count must match.")
    if expected_input_shape is not None and inputs.shape != expected_input_shape:
        raise ValueError(f"input shape must be {expected_input_shape}; got {inputs.shape}.")
    if expected_output_shape is not None and maps.shape != expected_output_shape:
        raise ValueError(f"output shape must be {expected_output_shape}; got {maps.shape}.")
    if not np.issubdtype(inputs.dtype, np.number) or not np.issubdtype(maps.dtype, np.number):
        raise ValueError("input and output arrays must contain numeric values.")
    if not np.all(np.isfinite(inputs)):
        raise ValueError("input contains non-finite telemetry values.")
    if not np.all(np.isfinite(maps)):
        raise ValueError("output contains non-finite temperature values.")
    return {"input": inputs, "output": maps}


def load_ryzen_dataset(
    path: str | Path,
    *,
    expected_input_shape: tuple[int, ...] | None = EXPECTED_INPUT_SHAPE,
    expected_output_shape: tuple[int, ...] | None = EXPECTED_OUTPUT_SHAPE,
) -> dict[str, np.ndarray]:
    """Load and validate a trusted Ryzen pickle file.

    Pickle deserialization can execute code. Only load dataset files from a
    trusted source.
    """
    with Path(path).open("rb") as stream:
        data = pickle.load(stream)
    return validate_ryzen_dataset(
        data,
        expected_input_shape=expected_input_shape,
        expected_output_shape=expected_output_shape,
    )


def _validated_maps(thermal_maps_c: Any) -> np.ndarray:
    maps = np.asarray(thermal_maps_c, dtype=float)
    if maps.ndim != 3 or any(size < 1 for size in maps.shape):
        raise ValueError("thermal_maps_c must have shape (maps, rows, columns) with positive sizes.")
    if not np.all(np.isfinite(maps)):
        raise ValueError("thermal_maps_c must contain only finite Celsius temperatures.")
    return maps


def analyze_thermal_maps(thermal_maps_c: Any) -> tuple[list[dict[str, int | float]], dict[str, int | float]]:
    """Compute per-map and pooled statistics from measured Celsius maps."""
    maps = _validated_maps(thermal_maps_c)
    rows: list[dict[str, int | float]] = []
    map_means = np.empty(maps.shape[0], dtype=float)
    map_maxima = np.empty(maps.shape[0], dtype=float)

    for map_index, image in enumerate(maps):
        maximum = float(np.max(image))
        hotspot_row, hotspot_column = np.unravel_index(int(np.argmax(image)), image.shape)
        mean = float(np.mean(image))
        map_means[map_index] = mean
        map_maxima[map_index] = maximum
        rows.append({
            "map_index": map_index,
            "minimum_c": float(np.min(image)),
            "maximum_c": maximum,
            "mean_c": mean,
            "median_c": float(np.median(image)),
            "standard_deviation_c": float(np.std(image)),
            "range_c": maximum - float(np.min(image)),
            "percentile_95_c": float(np.percentile(image, 95)),
            "percentile_99_c": float(np.percentile(image, 99)),
            "hotspot_row_index": int(hotspot_row),
            "hotspot_column_index": int(hotspot_column),
        })

    pooled = maps.ravel()
    summary: dict[str, int | float] = {
        "map_count": int(maps.shape[0]),
        "pixel_rows": int(maps.shape[1]),
        "pixel_columns": int(maps.shape[2]),
        "overall_minimum_c": float(np.min(pooled)),
        "overall_maximum_c": float(np.max(pooled)),
        "overall_mean_c": float(np.mean(pooled)),
        "overall_standard_deviation_c": float(np.std(pooled)),
        "highest_maximum_map_index": int(np.argmax(map_maxima)),
        "highest_mean_map_index": int(np.argmax(map_means)),
    }
    return rows, summary


def analyze_consecutive_map_changes(
    thermal_maps_c: Any,
) -> list[dict[str, int | float]]:
    """Summarize differences between consecutive stored measured maps.

    The differences are per stored-map step. No elapsed time is inferred, so
    these values are not rates or changes per second.
    """
    maps = _validated_maps(thermal_maps_c)
    changes: list[dict[str, int | float]] = []
    for previous_index in range(maps.shape[0] - 1):
        previous = maps[previous_index]
        current = maps[previous_index + 1]
        difference = current - previous
        changes.append({
            "previous_map_index": previous_index,
            "current_map_index": previous_index + 1,
            "mean_temperature_change_c": float(np.mean(current) - np.mean(previous)),
            "maximum_temperature_change_c": float(np.max(current) - np.max(previous)),
            "pixel_difference_minimum_c": float(np.min(difference)),
            "pixel_difference_maximum_c": float(np.max(difference)),
            "pixel_difference_mean_c": float(np.mean(difference)),
            "pixel_difference_median_c": float(np.median(difference)),
            "pixel_difference_standard_deviation_c": float(np.std(difference)),
            "pixel_difference_percentile_95_c": float(np.percentile(difference, 95)),
            "pixel_difference_percentile_99_c": float(np.percentile(difference, 99)),
        })
    return changes


def representative_map_index(map_rows: list[dict[str, int | float]]) -> int:
    """Return the map whose mean temperature is closest to the median mean."""
    if not map_rows:
        raise ValueError("map_rows cannot be empty.")
    means = np.asarray([row["mean_c"] for row in map_rows], dtype=float)
    return int(np.argmin(np.abs(means - np.median(means))))


def generate_thermal_figures(
    thermal_maps_c: Any,
    map_rows: list[dict[str, int | float]],
    output_dir: str | Path,
) -> list[Path]:
    """Save the required measured-map figures and return their paths."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    maps = _validated_maps(thermal_maps_c)
    if len(map_rows) != maps.shape[0]:
        raise ValueError("map_rows must contain one statistics row per thermal map.")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    max_map_index = int(np.argmax([row["maximum_c"] for row in map_rows]))
    mean_map_index = int(np.argmax([row["mean_c"] for row in map_rows]))
    representative_index = representative_map_index(map_rows)

    def save_map(index: int, filename: str, title: str) -> None:
        fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
        image = ax.imshow(maps[index], origin="upper", cmap="inferno", interpolation="nearest")
        ax.set_title(f"{title} (stored map {index})")
        ax.set_xlabel("Pixel column index")
        ax.set_ylabel("Pixel row index")
        colorbar = fig.colorbar(image, ax=ax)
        colorbar.set_label("Measured temperature (°C)")
        path = out / filename
        fig.savefig(path, dpi=160)
        plt.close(fig)
        saved.append(path)

    save_map(representative_index, "representative_measured_map.png", "Representative measured thermal map")
    save_map(max_map_index, "highest_maximum_measured_map.png", "Measured map with highest maximum temperature")
    save_map(mean_map_index, "highest_mean_measured_map.png", "Measured map with highest mean temperature")

    selected = sorted(set((0, maps.shape[0] // 4, maps.shape[0] // 2,
                           (3 * maps.shape[0]) // 4, maps.shape[0] - 1)))
    fig, axes = plt.subplots(1, len(selected), figsize=(4 * len(selected), 4.5), constrained_layout=True)
    low, high = float(np.min(maps[selected])), float(np.max(maps[selected]))
    for ax, index in zip(np.atleast_1d(axes), selected):
        image = ax.imshow(maps[index], origin="upper", cmap="inferno", vmin=low, vmax=high,
                          interpolation="nearest")
        ax.set_title(f"Stored map {index}")
        ax.set_xlabel("Pixel column index")
        ax.set_ylabel("Pixel row index")
    colorbar = fig.colorbar(image, ax=list(np.atleast_1d(axes)), shrink=0.82)
    colorbar.set_label("Measured temperature (°C)")
    path = out / "selected_stored_maps.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    saved.append(path)

    means = np.asarray([row["mean_c"] for row in map_rows], dtype=float)
    maxima = np.asarray([row["maximum_c"] for row in map_rows], dtype=float)
    indices = np.arange(maps.shape[0])
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.plot(indices, means, label="Mean temperature", color="tab:blue")
    ax.plot(indices, maxima, label="Maximum temperature", color="tab:red")
    ax.set_xlabel("Stored map index (relative sequence order)")
    ax.set_ylabel("Measured temperature (°C)")
    ax.set_title("Measured Ryzen thermal-map temperatures across stored maps")
    ax.grid(True, alpha=0.25)
    ax.legend()
    path = out / "temperature_statistics_across_maps.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    saved.append(path)
    return saved
