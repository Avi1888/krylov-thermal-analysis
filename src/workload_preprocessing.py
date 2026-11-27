"""Inspection and conservative preprocessing for processor workload data.

The Ryzen pickle stores unlabeled metrics. This module therefore requires an
explicit metric index when deriving workload states; it never guesses which
feature means utilization, power, or workload intensity.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np


def load_thermal_dataset(path: str | Path) -> dict[str, np.ndarray]:
    """Load the known dataset format from a trusted pickle file.

    Pickle files can execute code when loaded. Only load files from a trusted
    source. Expected keys are ``input`` and ``output``.
    """
    with Path(path).open("rb") as stream:
        data = pickle.load(stream)
    if not isinstance(data, dict) or not {"input", "output"}.issubset(data):
        raise ValueError("Dataset must be a dictionary with 'input' and 'output' keys.")
    inputs = np.asarray(data["input"])
    outputs = np.asarray(data["output"])
    if inputs.ndim not in (2, 3):
        raise ValueError("input must have shape (samples, features) or (samples, time, features).")
    if outputs.ndim != 3:
        raise ValueError("output must have shape (samples, height, width).")
    if inputs.shape[0] != outputs.shape[0]:
        raise ValueError("input and output sample counts do not match.")
    return {"input": inputs, "output": outputs}


def inspect_thermal_dataset(data_or_path: dict[str, Any] | str | Path) -> dict[str, Any]:
    """Return shapes, types, finite-value counts, and basic statistics."""
    data = (load_thermal_dataset(data_or_path)
            if isinstance(data_or_path, (str, Path)) else data_or_path)
    if not isinstance(data, dict) or not {"input", "output"}.issubset(data):
        raise ValueError("Expected a dictionary with 'input' and 'output' arrays.")
    inputs = np.asarray(data["input"])
    outputs = np.asarray(data["output"])
    report: dict[str, Any] = {"keys": list(data.keys()), "arrays": {}}
    for name in ("input", "output"):
        values = np.asarray(data[name])
        finite = np.isfinite(values)
        finite_values = values[finite]
        stats: dict[str, Any] = {
            "shape": tuple(values.shape),
            "ndim": int(values.ndim),
            "dtype": str(values.dtype),
            "finite_count": int(finite.sum()),
            "invalid_count": int(values.size - finite.sum()),
        }
        if finite_values.size:
            stats.update(min=float(finite_values.min()), max=float(finite_values.max()),
                         mean=float(finite_values.mean()), std=float(finite_values.std()))
        report["arrays"][name] = stats
    if inputs.ndim == 3:
        report["input_layout"] = {
            "samples": int(inputs.shape[0]),
            "sequence_length": int(inputs.shape[1]),
            "feature_count": int(inputs.shape[2]),
        }
        feature_stats = []
        for index in range(inputs.shape[2]):
            feature = np.asarray(inputs[..., index], dtype=float)
            finite_values = feature[np.isfinite(feature)]
            feature_stats.append({
                "metric_index": index,
                "finite_count": int(finite_values.size),
                "invalid_count": int(feature.size - finite_values.size),
                "min": float(finite_values.min()) if finite_values.size else None,
                "max": float(finite_values.max()) if finite_values.size else None,
                "mean": float(finite_values.mean()) if finite_values.size else None,
                "std": float(finite_values.std()) if finite_values.size else None,
            })
        report["feature_statistics"] = feature_stats
    elif inputs.ndim == 2:
        report["input_layout"] = {
            "samples": int(inputs.shape[0]),
            "feature_count": int(inputs.shape[1]),
        }
    report["sample_counts_match"] = bool(inputs.shape[0] == outputs.shape[0])
    report["input_has_sequence_axis"] = bool(inputs.ndim == 3)
    report["output_is_thermal_map_shaped"] = bool(outputs.ndim == 3)
    report["chronology_note"] = (
        "Array order is preserved, but chronology must be confirmed from dataset metadata."
    )
    return report


def workload_scores(
    inputs: Any,
    metric_index: int,
    temporal_reducer: str = "mean",
) -> np.ndarray:
    """Reduce one explicitly selected metric to one score per outer sample.

    For 3D input, ``metric_index`` selects the last-axis feature and
    ``temporal_reducer`` is ``mean`` or ``last`` across the sequence axis.
    For 2D input, each row is already one sample and the selected column is
    returned directly. Feature meaning and chronological sample order remain
    the caller's responsibility.
    """
    values = np.asarray(inputs, dtype=float)
    if values.ndim not in (2, 3):
        raise ValueError("inputs must be a 2D or 3D array.")
    n_features = values.shape[-1]
    if isinstance(metric_index, bool) or not 0 <= metric_index < n_features:
        raise ValueError(f"metric_index must be in [0, {n_features - 1}].")
    if not np.all(np.isfinite(values)):
        raise ValueError("inputs contain non-finite values; clean them before scoring.")
    selected = values[..., metric_index]
    if selected.ndim == 1:
        return selected.copy()
    if temporal_reducer == "mean":
        return selected.mean(axis=1)
    if temporal_reducer == "last":
        return selected[:, -1].copy()
    raise ValueError("temporal_reducer must be 'mean' or 'last'.")


def categorize_workload(
    scores: Any,
    quantiles: tuple[float, float] = (1 / 3, 2 / 3),
) -> tuple[np.ndarray, tuple[float, float]]:
    """Map scores to Low/Medium/High using documented empirical quantiles."""
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("scores must be a non-empty finite one-dimensional array.")
    low_q, high_q = quantiles
    if not (0 <= low_q < high_q <= 1):
        raise ValueError("quantiles must satisfy 0 <= low < high <= 1.")
    low_threshold, high_threshold = np.quantile(values, [low_q, high_q])
    labels = np.full(values.shape, "High", dtype="U6")
    labels[values <= low_threshold] = "Low"
    labels[(values > low_threshold) & (values <= high_threshold)] = "Medium"
    return labels, (float(low_threshold), float(high_threshold))


def map_workload_to_heat_intensity(
    state: str,
    heat_levels: dict[str, float] | None = None,
) -> float:
    """Map a state to a configurable relative heat level.

    Returned values are normalized relative intensities, not watts or
    calibrated processor power. Defaults are illustrative and can be replaced.
    """
    levels = ({"Low": 0.25, "Medium": 0.60, "High": 1.0}
              if heat_levels is None else dict(heat_levels))
    if set(levels) != {"Low", "Medium", "High"}:
        raise ValueError("heat_levels must define exactly Low, Medium, and High.")
    values = np.asarray([levels[name] for name in ("Low", "Medium", "High")], dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("Heat intensities must be finite and non-negative.")
    if not (values[0] <= values[1] <= values[2]):
        raise ValueError("Heat levels must be ordered Low <= Medium <= High.")
    if state not in levels:
        raise ValueError(f"Unknown workload state: {state!r}.")
    return float(levels[state])


def make_spatial_heat_source(
    state: str,
    grid_shape: tuple[int, int],
    heat_levels: dict[str, float] | None = None,
    spatial_weights: Any | None = None,
) -> np.ndarray:
    """Create a relative spatial source array for the thermal-model interface.

    By default, intensity is uniform across the grid. A supplied non-negative
    weight map is normalized to sum to one, so its shape controls distribution
    while the selected state controls total relative intensity.
    """
    if len(grid_shape) != 2 or any(isinstance(size, bool) or int(size) != size or size <= 0
                                  for size in grid_shape):
        raise ValueError("grid_shape must contain two positive integers.")
    grid_shape = tuple(int(size) for size in grid_shape)
    intensity = map_workload_to_heat_intensity(state, heat_levels)
    if spatial_weights is None:
        weights = np.full(grid_shape, 1.0 / (grid_shape[0] * grid_shape[1]), dtype=float)
    else:
        weights = np.asarray(spatial_weights, dtype=float)
        if weights.shape != tuple(grid_shape) or not np.all(np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError("spatial_weights must be finite, non-negative, and match grid_shape.")
        total = float(weights.sum())
        if total <= 0:
            raise ValueError("spatial_weights must have a positive sum.")
        weights = weights / total
    return intensity * weights
