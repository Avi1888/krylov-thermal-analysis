"""Unit tests for measured Ryzen thermal-map analysis."""

import pickle

import numpy as np
import pytest

from src.ryzen_thermal_analysis import (
    analyze_consecutive_map_changes,
    analyze_thermal_maps,
    load_ryzen_dataset,
    validate_ryzen_dataset,
)


def test_validates_and_loads_small_dataset(tmp_path):
    data = {"input": np.zeros((2, 3, 4)), "output": np.ones((2, 5, 6)) * 45}
    path = tmp_path / "sample.pkl"
    with path.open("wb") as stream:
        pickle.dump(data, stream)

    loaded = load_ryzen_dataset(
        path, expected_input_shape=data["input"].shape,
        expected_output_shape=data["output"].shape,
    )

    assert loaded["input"].shape == (2, 3, 4)
    assert loaded["output"].shape == (2, 5, 6)


def test_validates_required_dataset_structure():
    with pytest.raises(ValueError, match="input.*output"):
        validate_ryzen_dataset({"input": np.zeros((1, 2, 3))})
    with pytest.raises(ValueError, match="input must have shape"):
        validate_ryzen_dataset({"input": np.zeros((2, 3)), "output": np.zeros((2, 4, 5))})
    with pytest.raises(ValueError, match="sample count"):
        validate_ryzen_dataset({"input": np.zeros((2, 3, 4)), "output": np.zeros((1, 4, 5))})


def test_map_statistics_and_hotspot_indices():
    image = np.array([[1.0, 2.0], [3.0, 10.0]])
    rows, summary = analyze_thermal_maps(image[None, ...])

    assert rows[0]["minimum_c"] == 1.0
    assert rows[0]["maximum_c"] == 10.0
    assert rows[0]["mean_c"] == 4.0
    assert rows[0]["median_c"] == 2.5
    assert rows[0]["standard_deviation_c"] == pytest.approx(np.std(image))
    assert rows[0]["range_c"] == 9.0
    assert rows[0]["percentile_95_c"] == pytest.approx(np.percentile(image, 95))
    assert rows[0]["percentile_99_c"] == pytest.approx(np.percentile(image, 99))
    assert rows[0]["hotspot_row_index"] == 1
    assert rows[0]["hotspot_column_index"] == 1
    assert summary["map_count"] == 1
    assert summary["highest_maximum_map_index"] == 0
    assert summary["highest_mean_map_index"] == 0


def test_consecutive_map_changes_use_current_minus_previous():
    maps = np.array([
        [[1.0, 2.0], [3.0, 4.0]],
        [[2.0, 1.0], [5.0, 6.0]],
    ])

    rows = analyze_consecutive_map_changes(maps)

    assert len(rows) == 1
    assert rows[0]["previous_map_index"] == 0
    assert rows[0]["current_map_index"] == 1
    assert rows[0]["mean_temperature_change_c"] == pytest.approx(1.0)
    assert rows[0]["maximum_temperature_change_c"] == pytest.approx(2.0)
    assert rows[0]["pixel_difference_minimum_c"] == -1.0
    assert rows[0]["pixel_difference_maximum_c"] == 2.0
    assert rows[0]["pixel_difference_mean_c"] == pytest.approx(1.0)


@pytest.mark.parametrize("bad_maps", [
    np.zeros((2, 3)),
    np.zeros((0, 2, 3)),
])
def test_rejects_invalid_map_shapes(bad_maps):
    with pytest.raises(ValueError, match="thermal_maps_c"):
        analyze_thermal_maps(bad_maps)


def test_rejects_non_finite_thermal_values():
    maps = np.array([[[40.0, np.nan]]])
    with pytest.raises(ValueError, match="finite"):
        analyze_thermal_maps(maps)


def test_rejects_non_finite_dataset_values():
    data = {"input": np.zeros((1, 2, 3)), "output": np.zeros((1, 2, 2))}
    data["output"][0, 0, 0] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        validate_ryzen_dataset(data, expected_input_shape=None, expected_output_shape=None)
