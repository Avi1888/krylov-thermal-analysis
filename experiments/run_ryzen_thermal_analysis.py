"""Analyze measured thermal maps from the Ryzen 7 4800U dataset."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from src.ryzen_thermal_analysis import (
    EXPECTED_INPUT_SHAPE,
    EXPECTED_OUTPUT_SHAPE,
    analyze_consecutive_map_changes,
    analyze_thermal_maps,
    generate_thermal_figures,
    load_ryzen_dataset,
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path,
        default=project_root / "data" / "raw" / "CPU_R7_4800U.pkl",
        help="Trusted measured Ryzen 7 4800U pickle file.",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=project_root / "results" / "ryzen_measured",
        help="Output directory for measured-map tables and figures.",
    )
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    synthetic_dir = (project_root / "results" / "synthetic").resolve()
    if output_dir == synthetic_dir or synthetic_dir in output_dir.parents:
        parser.error("Measured results must not be written inside results/synthetic.")

    dataset = load_ryzen_dataset(args.dataset)
    maps = dataset["output"]
    map_rows, summary = analyze_thermal_maps(maps)
    change_rows = analyze_consecutive_map_changes(maps)

    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    map_csv = tables_dir / "measured_map_statistics.csv"
    change_csv = tables_dir / "consecutive_map_changes.csv"
    _write_csv(map_csv, map_rows)
    _write_csv(change_csv, change_rows)
    figures = generate_thermal_figures(maps, map_rows, figures_dir)

    max_index = int(summary["highest_maximum_map_index"])
    mean_index = int(summary["highest_mean_map_index"])
    print("Data source: measured AMD Ryzen 7 4800U dataset")
    print(f"Dataset: {args.dataset.resolve()}")
    print(f"Input shape: {EXPECTED_INPUT_SHAPE}; measured output shape: {maps.shape}")
    print(f"Measured maps analyzed: {summary['map_count']}")
    print(
        "Across all measured map pixels: "
        f"min={summary['overall_minimum_c']:.4f} deg C, "
        f"max={summary['overall_maximum_c']:.4f} deg C, "
        f"mean={summary['overall_mean_c']:.4f} deg C, "
        f"std={summary['overall_standard_deviation_c']:.4f} deg C"
    )
    print(
        f"Highest map maximum: index {max_index}, "
        f"{map_rows[max_index]['maximum_c']:.4f} deg C"
    )
    print(
        f"Highest map mean: index {mean_index}, "
        f"{map_rows[mean_index]['mean_c']:.4f} deg C"
    )
    print(f"Consecutive stored-map changes calculated: {len(change_rows)}")
    if change_rows:
        mean_deltas = [row["mean_temperature_change_c"] for row in change_rows]
        maximum_deltas = [row["maximum_temperature_change_c"] for row in change_rows]
        print(
            "Consecutive-map mean change range: "
            f"{min(mean_deltas):.4f} to {max(mean_deltas):.4f} deg C; "
            "maximum change range: "
            f"{min(maximum_deltas):.4f} to {max(maximum_deltas):.4f} deg C"
        )
    print("Map-to-map differences are per stored-map step, not per unit time.")
    print(f"Map statistics CSV: {map_csv}")
    print(f"Consecutive-map changes CSV: {change_csv}")
    print("Figures:")
    for figure in figures:
        print(f"  {figure}")


if __name__ == "__main__":
    main()
