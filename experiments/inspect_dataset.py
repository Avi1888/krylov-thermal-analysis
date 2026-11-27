"""Inspect the supplied thermal dataset without exporting full arrays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.benchmark import save_summary_csv
from src.workload_preprocessing import inspect_thermal_dataset


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path,
        default=project_root / "data" / "raw" / "CPU_R7_4800U.pkl",
        help="Path to the trusted dataset pickle.",
    )
    parser.add_argument(
        "--feature-csv", type=Path,
        default=project_root / "results" / "tables" / "dataset_feature_statistics.csv",
        help="Where to write per-metric summary statistics.",
    )
    args = parser.parse_args()
    report = inspect_thermal_dataset(args.dataset)
    feature_rows = report.pop("feature_statistics", [])
    print(json.dumps(report, indent=2))
    if feature_rows:
        save_summary_csv(feature_rows, args.feature_csv)
        print(f"Per-metric statistics saved to {args.feature_csv}")


if __name__ == "__main__":
    main()
