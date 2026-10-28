#!/usr/bin/env python3
"""Combine Ultralytics/YOLO results.csv files from multiple experiment folders.

Writes one per-epoch table and one per-run summary. The script uses only the
Python standard library and accepts the results directory as an argument.
"""

import argparse
import csv
from pathlib import Path


def read_result_file(csv_path: Path, results_root: Path):
    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle, skipinitialspace=True)
        if not reader.fieldnames:
            return [], []
        fields = [name.strip() for name in reader.fieldnames]
        rows = []
        for source_row in reader:
            row = {str(key).strip(): (value.strip() if value else "")
                   for key, value in source_row.items() if key is not None}
            row["run"] = csv_path.relative_to(results_root).parts[0]
            row["source_csv"] = csv_path.relative_to(results_root).as_posix()
            rows.append(row)
        return fields, rows


def number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir", type=Path,
        default=Path(__file__).resolve().parent / "results",
        help="Root folder containing the experiment folders (default: ./results).",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        help="Where to write output CSVs (default: same as --results-dir).",
    )
    args = parser.parse_args()
    root = args.results_dir.resolve()
    output_dir = (args.output_dir or root).resolve()
    if not root.is_dir():
        parser.error(f"Results directory does not exist: {root}")

    epoch_rows = []
    all_fields = ["run", "source_csv"]
    files = sorted(root.rglob("results.csv"))
    for csv_path in files:
        try:
            fields, rows = read_result_file(csv_path, root)
        except (OSError, csv.Error, UnicodeError) as error:
            print(f"Skipping {csv_path}: {error}")
            continue
        for field in fields:
            if field not in all_fields:
                all_fields.append(field)
        epoch_rows.extend(rows)

    if not epoch_rows:
        parser.error(f"No readable results.csv files found under {root}")

    output_dir.mkdir(parents=True, exist_ok=True)
    combined_path = output_dir / "combined_results.csv"
    with combined_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=all_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(epoch_rows)

    # Choose the best epoch by mAP50-95 when available; retain final-row values
    # as well so runs with missing metric columns remain useful in the summary.
    metric = next((field for field in all_fields
                   if field.lower().replace(" ", "") in
                   {"metrics/map50-95(b)", "metrics/map50-95"}), None)
    grouped = {}
    for row in epoch_rows:
        grouped.setdefault((row["run"], row["source_csv"]), []).append(row)

    summary_fields = ["run", "source_csv", "epochs_recorded", "best_epoch", "best_metric"]
    summary_fields += [f"final_{field}" for field in all_fields if field not in {"run", "source_csv"}]
    summary_rows = []
    for (run, source), rows in grouped.items():
        last = rows[-1]
        eligible = [(number(row.get(metric)), row) for row in rows] if metric else []
        eligible = [(score, row) for score, row in eligible if score is not None]
        best_row = max(eligible, key=lambda item: item[0])[1] if eligible else {}
        summary = {
            "run": run,
            "source_csv": source,
            "epochs_recorded": len(rows),
            "best_epoch": best_row.get("epoch", ""),
            "best_metric": best_row.get(metric, "") if metric else "",
        }
        summary.update({f"final_{field}": last.get(field, "")
                        for field in all_fields if field not in {"run", "source_csv"}})
        summary_rows.append(summary)

    summary_path = output_dir / "run_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Read {len(files)} results.csv file(s), {len(epoch_rows)} epoch row(s).")
    print(f"Per-epoch data: {combined_path}")
    print(f"Per-run summary: {summary_path}")


if __name__ == "__main__":
    main()
