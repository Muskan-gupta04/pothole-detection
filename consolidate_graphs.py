#!/usr/bin/env python3
"""Group YOLO plots by chart type and compare key metrics across runs."""

import argparse
import csv
import shutil
from pathlib import Path


PLOT_EXTENSIONS = {".png", ".jpg", ".jpeg", ".svg", ".pdf"}
METRICS = {
    "precision": ("metrics/precision(B)", "metrics/precision"),
    "recall": ("metrics/recall(B)", "metrics/recall"),
    "map50": ("metrics/mAP50(B)", "metrics/mAP50"),
    "map50_95": ("metrics/mAP50-95(B)", "metrics/mAP50-95"),
}


def category_for(filename):
    name = filename.lower()
    if "confusion_matrix" in name:
        return "confusion_matrices"
    if "f1_curve" in name:
        return "f1_curves"
    if "labels_correlogram" in name or "correlogram" in name:
        return "correlograms"
    if "p_curve" in name or "precision_curve" in name:
        return "precision_curves"
    if "r_curve" in name or "recall_curve" in name:
        return "recall_curves"
    if "pr_curve" in name:
        return "precision_recall_curves"
    if name == "results.png":
        return "training_loss_curves"
    if name.startswith("train_batch") or name.startswith("val_batch"):
        return "batch_visualizations"
    if name == "labels.jpg":
        return "label_distributions"
    return "other_graphs"


def read_metrics(root):
    """Return (run, epoch, metric values) rows from every YOLO results.csv."""
    records = []
    for path in sorted(root.rglob("results.csv")):
        try:
            with path.open("r", newline="", encoding="utf-8-sig") as handle:
                reader = csv.DictReader(handle, skipinitialspace=True)
                for raw in reader:
                    row = {str(key).strip(): value.strip() if value else ""
                           for key, value in raw.items() if key is not None}
                    try:
                        epoch = float(row.get("epoch", ""))
                    except ValueError:
                        continue
                    values = {}
                    for metric, aliases in METRICS.items():
                        key = next((candidate for candidate in aliases if candidate in row), None)
                        try:
                            values[metric] = float(row[key]) if key and row[key] else None
                        except ValueError:
                            values[metric] = None
                    records.append((path.relative_to(root).parts[0], epoch, values))
        except (OSError, csv.Error, UnicodeError) as error:
            print(f"Skipping {path}: {error}")
    return records


def write_metric_charts(records, metrics_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise SystemExit(
            "Metric comparison charts require matplotlib. Install it with: "
            "python -m pip install matplotlib"
        ) from error

    titles = {
        "precision": "Precision by epoch",
        "recall": "Recall by epoch",
        "map50": "mAP50 by epoch",
        "map50_95": "mAP50-95 by epoch",
    }
    for metric, title in titles.items():
        fig, ax = plt.subplots(figsize=(11, 6.5))
        grouped = {}
        for run, epoch, values in records:
            value = values[metric]
            if value is not None:
                grouped.setdefault(run, []).append((epoch, value))
        for run, points in sorted(grouped.items()):
            points.sort(key=lambda point: point[0])
            ax.plot([point[0] for point in points], [point[1] for point in points],
                    marker="o", markersize=2.5, linewidth=1.5, label=run)
        ax.set(title=title, xlabel="Epoch", ylabel=metric.replace("_", "-").upper())
        ax.grid(True, alpha=0.3)
        if grouped:
            ax.legend(fontsize="small", bbox_to_anchor=(1.02, 1), loc="upper left")
        else:
            ax.text(0.5, 0.5, "No matching metric values found", ha="center", va="center",
                    transform=ax.transAxes)
        fig.tight_layout()
        fig.savefig(metrics_dir / f"compare_{metric}.png", dpi=160, bbox_inches="tight")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir", type=Path,
        default=Path(__file__).resolve().parent / "results",
        help="Root folder containing experiment folders (default: ./results).",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        help="Destination for grouped charts (default: <results-dir>/all_graphs/grouped).",
    )
    args = parser.parse_args()
    root = args.results_dir.resolve()
    output = (args.output_dir or root / "all_graphs" / "grouped").resolve()
    if not root.is_dir():
        parser.error(f"Results directory does not exist: {root}")
    if output == root or output in root.parents:
        parser.error("Output folder cannot be the results root or one of its parent folders.")

    output.mkdir(parents=True, exist_ok=True)
    previous_collection = (root / "all_graphs").resolve()
    manifest = []
    for source in sorted(root.rglob("*")):
        if not source.is_file() or source.suffix.lower() not in PLOT_EXTENSIONS:
            continue
        resolved = source.resolve()
        if (resolved == output or output in resolved.parents
                or resolved == previous_collection or previous_collection in resolved.parents):
            continue
        relative = source.relative_to(root)
        run = relative.parts[0] if len(relative.parts) > 1 else "root_level"
        category = category_for(source.name)
        destination = output / category / run / Path(*relative.parts[1:])
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        manifest.append({
            "category": category,
            "run": run,
            "graph": destination.relative_to(output).as_posix(),
            "source": relative.as_posix(),
        })

    metrics_dir = output / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    records = read_metrics(root)
    write_metric_charts(records, metrics_dir)
    for metric in METRICS:
        manifest.append({
            "category": "metrics",
            "run": "all_runs",
            "graph": f"metrics/compare_{metric}.png",
            "source": "generated from results.csv files",
        })

    with (output / "graph_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["category", "run", "graph", "source"])
        writer.writeheader()
        writer.writerows(manifest)

    print(f"Grouped {len(manifest) - len(METRICS)} existing plot/image file(s).")
    print(f"Generated 4 metric comparison charts from {len(records)} epoch row(s).")
    print(f"Grouped output: {output}")
    print(f"Index: {output / 'graph_manifest.csv'}")


if __name__ == "__main__":
    main()
