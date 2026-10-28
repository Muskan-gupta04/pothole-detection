#!/usr/bin/env python3
"""Interactive dashboard for YOLO pothole runs and the cited paper benchmark.

Launch from this directory with: streamlit run dashboard.py
"""

from pathlib import Path
import csv

import pandas as pd
import plotly.express as px
import streamlit as st
import yaml


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
PLOT_EXTENSIONS = {".png", ".jpg", ".jpeg", ".svg"}

# Values transcribed from Fortin & Llantos (2025), Tables 1-4 and conclusion.
PAPER = {
    "YOLOv9-tiny": {
        "precision": 0.90807, "recall": 0.81686, "map50": 0.88365,
        "box_loss": 1.00696, "cls_loss": 0.57491, "dfl_loss": 1.07305,
        "epochs": 393, "initial_map50": 0.40021, "tp": 6485,
        "false_positives": 745, "miss_rate": 0.165, "seconds_per_epoch": 163.2,
    },
    "YOLOv10-nano": {
        "precision": 0.88353, "recall": 0.80817, "map50": 0.84483,
        "box_loss": 1.14732, "cls_loss": 0.69205, "dfl_loss": 1.14037,
        "epochs": 134, "initial_map50": 0.42257, "tp": 6106,
        "false_positives": 1147, "miss_rate": 0.213, "seconds_per_epoch": 329.5,
    },
    "YOLOv11-nano": {
        "precision": 0.90713, "recall": 0.82107, "map50": 0.86888,
        "box_loss": 1.06211, "cls_loss": 0.63428, "dfl_loss": 1.10862,
        "epochs": 265, "initial_map50": 0.40021, "tp": 6379,
        "false_positives": None, "miss_rate": 0.178, "seconds_per_epoch": 198.8,
    },
}

PAPER_FIGURES = [
    ("Figure 1", "mAP@0.5 convergence by epoch", "Available: mAP50 epoch histories in all run CSVs."),
    ("Figure 2", "Box loss convergence by epoch", "Available: train/box_loss history; shown in the YOLO results plots."),
    ("Figure 3", "Detection/error examples associated with confusion analysis", "Partial: confusion matrices and validation prediction batches exist; paper-style selected FP/FN examples are not curated."),
    ("Table 2", "Precision, recall, mAP and final losses", "Available: precision, recall, mAP50 and train losses; F1 is calculated in this dashboard."),
    ("Table 3", "GPU, image size, batch, dataset size and split", "Partial: each run has args.yaml with model, image size, batch and optimizer; exact dataset split counts and recorded GPU model are not available."),
    ("Table 4", "Initial/final LR and best metrics", "Partial: LR and per-epoch metrics exist; paper's statistical SD/p-values and comparable test protocol are absent."),
    ("Execution efficiency", "Training time, inference speed and memory", "Partial: cumulative training time exists; standardized inference speed and memory measurements are not in results.csv."),
    ("Confusion analysis", "TP, FP, FN/miss rate", "Partial: confusion matrix images exist; counts can be read visually, but raw confusion counts are not consolidated."),
]


def f1_score(precision, recall):
    if pd.isna(precision) or pd.isna(recall) or precision + recall == 0:
        return float("nan")
    return 2 * precision * recall / (precision + recall)


@st.cache_data(show_spinner=False)
def load_runs(results_dir_string):
    root = Path(results_dir_string)
    frames = []
    for path in sorted(root.rglob("results.csv")):
        # Only source YOLO files; ignore future aggregate outputs.
        if "all_graphs" in path.parts:
            continue
        try:
            frame = pd.read_csv(path, skipinitialspace=True)
        except (OSError, UnicodeError, pd.errors.ParserError):
            continue
        frame.columns = [str(column).strip() for column in frame.columns]
        if "epoch" not in frame.columns:
            continue
        frame["epoch"] = pd.to_numeric(frame["epoch"], errors="coerce")
        frame = frame.dropna(subset=["epoch"]).copy()
        frame["run"] = path.relative_to(root).parts[0]
        frame["source_csv"] = path.relative_to(root).as_posix()
        for column in frame.columns:
            if column not in {"run", "source_csv"}:
                frame[column] = pd.to_numeric(frame[column], errors="ignore")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def metric_column(frame, metric):
    aliases = {
        "precision": ["metrics/precision(B)", "metrics/precision"],
        "recall": ["metrics/recall(B)", "metrics/recall"],
        "map50": ["metrics/mAP50(B)", "metrics/mAP50"],
        "map50_95": ["metrics/mAP50-95(B)", "metrics/mAP50-95"],
        "box_loss": ["train/box_loss"],
        "cls_loss": ["train/cls_loss"],
        "dfl_loss": ["train/dfl_loss"],
        "val_box_loss": ["val/box_loss"],
        "time": ["time"],
    }
    return next((name for name in aliases[metric] if name in frame.columns), None)


def final_metrics(frame):
    rows = []
    for (run, source), group in frame.groupby(["run", "source_csv"], sort=True):
        group = group.sort_values("epoch")
        last = group.iloc[-1]
        row = {"Run": run, "Epochs": int(group["epoch"].nunique()), "Source": source}
        for label, key in [("Precision", "precision"), ("Recall", "recall"),
                           ("mAP50", "map50"), ("mAP50-95", "map50_95"),
                           ("Train box loss", "box_loss"), ("Train cls loss", "cls_loss"),
                           ("Train DFL loss", "dfl_loss"), ("Training time (s)", "time")]:
            column = metric_column(frame, key)
            row[label] = last.get(column, float("nan")) if column else float("nan")
        row["F1"] = f1_score(row["Precision"], row["Recall"])
        row["Best mAP50-95"] = group[metric_column(frame, "map50_95")].max() if metric_column(frame, "map50_95") else float("nan")
        row["Source"] = source
        rows.append(row)
    return pd.DataFrame(rows)


def locate_plots(results_dir):
    root = Path(results_dir)
    entries = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in PLOT_EXTENSIONS:
            continue
        if "all_graphs" in path.parts:
            continue
        relative = path.relative_to(root)
        run = relative.parts[0] if len(relative.parts) > 1 else "root_level"
        name = path.name.lower()
        if "confusion_matrix" in name:
            category = "Confusion matrices"
        elif "f1_curve" in name:
            category = "F1 curves"
        elif "correlogram" in name:
            category = "Correlograms"
        elif "p_curve" in name:
            category = "Precision curves"
        elif "r_curve" in name:
            category = "Recall curves"
        elif "pr_curve" in name:
            category = "Precision-recall curves"
        elif name == "results.png":
            category = "Training and validation losses"
        elif name.startswith(("train_batch", "val_batch")):
            category = "Train/validation examples"
        elif name == "labels.jpg":
            category = "Label distributions"
        else:
            category = "Other"
        entries.append({"Run": run, "Category": category, "File": path.name,
                        "Path": path, "Relative path": relative.as_posix()})
    return pd.DataFrame(entries)


def paper_table():
    rows = []
    for name, values in PAPER.items():
        row = {"Model": name, **values}
        row["f1_derived"] = f1_score(values["precision"], values["recall"])
        rows.append(row)
    return pd.DataFrame(rows)


def read_run_args(results_dir, source_csv):
    args_path = Path(results_dir) / Path(source_csv).parent / "args.yaml"
    if not args_path.is_file():
        return {}
    try:
        with args_path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    except (OSError, yaml.YAMLError):
        return {}


st.set_page_config(page_title="Pothole Detection Results", page_icon="🕳️", layout="wide")
st.title("Pothole Detection Results Dashboard")
st.caption("Compare your YOLO experiments, inspect their plots, and review the Fortin & Llantos (2025) paper benchmark.")

if not RESULTS_DIR.exists():
    st.error(f"Results directory not found: {RESULTS_DIR}")
    st.stop()

data = load_runs(str(RESULTS_DIR))
if data.empty:
    st.error("No readable YOLO results.csv files were found under results/.")
    st.stop()

summary = final_metrics(data)
plots = locate_plots(RESULTS_DIR)
paper_df = paper_table()

with st.sidebar:
    st.header("Filters")
    all_runs = sorted(summary["Run"].unique())
    selected_runs = st.multiselect("Runs to compare", all_runs, default=all_runs)
    if not selected_runs:
        st.warning("Select at least one run.")
        st.stop()
    filtered = data[data["run"].isin(selected_runs)].copy()
    filtered_summary = summary[summary["Run"].isin(selected_runs)].copy()

best_map = filtered_summary.sort_values("mAP50", ascending=False).iloc[0]
best_f1 = filtered_summary.sort_values("F1", ascending=False).iloc[0]
best_recall = filtered_summary.sort_values("Recall", ascending=False).iloc[0]
best_precision = filtered_summary.sort_values("Precision", ascending=False).iloc[0]

cards = st.columns(4)
for card, label, record, metric in [
    (cards[0], "Best final precision", best_precision, "Precision"),
    (cards[1], "Best final recall", best_recall, "Recall"),
    (cards[2], "Best calculated F1", best_f1, "F1"),
    (cards[3], "Best final mAP50", best_map, "mAP50"),
]:
    card.metric(label, f"{record[metric]:.3f}", help=record["Run"])

overview, learning, plots_tab, paper_tab, presentation_tab = st.tabs([
    "Overview & final metrics", "Training curves", "Saved graphs", "Paper comparison",
    "Presentation guide",
])

with overview:
    st.subheader("Final epoch metrics by run")
    display = filtered_summary.drop(columns=["Source"]).sort_values("mAP50", ascending=False)
    st.dataframe(display.style.format(precision=4), use_container_width=True, hide_index=True)
    metric_options = ["Precision", "Recall", "F1", "mAP50", "mAP50-95"]
    selected_metric = st.selectbox("Compare final metric", metric_options, index=3)
    chart_data = display[["Run", selected_metric]].sort_values(selected_metric, ascending=False)
    fig = px.bar(chart_data, x="Run", y=selected_metric, color=selected_metric,
                 color_continuous_scale="Teal", title=f"Final {selected_metric} across selected runs")
    fig.update_layout(xaxis_tickangle=-35, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    st.info("F1 is calculated as 2 × precision × recall / (precision + recall) from each run's final validation metrics. The paper does not report F1 directly; its F1 values below are calculated from the paper's precision and recall table.")

with learning:
    st.subheader("Metric and loss histories")
    curve_choices = {
        "mAP50": "map50", "Precision": "precision", "Recall": "recall",
        "mAP50-95": "map50_95", "Train box loss": "box_loss",
        "Train class loss": "cls_loss", "Train DFL loss": "dfl_loss",
        "Validation box loss": "val_box_loss",
    }
    chosen = st.multiselect("Curves", list(curve_choices), default=["mAP50", "Train box loss"])
    for label in chosen:
        column = metric_column(filtered, curve_choices[label])
        if not column:
            st.warning(f"{label} is not present in the available CSVs.")
            continue
        fig = px.line(filtered, x="epoch", y=column, color="run", title=f"{label} by epoch",
                      labels={"epoch": "Epoch", column: label, "run": "Run"})
        fig.update_layout(legend_title_text="Run")
        st.plotly_chart(fig, use_container_width=True)

with plots_tab:
    st.subheader("Existing training and evaluation plots")
    if plots.empty:
        st.warning("No plot image files found.")
    else:
        categories = sorted(plots["Category"].unique())
        category = st.selectbox("Graph type", categories)
        available = plots[plots["Category"] == category]
        run_options = sorted(available["Run"].unique())
        run = st.selectbox("Run", run_options)
        choices = available[available["Run"] == run].sort_values("Relative path")
        selected_path = st.selectbox("Plot file", choices["Relative path"].tolist())
        entry = choices[choices["Relative path"] == selected_path].iloc[0]
        st.image(str(entry["Path"]), caption=f"{run} · {entry['File']}", use_container_width=True)
        st.caption(f"{len(available)} saved file(s) in this graph category across all runs.")
    if not plots.empty:
        st.markdown("**Graph coverage in saved run folders**")
        coverage = plots.groupby("Category").agg(plots=("Path", "count"), runs=("Run", "nunique")).reset_index()
        st.dataframe(coverage, hide_index=True, use_container_width=True)

with paper_tab:
    st.subheader("Reported final metrics: paper and your experiments")
    st.caption("Paper values: Fortin & Llantos, *Performance Analysis of YOLO versions for Real-time Pothole Detection*, Procedia Computer Science 257 (2025), Tables 1–4 and conclusion. Paper: YOLOv9-tiny, YOLOv10-nano, YOLOv11-nano. Your runs use YOLOv5nu, YOLOv8n, YOLOv11n/s.")
    paper_display = paper_df[["Model", "precision", "recall", "f1_derived", "map50", "box_loss", "cls_loss", "dfl_loss", "epochs"]].rename(columns={
        "precision": "Precision", "recall": "Recall", "f1_derived": "F1 (derived)",
        "map50": "mAP50", "box_loss": "Box loss", "cls_loss": "Class loss",
        "dfl_loss": "DFL loss", "epochs": "Epochs",
    })
    st.dataframe(paper_display.style.format(precision=4), use_container_width=True, hide_index=True)

    st.markdown("#### Model-to-paper metric view")
    model_filter = st.selectbox("Your run", selected_runs, index=0, key="paper_run")
    run_values = filtered_summary[filtered_summary["Run"] == model_filter].iloc[0]
    own_row = pd.DataFrame([{
        "Model": model_filter,
        "Precision": run_values["Precision"], "Recall": run_values["Recall"],
        "F1 (calculated)": run_values["F1"], "mAP50": run_values["mAP50"],
        "Box loss": run_values["Train box loss"], "Class loss": run_values["Train cls loss"],
        "DFL loss": run_values["Train DFL loss"], "Epochs": run_values["Epochs"],
    }])
    paper_rows = paper_display.rename(columns={"F1 (derived)": "F1 (calculated)"})
    combined = pd.concat([paper_rows, own_row], ignore_index=True)
    st.dataframe(combined.style.format(precision=4), use_container_width=True, hide_index=True)

    st.markdown("#### Efficiency and confusion metrics reported by the paper")
    paper_efficiency = pd.DataFrame([
        {"Model": name, "Epochs": values["epochs"],
         "Paper avg seconds/epoch": values["seconds_per_epoch"],
         "Paper training hours": {"YOLOv9-tiny": 17.8, "YOLOv10-nano": 12.3,
                                  "YOLOv11-nano": 14.5}[name],
         "True positives": values["tp"], "False positives": values["false_positives"],
         "Miss rate": values["miss_rate"]}
        for name, values in PAPER.items()
    ])
    st.dataframe(paper_efficiency.style.format({"Paper avg seconds/epoch": "{:.1f}",
                                                "Paper training hours": "{:.1f}",
                                                "Miss rate": "{:.1%}"}),
                 use_container_width=True, hide_index=True)
    selected_efficiency = filtered_summary.copy()
    selected_efficiency["Average time/epoch (s)"] = (
        selected_efficiency["Training time (s)"] / selected_efficiency["Epochs"]
    )
    selected_efficiency["Training hours"] = selected_efficiency["Training time (s)"] / 3600
    st.caption("Your time values are taken from the final cumulative `time` field in each training CSV. They are not hardware-normalized; raw TP/FP/FN counts are not stored in these CSVs.")
    st.dataframe(selected_efficiency[["Run", "Epochs", "Average time/epoch (s)",
                                      "Training hours"]].style.format(precision=2),
                 use_container_width=True, hide_index=True)

    st.markdown("#### Training configuration from this run's args.yaml")
    args = read_run_args(RESULTS_DIR, run_values["Source"])
    if args:
        configuration = pd.DataFrame([
            {"Setting": "Model", "Your run": args.get("model", "not recorded"),
             "Paper setup": "YOLOv9-tiny / YOLOv10-nano / YOLOv11-nano"},
            {"Setting": "Input size", "Your run": str(args.get("imgsz", "not recorded")),
             "Paper setup": "320 × 320"},
            {"Setting": "Batch size", "Your run": args.get("batch", "not recorded"),
             "Paper setup": 16},
            {"Setting": "Optimizer", "Your run": args.get("optimizer", "not recorded"),
             "Paper setup": "AdamW"},
            {"Setting": "Epoch limit", "Your run": args.get("epochs", "not recorded"),
             "Paper setup": "750 max; reported runs stopped at 134–393"},
            {"Setting": "Initial LR", "Your run": args.get("lr0", "not recorded"),
             "Paper setup": "Method: 0.001; Table 3: 0.0670407 (paper discrepancy)"},
            {"Setting": "Validation split", "Your run": args.get("split", "val"),
             "Paper setup": "10% validation, 10% test"},
        ])
        st.dataframe(configuration, use_container_width=True, hide_index=True)
    else:
        st.info("No args.yaml was found beside this run's results.csv.")

    st.markdown("#### Paper figure and measurement coverage")
    coverage_rows = [{"Paper item": item, "What it shows": description, "In this project": status}
                     for item, description, status in PAPER_FIGURES]
    st.dataframe(pd.DataFrame(coverage_rows), use_container_width=True, hide_index=True)
    st.markdown("#### Important comparison limits")
    st.write("The paper's models, dataset (16,054 images), 320×320 input, batch size 16, 80/10/10 split, optimizer and test protocol differ from the run artifacts here. Your CSVs provide validation curves and final training logs, but do not establish matching held-out test results, inference speed, memory usage, statistical significance, or raw confusion counts. Treat the numbers as a reference comparison, not a reproduction or controlled head-to-head result.")
    st.write("The paper reports precision, recall and mAP@0.5 in its final metrics table, plus box/class/DFL losses, epoch count, training times, and some TP/FP/miss-rate conclusions. It does not report F1 in that table; the displayed paper F1 is computed from its precision and recall. The paper also gives slightly different precision figures in its confusion-matrix narrative and final metrics table, so this dashboard uses the final metrics table for the benchmark values.")

with presentation_tab:
    st.subheader("Talk track and dashboard walkthrough")
    guide_path = BASE_DIR / "PRESENTATION_SCRIPT.md"
    if guide_path.is_file():
        st.markdown(guide_path.read_text(encoding="utf-8"))
    else:
        st.info("Presentation guide not found. See PRESENTATION_SCRIPT.md in the project folder.")

st.caption("Run artifacts are read directly from results/**/results.csv and saved graph files; no aggregate CSV regeneration is required.")
