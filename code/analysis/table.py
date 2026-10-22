# analysis/table.py
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prettytable import PrettyTable


def _parameter_counts_from_log(run_dir, model_names):
    log_path = Path(run_dir) / "_MainModeling-RUN.txt"
    if not log_path.exists():
        return {}

    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    counts = {}
    current_model = None
    ordered_names = sorted(model_names, key=len, reverse=True)
    for line in lines:
        if line.startswith('Model: "'):
            model_name = line.split('"', 2)[1]
            current_model = next(
                (name for name in ordered_names if model_name.startswith(f"{name}_")),
                None,
            )
        elif current_model and line.startswith("Total params:"):
            match = re.search(r"Total params:\s*([\d,]+)", line)
            if match:
                counts.setdefault(current_model, int(match.group(1).replace(",", "")))
            current_model = None
    return counts


def write_experiment_model_summary(
    run_dir,
    out_dir=None,
    experiment_id=None,
    model_colors=None,
    plot_show=False,
):
    """Write a compact model comparison CSV and rendered table."""
    run_dir = Path(run_dir)
    out_dir = Path(out_dir) if out_dir else run_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = run_dir / "all_repetitions.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Run metrics not found: {metrics_path}")
    metrics = pd.read_csv(metrics_path)
    if experiment_id is not None and "experiment_id" in metrics.columns:
        metrics = metrics[metrics["experiment_id"].astype(str) == str(experiment_id)]
    if metrics.empty:
        raise ValueError("No model records remain after filtering the experiment.")

    metric_summary = (
        metrics.groupby(["model_type", "utilization"], as_index=False)
        .agg(nmae=("nmae", "mean"), mae=("mae", "mean"), rmse=("rmse", "mean"))
    )
    utilizations = sorted(metric_summary["utilization"].unique())
    models = list(metric_summary["model_type"].drop_duplicates())
    summary = pd.DataFrame({"model_type": models})
    frequency_counts = metrics.groupby("model_type")["frequency"].nunique()
    summary["frequency_count"] = summary["model_type"].map(frequency_counts)
    for utilization in utilizations:
        util_rows = metric_summary[np.isclose(metric_summary["utilization"], utilization)]
        util_label = f"U{utilization * 100:g}"
        for metric in ("nmae", "mae", "rmse"):
            values = util_rows.set_index("model_type")[metric]
            summary[f"{metric}_{util_label}"] = summary["model_type"].map(values)

    run_results_path = run_dir / "run_results.csv"
    time_column = None
    if run_results_path.exists():
        run_results = pd.read_csv(run_results_path)
        if experiment_id is not None and "experiment_id" in run_results.columns:
            run_results = run_results[run_results["experiment_id"].astype(str) == str(experiment_id)]
        if "training_time_sec" in run_results.columns:
            time_column = "training_time_sec"
        elif "loop_time_sec" in run_results.columns:
            time_column = "loop_time_sec"
        if time_column:
            mean_time = pd.to_numeric(run_results[time_column], errors="coerce").groupby(run_results["model_type"]).mean()
            output_name = "mean_training_time_sec" if time_column == "training_time_sec" else "mean_loop_time_sec"
            summary[output_name] = summary["model_type"].map(mean_time)

        if "model_param_count" in run_results.columns:
            parameter_count = (
                pd.to_numeric(run_results["model_param_count"], errors="coerce")
                .groupby(run_results["model_type"])
                .max()
            )
            summary["parameter_count"] = summary["model_type"].map(parameter_count)

    latency_path = run_dir / "inference_latency.csv"
    if latency_path.exists():
        latency = pd.read_csv(latency_path)
        if experiment_id is not None and "experiment_id" in latency.columns:
            latency = latency[latency["experiment_id"].astype(str) == str(experiment_id)]
        latency = latency[(latency["source"] == "test") & (latency["mode"] == "single")]
        mean_latency = pd.to_numeric(latency["mean_per_sample_ms"], errors="coerce").groupby(latency["model_type"]).mean()
        summary["mean_single_inference_ms"] = summary["model_type"].map(mean_latency)

    if "parameter_count" not in summary or summary["parameter_count"].isna().any():
        parsed_counts = _parameter_counts_from_log(run_dir, models)
        parsed = summary["model_type"].map(parsed_counts)
        if "parameter_count" not in summary:
            summary["parameter_count"] = parsed
        else:
            summary["parameter_count"] = summary["parameter_count"].fillna(parsed)

    max_utilization = max(utilizations)
    summary = summary.sort_values(f"nmae_U{max_utilization * 100:g}", na_position="last").reset_index(drop=True)
    csv_path = out_dir / "experiment_model_summary.csv"
    summary.to_csv(csv_path, index=False)

    display_columns = ["model_type", "frequency_count"] + [f"nmae_U{u * 100:g}" for u in utilizations]
    if "mean_single_inference_ms" in summary:
        display_columns.append("mean_single_inference_ms")
    display_columns.append("mean_training_time_sec" if "mean_training_time_sec" in summary else "mean_loop_time_sec")
    display_columns.append("parameter_count")
    display = summary[display_columns].copy()
    column_labels = ["Model", "Freqs"] + [f"NMAE\nU={u * 100:g}%" for u in utilizations]
    if "mean_single_inference_ms" in display_columns:
        column_labels.append("Single inference\n(ms)")
    column_labels.append("Mean training\n(s)" if "mean_training_time_sec" in display_columns else "Mean train/eval loop\n(s)")
    column_labels.append("Parameters")

    formatted_rows = []
    for _, row in display.iterrows():
        values = [str(row["model_type"])]
        values.append(str(int(row["frequency_count"])))
        values.extend(f"{row[column]:.5f}" if pd.notna(row[column]) else "-" for column in display_columns if column.startswith("nmae_"))
        if "mean_single_inference_ms" in display_columns:
            value = row["mean_single_inference_ms"]
            values.append(f"{value:.2f}" if pd.notna(value) else "-")
        time_name = "mean_training_time_sec" if "mean_training_time_sec" in display_columns else "mean_loop_time_sec"
        value = row[time_name]
        values.append(f"{value:.1f}" if pd.notna(value) else "-")
        value = row["parameter_count"]
        values.append(f"{int(value):,}" if pd.notna(value) else "-")
        formatted_rows.append(values)

    fig_width = max(10.5, 1.55 * len(column_labels))
    fig_height = max(3.2, 0.48 * len(formatted_rows) + 1.8)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    table = ax.table(cellText=formatted_rows, colLabels=column_labels, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.65)
    for column in range(len(column_labels)):
        table[(0, column)].set_facecolor("#333333")
        table[(0, column)].set_text_props(color="white", weight="bold")
    for row_index, model in enumerate(display["model_type"], start=1):
        table[(row_index, 0)].set_text_props(weight="bold")
        if model_colors and model in model_colors:
            table[(row_index, 0)].set_facecolor(model_colors[model])
            table[(row_index, 0)].set_text_props(color="white", weight="bold")
    for offset, utilization in enumerate(utilizations, start=2):
        column = f"nmae_U{utilization * 100:g}"
        best_index = summary[column].idxmin() + 1
        table[(best_index, offset)].set_facecolor("#DFF0D8")
        table[(best_index, offset)].set_text_props(weight="bold")

    for column_name in ("mean_single_inference_ms", "mean_training_time_sec", "mean_loop_time_sec"):
        if column_name not in display_columns or summary[column_name].dropna().empty:
            continue
        table_column = display_columns.index(column_name)
        best_index = summary[column_name].idxmin() + 1
        table[(best_index, table_column)].set_facecolor("#DFF0D8")
        table[(best_index, table_column)].set_text_props(weight="bold")

    timing_note = "Mean fit time" if time_column == "training_time_sec" else "Legacy runs report the complete train/evaluate loop time"
    ax.set_title(f"Experiment Model Summary\n{timing_note}", fontsize=13, fontweight="bold", pad=14)
    fig.tight_layout()
    image_path = out_dir / "experiment_model_summary.png"
    fig.savefig(image_path, dpi=220, bbox_inches="tight", facecolor="white")
    if plot_show:
        plt.show()
    plt.close(fig)
    return str(csv_path), str(image_path)

def print_results_table(
        csv_path: Path,
        experiment_id: str | None = None,
        model_type: str | None = None,
    ):
    df = pd.read_csv(csv_path)
    # Optional filters
    if experiment_id is not None:
        df = df[df["experiment_id"].astype(str) == str(experiment_id)]
    if model_type is not None:
        df = df[df["model_type"] == model_type]
    if df.empty:
        print("No data after filtering.")
        return
    # Aggregate exactly like experiment.py table:
    # repetition -> mean per (freq, util)
    agg = (
        df.groupby(["model_type", "frequency", "utilization"], as_index=False)
        .agg(
            mae=("mae", "mean"),
            mape=("mape", "mean"),
            rmse=("rmse", "mean"),
            nmae=("nmae", "mean"),
        )
    )
    for model in agg["model_type"].unique():
        sub = agg[agg["model_type"] == model]
        table = PrettyTable(["Freq (Hz)", "Utilization (%)", "MAE", "MAPE (%)", "RMSE", "NMAE"])
        table.title = f"{model} Results"
        for _, row in sub.sort_values(["frequency", "utilization"]).iterrows():
            # Later, find where should I add the below average row:
            # table_row = [f"Average", f"Only 100% Util", f"{np.mean(state.metric_mae):.4f}", f"{np.mean(state.metric_mape):.2f} %", f"{np.mean(state.metric_rmse):.4f}"]
            # state.table.add_row(table_row)
            table.add_row([f"{row.frequency:.0f}", f"{row.utilization * 100:.0f}", f"{row.mae:.4f}", f"{row.mape:.2f}", f"{row.rmse:.4f}", f"{row.nmae:.4f}"])
        print("\n", table, "\n")
