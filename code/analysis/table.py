# code/analysis/table.py
"""Console-friendly summary tables for experiment results."""
from pathlib import Path
import pandas as pd
from prettytable import PrettyTable

def print_results_table(
        csv_path: Path,
        experiment_id: str | None = None,
        model_type: str | None = None,
    ):
    """Print per-model aggregated metrics grouped by frequency and utilization."""
    df = pd.read_csv(csv_path)
    # Apply optional filters before aggregating.
    if experiment_id is not None:
        df = df[df["experiment_id"].astype(str) == str(experiment_id)]
    if model_type is not None:
        df = df[df["model_type"] == model_type]
    if df.empty:
        print("No data after filtering.")
        return
    # Match the experiment summary by averaging repetitions first.
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
            table.add_row([f"{row.frequency:.0f}", f"{row.utilization * 100:.0f}", f"{row.mae:.4f}", f"{row.mape:.2f}", f"{row.rmse:.4f}", f"{row.nmae:.4f}"])
        print("\n", table, "\n")
