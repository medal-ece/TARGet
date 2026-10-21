# code/analysis/core.py
"""Core helpers for loading and aggregating experiment results."""
from pathlib import Path
from typing import Union
import pandas as pd

META_COLS = {
    "timestamp", "experiment_id", "schema_version", "user",
    "model_type", "architecture", "coordinate", "topology",
    "frequency", "utilization", "repetition",
    "seed", "optimizer", "learning_rate",
    "batch_size", "scaler", "loss",
    "adapter_mode"

}

def load_runs(csv_path: Union[str, Path]) -> pd.DataFrame:
    """Load a run log CSV, validate its schema, and normalize metric dtypes."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    metric_cols = [c for c in df.columns if c not in META_COLS]
    if not metric_cols:
        raise ValueError("No metric columns found in CSV.")
    # Ensure the CSV has the minimum columns needed for downstream analysis.
    required = {"model_type", "frequency", "utilization", "repetition"} | set(metric_cols)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns in {csv_path.name}: {sorted(missing)}\n"
            f"Found columns: {list(df.columns)}"
        )
    # Normalize columns that analysis expects to be numeric.
    df["frequency"] = pd.to_numeric(df["frequency"], errors="coerce")
    df["utilization"] = pd.to_numeric(df["utilization"], errors="coerce")
    df["repetition"] = pd.to_numeric(df["repetition"], errors="coerce")
    for m in metric_cols:
        df[m] = pd.to_numeric(df[m], errors="coerce")
    df = df.dropna(subset=["frequency", "utilization", "repetition"] + metric_cols)
    return df


def aggregate_metrics(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Aggregate metric columns into a long-form table with mean, std, best, and count."""
    metric_cols = [c for c in df.columns if c not in META_COLS]
    if not metric_cols:
        raise ValueError("No metric columns found in CSV.")
    rows = []
    for metric in metric_cols:
        g = df.groupby(group_cols)[metric]
        agg = g.agg(["mean", "std", "min", "count"]).reset_index()
        agg = agg.rename(columns={"min": "best", "count": "n"})
        agg.insert(len(group_cols), "metric", metric)
        rows.append(agg)

    res = pd.concat(rows, ignore_index=True)
    # Keep output ordering stable for plots and saved CSVs.
    sort_cols = group_cols + ["metric"]
    res = res.sort_values(sort_cols).reset_index(drop=True)
    return res


def pivot_metric(agg_long: pd.DataFrame, metric: str, value_col: str = "mean") -> pd.DataFrame:
    """Pivot one metric from the long-form table into a frequency-by-utilization matrix."""
    sub = agg_long[agg_long["metric"] == metric].copy()
    if not {"frequency", "utilization"} <= set(sub.columns):
        raise ValueError("pivot_metric requires grouping that includes 'frequency' and 'utilization'.")
    mat = sub.pivot(index="frequency", columns="utilization", values=value_col)
    mat = mat.sort_index().sort_index(axis=1)
    return mat
