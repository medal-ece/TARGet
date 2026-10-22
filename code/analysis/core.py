# analysis/core.py
from pathlib import Path
from typing import Union
import numpy as np
import pandas as pd
from config import Config

META_COLS = {
    "timestamp", "experiment_id", "schema_version", "user",
    "model_type", "architecture", "coordinate", "topology",
    "dataset_version", "dataset_variant", "train_dataset_split", "heldout_dataset_split",
    "frequency", "utilization", "repetition",
    "seed", "optimizer", "learning_rate",
    "batch_size", "scaler", "loss", "loss_function",
    "adapter_mode", "metrics",
    "training_time_sec", "model_param_count",

}

def load_runs(csv_path: Union[str, Path]) -> pd.DataFrame:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    metric_cols = [c for c in df.columns if c not in META_COLS]
    if not metric_cols:
        raise ValueError("No metric columns found in CSV.")
    # Basic sanity checks
    required = {"model_type", "frequency", "utilization", "repetition"} | set(metric_cols)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns in {csv_path.name}: {sorted(missing)}\n"
            f"Found columns: {list(df.columns)}"
        )
    # Normalize dtypes
    df["frequency"] = pd.to_numeric(df["frequency"], errors="coerce")
    df["utilization"] = pd.to_numeric(df["utilization"], errors="coerce")
    df["repetition"] = pd.to_numeric(df["repetition"], errors="coerce")
    for m in metric_cols:
        df[m] = pd.to_numeric(df[m], errors="coerce")
    df = df.dropna(subset=["frequency", "utilization", "repetition"] + metric_cols)
    return df


def filter_df(
        df: pd.DataFrame,
        experiment_id: str | None,
        model_type: str | None,
        freq: float | None,
        util: float | None,
    ) -> pd.DataFrame:
    out = df.copy()
    # experiment_id is optional (only if you log it)
    if experiment_id is not None:
        if "experiment_id" not in out.columns:
            raise ValueError("experiment_id filtering requested but column 'experiment_id' not present in CSV.")
        out = out[out["experiment_id"].astype(str) == str(experiment_id)]
    if model_type is not None:
        out = out[out["model_type"].astype(str) == str(model_type)]
    if freq is not None:
        out = out[np.isclose(out["frequency"].astype(float), float(freq))]
    if util is not None:
        out = out[np.isclose(out["utilization"].astype(float), float(util))]
    return out


def aggregate_metrics(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """
    Returns long-form aggregated table with columns:
      group_cols + metric + mean + std + best + n
    """
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
    # Consistent ordering
    sort_cols = group_cols + ["metric"]
    res = res.sort_values(sort_cols).reset_index(drop=True)
    return res

# def aggregate_like_experiment_py(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Reproduces the logic from experiment.py:
#     repetition → mean per (model, freq, util)
#     then
#     frequency → mean per (model, util)
#     """
#     # 1) Average over repetitions
#     per_freq = (
#         df.groupby(["model_type", "frequency", "utilization"], as_index=False)
#           .agg({m: "mean" for m in METRICS})
#     )
#     # 2) Average over frequencies
#     per_util = (
#         per_freq.groupby(["model_type", "utilization"], as_index=False)
#                 .agg({m: "mean" for m in METRICS})
#     )
#     return per_util


def pivot_metric(agg_long: pd.DataFrame, metric: str, value_col: str = "mean") -> pd.DataFrame:
    """
    Convert long-form agg into a pivoted matrix:
      index: frequency
      columns: utilization
      values: e.g. mean
    Only works if agg_long includes frequency+utilization in its grouping.
    """
    sub = agg_long[agg_long["metric"] == metric].copy()
    if not {"frequency", "utilization"} <= set(sub.columns):
        raise ValueError("pivot_metric requires grouping that includes 'frequency' and 'utilization'.")
    mat = sub.pivot(index="frequency", columns="utilization", values=value_col)
    mat = mat.sort_index().sort_index(axis=1)
    return mat
