# analysis/plots.py
import math
import colorsys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from prettytable import PrettyTable

from analysis.core import load_runs, aggregate_metrics

def color_variants(base_color, lightness_factors=[1.0, 1.35, 0.7]):
    """
    Generate visually close colors from a base color.
    base_color : str or RGB
        Matplotlib color (e.g. 'tab:blue')
    lightness_factors : list[float]
        Values <1 darker, >1 lighter (typical range: 0.6–1.4)
    """
    r, g, b = mcolors.to_rgb(base_color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    colors = []
    for f in lightness_factors:
        l_new = max(0, min(1, l * f))
        colors.append(colorsys.hls_to_rgb(h, l_new, s))
    return colors

def auto_subplot_grid(n_plots, max_cols=3):
    """
    Compute a near-square (nrows, ncols) layout for n_plots,
    preferring compact grids and avoiding empty columns.
    """
    ncols = min(max_cols, math.ceil(math.sqrt(n_plots)))
    nrows = math.ceil(n_plots / ncols)
    return nrows, ncols


def plot_utilization_curves_per_freq(
    agg_long: pd.DataFrame,
    out_dir: Path,
    title_suffix: str = "",
    MODEL_BASE_COLORS=None,
):
    """
    For each frequency and metric:
      plot metric vs utilization
      with mean and best curves per model_type.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    required = {"model_type", "frequency", "utilization", "metric", "mean", "best"}
    if not required <= set(agg_long.columns):
        raise ValueError(f"Missing required columns: {required - set(agg_long.columns)}")

    for metric in sorted(agg_long["metric"].unique()):
        for freq in sorted(agg_long["frequency"].unique()):
            plt.figure(figsize=(8, 4))
            sub = agg_long[
                (agg_long["metric"] == metric) &
                (agg_long["frequency"] == freq)
            ]
            for model in sorted(sub["model_type"].unique()):
                mean, best, _ = color_variants(MODEL_BASE_COLORS[model])
                d = (sub[sub["model_type"] == model].sort_values("utilization"))
                x = d["utilization"] * 100
                plt.plot(x, d["mean"], marker="o", linestyle="-", color=mean, label=f"{model} (mean)")
                plt.plot(x, d["best"], marker="x", linestyle="--", color=best, label=f"{model} (best)")
            plt.xlabel("Utilization (%)")
            plt.ylabel(metric.upper())
            n_rep = int(sub["n"].iloc[0])
            plt.title(f"{metric.upper()} vs Utilization @ {freq/1e9:.2f} GHz (n_rep={n_rep}) {title_suffix}".strip())
            plt.suptitle(f"n_rep={n_rep} {title_suffix}", fontsize=10, y=0.9)
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=9)
            plt.tight_layout()
            out_path = out_dir / f"{str(freq / 1e9)} GHz_{metric.upper()}.png"
            plt.savefig(out_path, dpi=200)
            plt.close()


# def plot_utilization_curves_total(
#     agg_long: pd.DataFrame,
#     out_dir: Path,
#     title_suffix: str = "",
#     MODEL_BASE_COLORS=None,
# ):
#     """
#     Aggregate metrics over frequency, then plot vs utilization.
#     """
#     out_dir.mkdir(parents=True, exist_ok=True)
#     required = {"model_type", "frequency", "utilization", "metric", "mean", "best"}
#     if not required <= set(agg_long.columns):
#         raise ValueError(f"Missing required columns: {required - set(agg_long.columns)}")
#     # ---- average over frequency ----
#     avg = (
#         agg_long
#         .groupby(["model_type", "utilization", "metric"], as_index=False)
#         .agg(
#             mean_over_freq=("mean", "mean"),
#             # Creates a new column called "best_over_freq" by applying
#             # the aggregation function "mean" to the column "best", within each group.
#             best_over_freq_avg=("best", "mean"),
#             best_over_freq_best=("best", "min"),
#         )
#     )
#     n_rep = int(agg_long["n"].iloc[0])
#     # ---- plotting ----
#     for metric in sorted(avg["metric"].unique()):
#         plt.figure(figsize=(8, 4))
#         sub = avg[avg["metric"] == metric]
#         for model in sorted(sub["model_type"].unique()):
#             mean_c, avg_best_c, best_best_c = color_variants(MODEL_BASE_COLORS[model])
#             d = sub[sub["model_type"] == model].sort_values("utilization")
#             x = d["utilization"] * 100
#             plt.plot(x, d["mean_over_freq"], marker="o", linestyle="-", color=mean_c, alpha=1.0, label=f"{model} (mean)")
#             plt.plot(x, d["best_over_freq_avg"], marker="x", linestyle="--", color=avg_best_c, alpha=0.6, label=f"{model} (Average of best)")
#             plt.plot(x, d["best_over_freq_best"], marker="*", linestyle="-.", color=best_best_c, alpha=0.3, label=f"{model} (Best of best)")
#         plt.xlabel("Utilization (%)")
#         plt.ylabel(metric.upper())
#         plt.title(f"{metric.upper()} vs Utilization (avg over freq) (n_rep={n_rep}) {title_suffix}".strip())
#         plt.grid(True, alpha=0.3)
#         plt.legend(fontsize=9)
#         plt.tight_layout()
#         out_path = out_dir / f"{metric}_util_avg_over_freq.png"
#         plt.savefig(out_path, dpi=200)
#         plt.close()
#     avg.to_csv(out_dir / "aggregated_metrics_avg_over_freq.csv", index=False)


def plot_all_metrics_per_freq(agg_long: pd.DataFrame, out_dir: Path, title_suffix: str = "", MODEL_BASE_COLORS=None, coord_system: str = "Unknown"):
    out_dir.mkdir(parents=True, exist_ok=True)
    required = {"model_type", "frequency", "utilization", "metric", "mean", "best", "n"}
    if not required <= set(agg_long.columns):
        raise ValueError(f"Missing required columns: {required - set(agg_long.columns)}")
    metrics = sorted(agg_long["metric"].unique())
    for freq in sorted(agg_long["frequency"].unique()):
        sub_f = agg_long[agg_long["frequency"] == freq]
        n_rep = int(sub_f["n"].iloc[0])
        nrows, ncols = auto_subplot_grid(len(metrics))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.6 * nrows), sharex=True)
        axes = axes.flatten() if len(metrics) > 1 else [axes]
        for ax, metric in zip(axes, metrics):
            sub = sub_f[sub_f["metric"] == metric]
            for model in sorted(sub["model_type"].unique()):
                mean_c, best_c, _ = color_variants(MODEL_BASE_COLORS[model])
                d = sub[sub["model_type"] == model].sort_values("utilization")
                x = d["utilization"] * 100
                ax.plot(x, d["mean"], marker="o", linestyle="-", color=mean_c, label=f"{model} (mean)")
                ax.plot(x, d["best"], marker="x", linestyle="--", color=best_c, label=f"{model} (best)")
            ax.set_title(metric.upper())
            ax.grid(True, alpha=0.3)
            ax.set_ylabel(metric.upper())
        # Remove unused axes
        for ax in axes[len(metrics):]:
            ax.remove()
        # Global labels and title
        fig.supxlabel("Utilization (%)")
        fig.suptitle(f"Metrics vs Utilization @ {freq/1e9:.2f} GHz (n_rep={n_rep}) {title_suffix}".strip(), fontsize=14)
        # Single legend
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=len(labels), frameon=False,)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        out_path = out_dir / f"all_metrics_{freq/1e9:.2f}GHz.png"
        fig.savefig(out_path, dpi=200)
        plt.close(fig)


def plot_utilization_curves_total(agg_long: pd.DataFrame, out_dir: Path, title_suffix: str = "", MODEL_BASE_COLORS=None, coord_system: str = "Unknown", plot_show: bool = False, plot_m_avg=True, plot_m_best=False, plot_b_avg=False, plot_b_best=False):
    """
    Aggregate metrics over frequency, then plot ALL metrics
    in one figure with multiple subplots (vs utilization).
    """
    # ---- frequency string ----
    freq_list = sorted(agg_long["frequency"].unique())
    freq_string = "Freqs=[" + ",".join(f"{freq/1e9:.2f}GHz" for freq in freq_list) + "]"
    out_dir.mkdir(parents=True, exist_ok=True)
    required = {"model_type", "frequency", "utilization", "metric", "mean", "best", "n"}
    if not required <= set(agg_long.columns):
        raise ValueError(f"Missing required columns: {required - set(agg_long.columns)}")
    # ---- aggregate over frequency ----
    avg = (
        agg_long
        .groupby(["model_type", "utilization", "metric"], as_index=False)
        .agg(
            mean_over_freq=("mean", "mean"),
            # Creates a new column called "best_over_freq" by applying
            # the aggregation function "mean" to the column "best", within each group.
            mean_over_freq_best=("mean", "min"),
            best_over_freq_avg=("best", "mean"),
            best_over_freq_best=("best", "min"),
        )
    )
    metrics = sorted(avg["metric"].unique())
    n_rep = int(agg_long["n"].iloc[0])
    # ---- subplot grid ----
    nrows, ncols = auto_subplot_grid(len(metrics))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 3.6 * nrows), sharex=True)
    axes = axes.flatten() if len(metrics) > 1 else [axes]
    # ---- plotting ----
    for ax, metric in zip(axes, metrics):
        sub = avg[avg["metric"] == metric]
        for model in sorted(sub["model_type"].unique()):
            mean_c, avg_best_c, best_best_c = color_variants(MODEL_BASE_COLORS[model])
            d = sub[sub["model_type"] == model].sort_values("utilization")
            x = d["utilization"] * 100
            if plot_m_avg:
                ax.plot(x, d["mean_over_freq"], marker="o", linestyle="-", color=mean_c, label=f"{model} (mean)")
            if plot_m_best:
                ax.plot(x, d["mean_over_freq_best"], marker="*", linestyle="-.", color=best_best_c, label=f"{model} (best of best)")
            if plot_b_avg:
                ax.plot(x, d["best_over_freq_avg"], marker="x", linestyle="--", color=avg_best_c, label=f"{model} (avg of best)")
            if plot_b_best:
                ax.plot(x, d["best_over_freq_best"], marker="*", linestyle="-.", color=best_best_c, label=f"{model} (best of best)")
        ax.set_title(metric.upper())
        ax.set_ylabel(metric.upper())
        ax.grid(True, alpha=0.3)
    # ---- remove unused axes ----
    for ax in axes[len(metrics):]:
        ax.remove()
    # ---- global labels & title ----
    fig.supxlabel("Utilization (%)")
    fig.suptitle(f"Metrics vs Utilization | avg over freqs | {freq_string} | {coord_system} | n_rep={n_rep} {title_suffix}".strip(), fontsize=14)
    # ---- single legend (no overlap) ----
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.88), ncol=min(4, len(labels)), frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.80])
    if plot_show:
        plt.show()
    out_path = out_dir / f"all_metrics_util_avg_over_freq_{coord_system}.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    # ---- save numeric table ----
    avg.to_csv(out_dir / "aggregated_metrics_avg_over_freq.csv", index=False)


def plot_all_metrics_per_util(agg_long: pd.DataFrame, out_dir: Path, util: float = 1.0, title_suffix: str = "", MODEL_BASE_COLORS=None, coord_system: str = "Unknown", prefix: str = "errbyfreq",):
    """
    Plot ALL metrics vs frequency at a fixed utilization (e.g. 100%).
    One figure per utilization.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    required = {"model_type", "frequency", "utilization", "metric", "mean", "best", "n"}
    if not required <= set(agg_long.columns):
        raise ValueError(f"Missing required columns: {required - set(agg_long.columns)}")
    # ---- filter utilization ----
    sub_u = agg_long[agg_long["utilization"] == util]
    if sub_u.empty:
        raise ValueError(f"No data for utilization={util}")
    metrics = sorted(sub_u["metric"].unique())
    freqs = sorted(sub_u["frequency"].unique())
    n_rep = int(sub_u["n"].iloc[0])
    # ---- subplot grid ----
    nrows, ncols = auto_subplot_grid(len(metrics))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 3.6 * nrows), sharex=True,)
    axes = axes.flatten() if len(metrics) > 1 else [axes]
    # ---- plotting ----
    for ax, metric in zip(axes, metrics):
        sub_m = sub_u[sub_u["metric"] == metric]
        for model in sorted(sub_m["model_type"].unique()):
            mean_c, _, _ = color_variants(MODEL_BASE_COLORS[model])
            d = (sub_m[sub_m["model_type"] == model].sort_values("frequency"))
            x = d["frequency"] / 1e9  # GHz
            # ax.plot(x, d["mean"], marker="o", linestyle="-", color=mean_c, label=model,)
            y = d["mean"].values
            ax.plot(x, y, marker="o", linestyle="-", color=mean_c, label=model)
            ax.set_xticks(x)
            ax.set_xticklabels([f"{v:.2g}" for v in x], rotation=45)
            for xi, yi in zip(x, y):
                ax.annotate(f"{yi:.3f}", (xi, yi), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8, alpha=0.85,)
        ax.set_title(metric.upper())
        ax.set_ylabel(metric.upper())
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log")
    # ---- remove unused axes ----
    for ax in axes[len(metrics):]:
        ax.remove()
    # ---- global labels & title ----
    fig.supxlabel("Frequency (GHz)")
    fig.suptitle(f"Error vs Frequency @ {int(util*100)}% Utilization | {coord_system} | n_rep={n_rep} {title_suffix}".strip(), fontsize=14,)
    # ---- legend ----
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.92), ncol=min(4, len(labels)), frameon=False,)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    out_path = out_dir / f"{prefix}_U{int(util*100)}_{coord_system}.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_y_vs_frequency_from_predictions(
    pred_csv_path: Path,
    out_dir: Path,
    experiment_id: str | None = None,
    util: float = 1.0,
    title_suffix: str = "",
    MODEL_BASE_COLORS=None,
    plot_show: bool = False,
    prefix: str = "ybyfreq",
):
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(pred_csv_path)
    required = {"model_type", "frequency", "utilization", "sample_id", "y_true", "y_pred"}
    if not required <= set(df.columns):
        raise ValueError(f"Missing required columns: {required - set(df.columns)}")
    if experiment_id is not None and "experiment_id" in df.columns:
        df = df[df["experiment_id"].astype(str) == str(experiment_id)]
    if df.empty:
        raise ValueError("No data left after filtering.")
    coord_system = df["coordinate"].iloc[0] if "coordinate" in df.columns else "Unknown"
    sub = df[np.isclose(df["utilization"].astype(float), float(util))].copy()
    if sub.empty:
        raise ValueError(f"No data for utilization={util}")
    pred_agg = (sub.groupby(["model_type", "frequency"], as_index=False).agg(y_pred_mean=("y_pred", "mean"), y_pred_std=("y_pred", "std"), n_pred=("y_pred", "count"),).sort_values(["model_type", "frequency"]))
    true_agg = (sub.groupby(["frequency"], as_index=False).agg(y_true_mean=("y_true", "mean"),y_true_std=("y_true", "std"), n_true=("y_true", "count"),).sort_values("frequency"))
    plt.figure(figsize=(8.5, 5.0))
    x_true = true_agg["frequency"] / 1e9
    y_true = true_agg["y_true_mean"].values
    plt.plot(x_true, y_true, marker="s", linestyle="-", linewidth=2.2, label="Real y")
    for xi, yi in zip(x_true, y_true):
        plt.annotate(f"{yi:.3f}", (xi, yi), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8, alpha=0.85,)
    for model in sorted(pred_agg["model_type"].unique()):
        d = pred_agg[pred_agg["model_type"] == model].sort_values("frequency")
        x = d["frequency"] / 1e9
        y = d["y_pred_mean"].values
        color = None
        if MODEL_BASE_COLORS is not None and model in MODEL_BASE_COLORS:
            color, _, _ = color_variants(MODEL_BASE_COLORS[model])
        plt.plot(x, y, marker="o", linestyle="--", color=color, label=model)
        for xi, yi in zip(x, y):
            plt.annotate(f"{yi:.3f}", (xi, yi), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8, alpha=0.85,)

    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Average y")
    plt.title(f"Average y vs Frequency @ {int(util*100)}% Utilization | {coord_system} {title_suffix}".strip())
    plt.grid(True, alpha=0.3)
    plt.xscale("log")
    plt.xticks(x_true, [f"{v:.2g}" for v in x_true], rotation=45)
    plt.legend()
    plt.tight_layout()
    out_path = out_dir / f"{prefix}_U{int(util*100)}_{coord_system}.png"
    plt.savefig(out_path, dpi=200)
    if plot_show:
        plt.show()
    plt.close()
    pred_agg.to_csv(out_dir / f"{prefix}_pred_aggregated.csv", index=False)
    true_agg.to_csv(out_dir / f"{prefix}_true_aggregated.csv", index=False)


def plot_tolerance_heatmaps_from_predictions(pred_csv_path: Path, out_dir: Path, experiment_id: str | None = None, util: float = 1.0, thresholds: list[float] | None = None, title_suffix: str = "", plot_show: bool = False, prefix: str = "tolheat",):
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(pred_csv_path)
    required = {"model_type", "frequency", "utilization", "y_true", "y_pred"}
    if not required <= set(df.columns):
        raise ValueError(f"Missing required columns: {required - set(df.columns)}")
    if thresholds is None:
        thresholds = [0.01, 0.02, 0.05, 0.10]
    if len(thresholds) != 4:
        raise ValueError("This figure expects exactly 4 thresholds so it can render a 2x2 subplot layout.")
    if experiment_id is not None and "experiment_id" in df.columns:
        df = df[df["experiment_id"].astype(str) == str(experiment_id)]
    if df.empty:
        raise ValueError("No data left after filtering.")
    coord_system = df["coordinate"].iloc[0] if "coordinate" in df.columns else "Unknown"
    sub = df[np.isclose(df["utilization"].astype(float), float(util))].copy()
    if sub.empty:
        raise ValueError(f"No data for utilization={util}")
    # ---------------------------------------------------------
    # Per-frequency normalization for sample-level NMAE-style error
    # ---------------------------------------------------------
    freq_range = (sub.groupby("frequency", as_index=False).agg(y_min=("y_true", "min"), y_max=("y_true", "max"),))
    freq_range["y_range"] = freq_range["y_max"] - freq_range["y_min"]
    sub = sub.merge(freq_range[["frequency", "y_range"]], on="frequency", how="left")
    sub["abs_err"] = (sub["y_true"] - sub["y_pred"]).abs()
    # NMAE-style normalized absolute error at sample level
    sub["norm_abs_err"] = np.where(sub["y_range"] > 0, sub["abs_err"] / sub["y_range"], np.nan)
    detail_cols = ["model_type", "frequency", "utilization", "repetition", "sample_id", "y_true", "y_pred", "abs_err", "y_range", "norm_abs_err"]
    detail_cols = [c for c in detail_cols if c in sub.columns]
    sub[detail_cols].to_csv(out_dir / f"{prefix}_detail.csv", index=False)
    # ---------------------------------------------------------
    # Build aggregated heatmap matrices for each threshold
    # ---------------------------------------------------------
    mats = []
    summary_rows = []
    for thr in thresholds:
        hit_col = f"hit_le_{thr:.4f}"
        sub[hit_col] = (sub["norm_abs_err"] <= thr).astype(float)
        agg = (sub.groupby(["model_type", "frequency"], as_index=False).agg(hit_rate=(hit_col, "mean"), n_valid=("norm_abs_err", lambda s: s.notna().sum()), mean_nerr=("norm_abs_err", "mean"), median_nerr=("norm_abs_err", "median"),).sort_values(["model_type", "frequency"]))
        agg["threshold_nmae"] = thr
        agg["freq_ghz"] = agg["frequency"] / 1e9
        summary_rows.append(agg)
        heat = (agg.pivot(index="model_type", columns="freq_ghz", values="hit_rate").sort_index())
        mats.append((thr, heat))
    summary_df = pd.concat(summary_rows, ignore_index=True)
    summary_df.to_csv(out_dir / f"{prefix}_summary.csv", index=False)
    compact = (summary_df.groupby(["threshold_nmae", "model_type"], as_index=False).agg(mean_hit_rate=("hit_rate", "mean"), median_hit_rate=("hit_rate", "median"), min_hit_rate=("hit_rate", "min"), max_hit_rate=("hit_rate", "max"), mean_nerr=("mean_nerr", "mean"),).sort_values(["threshold_nmae", "mean_hit_rate"], ascending=[True, False]))
    compact.to_csv(out_dir / f"{prefix}_compact_summary.csv", index=False)
    # ---------------------------------------------------------
    # One figure with 4 heatmaps (2x2)
    # ---------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(16, 8.5), sharex=False, sharey=False)
    axes = axes.ravel()
    for ax, (thr, heat) in zip(axes, mats):
        im = ax.imshow(heat.values, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0, interpolation="nearest",)
        ax.set_yticks(np.arange(heat.shape[0]))
        ax.set_yticklabels(heat.index)
        xvals = heat.columns.values
        n_freq = len(xvals)
        step = max(1, n_freq // 10)
        xtick_idx = np.arange(0, n_freq, step)
        ax.set_xticks(xtick_idx)
        ax.set_xticklabels([f"{xvals[i]:.2g}" for i in xtick_idx], rotation=45, ha="right")
        ax.set_xlabel("Frequency (GHz)")
        ax.set_ylabel("Model")
        ax.set_title(f"NMAE tolerance ≤ {thr:.3f}")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Hit rate")
    main_title = f"Tolerance Heatmaps | {int(util*100)}% Utilization | {coord_system}"
    if title_suffix:
        main_title += f" | {title_suffix}"
    fig.suptitle(main_title, fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_dir / f"{prefix}_all4_U{int(util*100)}_{coord_system}.png", dpi=220)
    if plot_show:
        plt.show()
    plt.close(fig)


def plot_tolerance_curves_from_predictions(pred_csv_path: Path, out_dir: Path, experiment_id: str | None = None, util: float = 1.0, threshold_grid: np.ndarray | None = None, title_suffix: str = "", MODEL_BASE_COLORS=None, plot_show: bool = False, prefix: str = "tolcurve",):
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(pred_csv_path)
    required = {"model_type", "frequency", "utilization", "y_true", "y_pred"}
    if not required <= set(df.columns):
        raise ValueError(f"Missing required columns: {required - set(df.columns)}")
    if threshold_grid is None:
        threshold_grid = np.linspace(0.0, 0.15, 61)   # 0.000, 0.0025, ..., 0.15
    if experiment_id is not None and "experiment_id" in df.columns:
        df = df[df["experiment_id"].astype(str) == str(experiment_id)]
    if df.empty:
        raise ValueError("No data left after filtering.")
    coord_system = df["coordinate"].iloc[0] if "coordinate" in df.columns else "Unknown"
    sub = df[np.isclose(df["utilization"].astype(float), float(util))].copy()
    if sub.empty:
        raise ValueError(f"No data for utilization={util}")
    # ---------------------------------------------------------
    # Per-frequency normalization
    # ---------------------------------------------------------
    freq_range = (sub.groupby("frequency", as_index=False).agg(y_min=("y_true", "min"), y_max=("y_true", "max"),))
    freq_range["y_range"] = freq_range["y_max"] - freq_range["y_min"]
    sub = sub.merge(freq_range[["frequency", "y_range"]], on="frequency", how="left")
    sub["abs_err"] = (sub["y_true"] - sub["y_pred"]).abs()
    sub["norm_abs_err"] = np.where(sub["y_range"] > 0, sub["abs_err"] / sub["y_range"], np.nan)
    sub_valid = sub[sub["norm_abs_err"].notna()].copy()
    if sub_valid.empty:
        raise ValueError("All normalized errors are NaN. Check target range at each frequency.")
    rows = []
    auc_rows = []
    models = sorted(sub_valid["model_type"].unique())
    for model in models:
        errs = np.sort(sub_valid.loc[sub_valid["model_type"] == model, "norm_abs_err"].values)
        if len(errs) == 0:
            continue
        hit_rates = [(errs <= thr).mean() for thr in threshold_grid]
        for thr, hr in zip(threshold_grid, hit_rates):
            rows.append({"model_type": model, "threshold_nmae": float(thr), "hit_rate": float(hr),})
        auc = np.trapz(hit_rates, threshold_grid) / (threshold_grid.max() - threshold_grid.min())
        auc_rows.append({"model_type": model, "auc_normalized": float(auc), "median_nerr": float(np.median(errs)), "mean_nerr": float(np.mean(errs)), "p90_nerr": float(np.quantile(errs, 0.90)),})
    curve_df = pd.DataFrame(rows)
    auc_df = pd.DataFrame(auc_rows).sort_values("auc_normalized", ascending=False)
    curve_df.to_csv(out_dir / f"{prefix}_curve_data.csv", index=False)
    auc_df.to_csv(out_dir / f"{prefix}_summary.csv", index=False)
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    for model in models:
        d = curve_df[curve_df["model_type"] == model].sort_values("threshold_nmae")
        color = None
        if MODEL_BASE_COLORS is not None and model in MODEL_BASE_COLORS:
            color, _, _ = color_variants(MODEL_BASE_COLORS[model])
        label = model
        if model in set(auc_df["model_type"]):
            auc_val = auc_df.loc[auc_df["model_type"] == model, "auc_normalized"].iloc[0]
            label = f"{model} (AUC={auc_val:.3f})"
        ax.plot(d["threshold_nmae"].values, d["hit_rate"].values, linewidth=2.0, color=color, label=label)
    ax.set_xlabel("NMAE tolerance threshold")
    ax.set_ylabel("Fraction of samples within threshold")
    ax.set_title(f"Tolerance Curves (Normalized Error Coverage) | {int(util*100)}% Utilization | {coord_system}" + (f" | {title_suffix}" if title_suffix else ""))
    ax.set_xlim(float(threshold_grid.min()), float(threshold_grid.max()))
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_U{int(util*100)}_{coord_system}.png", dpi=220)
    if plot_show:
        plt.show()
    plt.close(fig)


def build_main_summary_table(runs_csv_path: Path, pred_csv_path: Path, out_dir: Path, experiment_id: str | None = None, util: float = 1.0, thresholds: list[float] | None = None, auc_threshold_grid: np.ndarray | None = None, prefix: str = "table_main_summary", print_table: bool = True,):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if thresholds is None:
        thresholds = [0.01, 0.02, 0.05, 0.10]
    if auc_threshold_grid is None:
        auc_threshold_grid = np.linspace(0.0, 0.15, 61)
    # ------------------ Load runs ------------------
    runs_df = pd.read_csv(runs_csv_path)
    if experiment_id is not None and "experiment_id" in runs_df.columns:
        runs_df = runs_df[runs_df["experiment_id"].astype(str) == str(experiment_id)]
    runs_df = runs_df[np.isclose(runs_df["utilization"].astype(float), float(util))]
    per_freq = (
        runs_df.groupby(["model_type", "frequency"], as_index=False).agg(mae_mean=("mae", "mean"), rmse_mean=("rmse", "mean"), nmae_mean=("nmae", "mean"),))
    metrics_summary = (
        per_freq.groupby("model_type", as_index=False).agg(mean_nmae=("nmae_mean", "mean"), mean_mae=("mae_mean", "mean"), mean_rmse=("rmse_mean", "mean"),))
    # ------------------ Predictions ------------------
    pred_df = pd.read_csv(pred_csv_path)
    if experiment_id is not None and "experiment_id" in pred_df.columns:
        pred_df = pred_df[pred_df["experiment_id"].astype(str) == str(experiment_id)]
    pred_df = pred_df[np.isclose(pred_df["utilization"].astype(float), float(util))]
    freq_range = pred_df.groupby("frequency").agg(y_min=("y_true", "min"), y_max=("y_true", "max"),).reset_index()
    freq_range["y_range"] = freq_range["y_max"] - freq_range["y_min"]
    pred_df = pred_df.merge(freq_range[["frequency", "y_range"]], on="frequency")
    pred_df["norm_abs_err"] = np.abs(pred_df["y_true"] - pred_df["y_pred"]) / pred_df["y_range"]
    pred_df = pred_df.dropna(subset=["norm_abs_err"])
    # ------------------ Hit rates ------------------
    hit_tables = []
    for thr in thresholds:
        col = f"hit@{thr:.2f}"
        tmp = (pred_df.assign(_hit=(pred_df["norm_abs_err"] <= thr).astype(float)).groupby("model_type", as_index=False).agg(**{col: ("_hit", "mean")}))
        hit_tables.append(tmp)
    tol_summary = hit_tables[0]
    for t in hit_tables[1:]:
        tol_summary = tol_summary.merge(t, on="model_type")
    # ------------------ AUC ------------------
    auc_rows = []
    for model, g in pred_df.groupby("model_type"):
        errs = np.sort(g["norm_abs_err"].values)
        hit_rates = [(errs <= thr).mean() for thr in auc_threshold_grid]
        auc = np.trapz(hit_rates, auc_threshold_grid) / (auc_threshold_grid.max() - auc_threshold_grid.min())
        auc_rows.append({"model_type": model, "AUC": auc})
    auc_df = pd.DataFrame(auc_rows)
    # ------------------ Final table ------------------
    final_df = (metrics_summary.merge(tol_summary, on="model_type").merge(auc_df, on="model_type").sort_values("mean_nmae"))
    final_df.to_csv(out_dir / f"{prefix}.csv", index=False)
    # ------------------ PRINT ------------------
    if print_table:
        table = PrettyTable()
        table.field_names = list(final_df.columns)
        for _, row in final_df.iterrows():
            table.add_row([f"{v:.4f}" if isinstance(v, float) else v for v in row])
        table.title = "MAIN SUMMARY TABLE"
        print("\n", table, "\n")
    return final_df

def build_appendix_frequency_table(runs_csv_path: Path, pred_csv_path: Path, out_dir: Path, experiment_id: str | None = None, util: float = 1.0, thresholds: list[float] | None = None, prefix: str = "table_appendix_frequency", print_rows: int = 30,):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if thresholds is None:
        thresholds = [0.01, 0.02, 0.05, 0.10]
    # ------------------ runs ------------------
    runs_df = pd.read_csv(runs_csv_path)
    if experiment_id is not None and "experiment_id" in runs_df.columns:
        runs_df = runs_df[runs_df["experiment_id"].astype(str) == str(experiment_id)]
    runs_df = runs_df[np.isclose(runs_df["utilization"].astype(float), float(util))]
    metrics = (
        runs_df.groupby(["model_type", "frequency"], as_index=False).agg(mae=("mae", "mean"), rmse=("rmse", "mean"), nmae=("nmae", "mean"),))
    # ------------------ predictions ------------------
    pred_df = pd.read_csv(pred_csv_path)
    if experiment_id is not None and "experiment_id" in pred_df.columns:
        pred_df = pred_df[pred_df["experiment_id"].astype(str) == str(experiment_id)]
    pred_df = pred_df[np.isclose(pred_df["utilization"].astype(float), float(util))]
    freq_range = pred_df.groupby("frequency").agg(y_min=("y_true", "min"), y_max=("y_true", "max"),).reset_index()
    freq_range["y_range"] = freq_range["y_max"] - freq_range["y_min"]
    pred_df = pred_df.merge(freq_range[["frequency", "y_range"]], on="frequency")
    pred_df["norm_abs_err"] = np.abs(pred_df["y_true"] - pred_df["y_pred"]) / pred_df["y_range"]
    # ------------------ hit rates ------------------
    hit_tables = []
    for thr in thresholds:
        col = f"hit@{thr:.2f}"
        tmp = (pred_df.assign(_hit=(pred_df["norm_abs_err"] <= thr).astype(float)).groupby(["model_type", "frequency"], as_index=False).agg(**{col: ("_hit", "mean")}))
        hit_tables.append(tmp)
    hit_df = hit_tables[0]
    for t in hit_tables[1:]:
        hit_df = hit_df.merge(t, on=["model_type", "frequency"])
    final_df = (metrics.merge(hit_df, on=["model_type", "frequency"]).sort_values(["model_type", "frequency"]))
    final_df.to_csv(out_dir / f"{prefix}.csv", index=False)
    # ------------------ PRINT (preview) ------------------
    table = PrettyTable()
    table.field_names = list(final_df.columns)
    preview = final_df.head(print_rows)
    for _, row in preview.iterrows():
        table.add_row([f"{v:.4f}" if isinstance(v, float) else v for v in row])
    table.title = f"APPENDIX TABLE (first {print_rows} rows)"
    print("\n", table, "\n")
    return final_df


def _mean_metrics_for_mode(csv_path: Path, mode: str, experiment_id: str | None = None, util: float = 1.0):
    df = load_runs(csv_path)
    if experiment_id is not None and "experiment_id" in df.columns:
        df = df[df["experiment_id"].astype(str) == str(experiment_id)]
    df = df[np.isclose(df["utilization"].astype(float), float(util))]
    if df.empty:
        raise ValueError(f"No data for mode={mode}, utilization={util}: {csv_path}")
    metrics = [c for c in ["mae", "mape", "rmse", "nmae"] if c in df.columns]
    by_freq = (
        df.groupby(["model_type", "frequency"], as_index=False)
        .agg({metric: "mean" for metric in metrics})
    )
    by_freq["mode"] = mode
    return by_freq, metrics


def build_heldout_comparison_tables(
    normal_csv_path: Path,
    heldout_csv_path: Path,
    out_dir: Path,
    experiment_id: str | None = None,
    util: float = 1.0,
    prefix: str = "heldout_vs_normal",
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    normal, normal_metrics = _mean_metrics_for_mode(normal_csv_path, "normal", experiment_id=experiment_id, util=util)
    heldout, heldout_metrics = _mean_metrics_for_mode(heldout_csv_path, "heldout", experiment_id=experiment_id, util=util)
    metrics = [m for m in normal_metrics if m in heldout_metrics]
    merged = normal.merge(heldout, on=["model_type", "frequency"], suffixes=("_normal", "_heldout"))
    for metric in metrics:
        merged[f"{metric}_gap"] = merged[f"{metric}_heldout"] - merged[f"{metric}_normal"]
        merged[f"{metric}_ratio"] = np.where(
            merged[f"{metric}_normal"] != 0,
            merged[f"{metric}_heldout"] / merged[f"{metric}_normal"],
            np.nan,
        )
    merged["frequency_ghz"] = merged["frequency"] / 1e9
    merged.to_csv(out_dir / f"{prefix}_by_frequency_U{int(util*100)}.csv", index=False)

    agg_spec = {}
    for metric in metrics:
        agg_spec[f"{metric}_normal"] = (f"{metric}_normal", "mean")
        agg_spec[f"{metric}_heldout"] = (f"{metric}_heldout", "mean")
        agg_spec[f"{metric}_gap"] = (f"{metric}_gap", "mean")
        agg_spec[f"{metric}_ratio"] = (f"{metric}_ratio", "mean")
    summary = (
        merged.groupby("model_type", as_index=False)
        .agg(**agg_spec)
        .sort_values("nmae_heldout" if "nmae" in metrics else f"{metrics[0]}_heldout")
    )
    summary.to_csv(out_dir / f"{prefix}_summary_U{int(util*100)}.csv", index=False)

    if "nmae" in metrics:
        base_col = "nmae_heldout"
    else:
        base_col = f"{metrics[0]}_heldout"
    best_model = summary.iloc[0]["model_type"]
    improvement_rows = []
    for metric in metrics:
        best_target = float(summary.loc[summary["model_type"] == best_model, f"{metric}_heldout"].iloc[0])
        baselines = summary[summary["model_type"] != best_model]
        best_baseline = baselines.sort_values(f"{metric}_heldout").iloc[0] if not baselines.empty else None
        if best_baseline is not None:
            baseline_value = float(best_baseline[f"{metric}_heldout"])
            improvement_rows.append({
                "metric": metric,
                "best_model": best_model,
                "best_model_heldout": best_target,
                "best_baseline": best_baseline["model_type"],
                "best_baseline_heldout": baseline_value,
                "relative_improvement_percent": 100.0 * (baseline_value - best_target) / baseline_value if baseline_value else np.nan,
            })
    improvement = pd.DataFrame(improvement_rows)
    improvement.to_csv(out_dir / f"{prefix}_best_model_improvement_U{int(util*100)}.csv", index=False)
    return merged, summary, improvement


def plot_heldout_comparison(
    normal_csv_path: Path,
    heldout_csv_path: Path,
    out_dir: Path,
    experiment_id: str | None = None,
    util: float = 1.0,
    MODEL_BASE_COLORS=None,
    plot_show: bool = False,
    prefix: str = "heldout_vs_normal",
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    merged, summary, _ = build_heldout_comparison_tables(
        normal_csv_path=normal_csv_path,
        heldout_csv_path=heldout_csv_path,
        out_dir=out_dir,
        experiment_id=experiment_id,
        util=util,
        prefix=prefix,
    )
    metrics = [m[:-8] for m in summary.columns if m.endswith("_heldout") and f"{m[:-8]}_normal" in summary.columns]
    model_order = list(summary["model_type"])

    nrows, ncols = auto_subplot_grid(len(metrics), max_cols=2)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 4.2 * nrows))
    axes = np.asarray(axes).reshape(-1) if len(metrics) > 1 else [axes]
    x = np.arange(len(model_order))
    width = 0.36
    for ax, metric in zip(axes, metrics):
        normal_vals = [float(summary.loc[summary["model_type"] == model, f"{metric}_normal"].iloc[0]) for model in model_order]
        heldout_vals = [float(summary.loc[summary["model_type"] == model, f"{metric}_heldout"].iloc[0]) for model in model_order]
        ax.bar(x - width / 2, normal_vals, width, label="Normal test", color="#4C78A8")
        ax.bar(x + width / 2, heldout_vals, width, label="Heldout zero-shot", color="#F58518")
        ax.set_title(metric.upper())
        ax.set_xticks(x)
        ax.set_xticklabels(model_order, rotation=25, ha="right")
        ax.grid(True, axis="y", alpha=0.25)
    for ax in axes[len(metrics):]:
        ax.remove()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle(f"Normal vs Heldout Metrics | {int(util*100)}% Utilization", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_dir / f"{prefix}_metric_bars_U{int(util*100)}.png", dpi=220)
    if plot_show:
        plt.show()
    plt.close(fig)

    if "nmae_gap" in merged.columns:
        fig, ax = plt.subplots(figsize=(9.5, 5.6))
        for model in sorted(merged["model_type"].unique()):
            d = merged[merged["model_type"] == model].sort_values("frequency")
            color = None
            if MODEL_BASE_COLORS is not None and model in MODEL_BASE_COLORS:
                color, _, _ = color_variants(MODEL_BASE_COLORS[model])
            ax.plot(d["frequency_ghz"], d["nmae_gap"], marker="o", linewidth=2, color=color, label=model)
        ax.axhline(0, color="black", linewidth=1, alpha=0.5)
        ax.set_xlabel("Frequency (GHz)")
        ax.set_ylabel("Heldout NMAE - Normal NMAE")
        ax.set_title(f"Zero-Shot Generalization Gap | {int(util*100)}% Utilization")
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out_dir / f"{prefix}_nmae_gap_by_frequency_U{int(util*100)}.png", dpi=220)
        if plot_show:
            plt.show()
        plt.close(fig)

    if "nmae_normal" in merged.columns and "nmae_heldout" in merged.columns:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)
        for ax, mode in zip(axes, ["normal", "heldout"]):
            for model in sorted(merged["model_type"].unique()):
                d = merged[merged["model_type"] == model].sort_values("frequency")
                color = None
                if MODEL_BASE_COLORS is not None and model in MODEL_BASE_COLORS:
                    color, _, _ = color_variants(MODEL_BASE_COLORS[model])
                ax.plot(d["frequency_ghz"], d[f"nmae_{mode}"], marker="o", linewidth=2, color=color, label=model)
            ax.set_title("Normal Test" if mode == "normal" else "Heldout Zero-Shot")
            ax.set_xlabel("Frequency (GHz)")
            ax.grid(True, alpha=0.3)
        axes[0].set_ylabel("NMAE")
        axes[1].legend(frameon=False)
        fig.suptitle(f"NMAE by Frequency | {int(util*100)}% Utilization", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        fig.savefig(out_dir / f"{prefix}_nmae_by_frequency_U{int(util*100)}.png", dpi=220)
        if plot_show:
            plt.show()
        plt.close(fig)
    return summary


def plot_heldout_from_logs(
    normal_csv_path: Path,
    normal_pred_csv_path: Path,
    heldout_csv_path: Path,
    heldout_pred_csv_path: Path,
    out_dir: Path,
    experiment_id: str | None,
    title: str = "",
    MODEL_BASE_COLORS=None,
    plot_show: bool = False,
    util: float = 1.0,
):
    out_dir = Path(out_dir)
    heldout_dir = out_dir / "heldout"
    comparison_dir = out_dir / "heldout_comparison"
    plot_utilization_from_logs(
        csv_path=Path(heldout_csv_path),
        pred_csv_path=Path(heldout_pred_csv_path) if heldout_pred_csv_path and Path(heldout_pred_csv_path).exists() else None,
        out_dir=heldout_dir,
        experiment_id=experiment_id,
        title=(title + " | Heldout zero-shot").strip(),
        MODEL_BASE_COLORS=MODEL_BASE_COLORS,
        plot_show=plot_show,
    )
    summary = plot_heldout_comparison(
        normal_csv_path=Path(normal_csv_path),
        heldout_csv_path=Path(heldout_csv_path),
        out_dir=comparison_dir,
        experiment_id=experiment_id,
        util=util,
        MODEL_BASE_COLORS=MODEL_BASE_COLORS,
        plot_show=plot_show,
    )
    return summary


def plot_utilization_from_logs(csv_path: Path,  pred_csv_path: Path, out_dir: Path, experiment_id: str | None, title: str = "", MODEL_BASE_COLORS=None, plot_show: bool = False):
    df = load_runs(csv_path)
    if experiment_id is not None:
        df = df[df["experiment_id"].astype(str) == str(experiment_id)]
    if df.empty:
        raise ValueError("No data left after filtering.")
    coord_system = (df["coordinate"].iloc[0] if "coordinate" in df.columns else "Unknown")
    agg = aggregate_metrics(df, group_cols=["model_type", "frequency", "utilization"])
    # # ---- per-frequency plots ----
    # plot_utilization_curves_per_freq(agg_long=agg, out_dir=Path(out_dir), title_suffix=title, MODEL_BASE_COLORS=MODEL_BASE_COLORS)
    plot_all_metrics_per_freq( agg_long=agg, out_dir=Path(out_dir), title_suffix=title, MODEL_BASE_COLORS=MODEL_BASE_COLORS, coord_system=coord_system,)
    # ---- overall over-frequencies plots ----
    plot_utilization_curves_total(agg_long=agg, out_dir=Path(out_dir), title_suffix=title, MODEL_BASE_COLORS=MODEL_BASE_COLORS, coord_system=coord_system, plot_show=plot_show)
    # Error vs frequency @ 100%
    plot_all_metrics_per_util(agg_long=agg, out_dir=Path(out_dir), util=1.0, title_suffix=title, MODEL_BASE_COLORS=MODEL_BASE_COLORS, coord_system=coord_system, prefix="errbyfreq",)
    agg.to_csv(Path(out_dir) / "aggregated_metrics_long.csv", index=False)
    if pred_csv_path is not None and Path(pred_csv_path).exists():
        plot_y_vs_frequency_from_predictions(pred_csv_path=Path(pred_csv_path), out_dir=Path(out_dir), experiment_id=experiment_id, util=1.0, title_suffix=title, MODEL_BASE_COLORS=MODEL_BASE_COLORS, plot_show=plot_show, prefix="ybyfreq",)
        plot_tolerance_heatmaps_from_predictions(pred_csv_path=Path(pred_csv_path), out_dir=Path(out_dir), experiment_id=experiment_id, util=1.0, thresholds=[0.01, 0.02, 0.05, 0.10], title_suffix=title, plot_show=plot_show, prefix="tolheat",)
        plot_tolerance_curves_from_predictions(pred_csv_path=Path(pred_csv_path), out_dir=Path(out_dir), experiment_id=experiment_id, util=1.0, threshold_grid=np.linspace(0.0, 0.15, 61), title_suffix=title, MODEL_BASE_COLORS=MODEL_BASE_COLORS, plot_show=plot_show, prefix="tolcurve",)
        main_table = build_main_summary_table(runs_csv_path=Path(csv_path), pred_csv_path=Path(pred_csv_path), out_dir=Path(out_dir), experiment_id=experiment_id, util=1.0, thresholds=[0.01, 0.02, 0.05, 0.10], auc_threshold_grid=np.linspace(0.0, 0.15, 61), prefix="table_main_summary",)
        appendix_table = build_appendix_frequency_table(runs_csv_path=Path(csv_path), pred_csv_path=Path(pred_csv_path), out_dir=Path(out_dir), experiment_id=experiment_id, util=1.0, thresholds=[0.01, 0.02, 0.05, 0.10], prefix="table_appendix_frequency",)
