# analysis/reports.py

import json
import math
import os
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _json_safe(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (np.ndarray, list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, Path):
        return str(value)
    return value


def _read_json(path):
    path = Path(path)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        return {"read_error": str(exc)}


def _describe_files(files):
    rows = []
    for file_path in files:
        path = Path(file_path)
        rows.append({
            "name": path.name,
            "path": str(path),
            "size_mb": round(path.stat().st_size / (1024 * 1024), 3) if path.exists() else None,
            "exists": path.exists(),
        })
    return rows


def _numeric_summary(series):
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
        }
    return {
        "count": int(numeric.shape[0]),
        "min": _json_safe(numeric.min()),
        "max": _json_safe(numeric.max()),
        "mean": _json_safe(numeric.mean()),
        "std": _json_safe(numeric.std(ddof=0)),
    }


def _value_counts(series, limit=20):
    counts = Counter(series.dropna().tolist())
    ordered = sorted(counts.items(), key=lambda item: (-item[1], str(item[0])))
    return [{"value": _json_safe(value), "count": int(count)} for value, count in ordered[:limit]]


def summarize_data_by_frequency(data_by_freq, config, split_name):
    """Summarize the already selected frequency slices used by this run."""
    if data_by_freq is None:
        return None

    target_cols = list(getattr(config, "targets", []) or [])
    feature_cols = list(getattr(config, "features", []) or [])
    topology_col = getattr(config, "TOPOLOGY_COL", "topology_index")

    summaries = []
    for freq in getattr(config, "freq_list", []):
        df = data_by_freq.get(freq)
        if df is None:
            summaries.append({
                "split": split_name,
                "frequency_hz": _json_safe(freq),
                "frequency_ghz": _json_safe(freq / 1e9),
                "rows": 0,
                "status": "missing_frequency",
            })
            continue

        columns = list(df.columns)
        missing_targets = [col for col in target_cols if col not in columns]
        missing_features = [col for col in feature_cols if col not in columns]

        summary = {
            "split": split_name,
            "frequency_hz": _json_safe(freq),
            "frequency_ghz": _json_safe(freq / 1e9),
            "rows": int(len(df)),
            "columns": int(len(columns)),
            "status": "ok" if not df.empty else "empty",
            "target_columns": target_cols,
            "feature_columns": feature_cols,
            "missing_targets": missing_targets,
            "missing_features": missing_features,
            "target_summary": {
                col: _numeric_summary(df[col]) for col in target_cols if col in df.columns
            },
        }
        if topology_col in df.columns:
            summary["topology_counts"] = _value_counts(df[topology_col])
        summaries.append(summary)
    return summaries


def write_experiment_dataset_report(
    *,
    out_dir,
    experiment_id,
    config,
    paths,
    heldout_paths,
    loaded_summaries,
    preprocessed_summaries,
    dataset_read_time_sec,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    main_metadata = _read_json(Path(paths.dataset_root) / "metadata.json")
    heldout_metadata = _read_json(Path(heldout_paths.dataset_root) / "metadata.json") if heldout_paths else None
    main_cache = {
        "train": _read_json(Path(paths.dataset_root) / ".cache" / "train_all.meta.json"),
        "test": _read_json(Path(paths.dataset_root) / ".cache" / "test_all.meta.json"),
    }
    heldout_cache = None
    if heldout_paths:
        heldout_cache = {
            "test": _read_json(Path(heldout_paths.dataset_root) / ".cache" / "test_all.meta.json"),
        }

    report = {
        "schema_version": "dataset_report_v1",
        "experiment_id": experiment_id,
        "experiment_dir": str(out_dir),
        "dataset_context": {
            "main_arch": getattr(config, "main_arch", None),
            "main_topology": getattr(config, "main_topology", None),
            "selected_topologies": list(getattr(config, "selected_topologies", []) or []),
            "target": (getattr(config, "targets", [None]) or [None])[0],
            "targets": list(getattr(config, "targets", []) or []),
            "cartesian": bool(getattr(config, "cartesian", False)),
            "preprocess": bool(getattr(config, "preprocess", False)),
            "dataset_version": getattr(paths, "dataset_version", None),
            "dataset_variant": getattr(paths, "dataset_variant", None),
            "dataset_split": getattr(paths, "dataset_split", None),
            "heldout_eval_enabled": bool(getattr(config, "run_heldout_eval", False)),
            "heldout_dataset_split": getattr(config, "heldout_dataset_split", None) if heldout_paths else None,
            "frequency_mode": getattr(config, "freq_mode", None),
            "frequency_count": getattr(config, "freq_count", None),
            "selected_frequencies_hz": _json_safe(getattr(config, "freq_list", [])),
            "selected_frequencies_ghz": _json_safe([freq / 1e9 for freq in getattr(config, "freq_list", [])]),
            "utilization": _json_safe(getattr(config, "utilization", [])),
            "models": list(getattr(config, "model_types", []) or []),
            "dataset_cache_enabled": bool(getattr(config, "use_dataset_cache", False)),
            "rebuild_dataset_cache": bool(getattr(config, "rebuild_dataset_cache", False)),
            "dataset_read_time_sec": _json_safe(dataset_read_time_sec),
        },
        "paths": {
            "main_dataset_root": str(paths.dataset_root),
            "heldout_dataset_root": str(heldout_paths.dataset_root) if heldout_paths else None,
            "weights_path": str(paths.weights_path),
            "results_dir": str(paths.run_result_dir),
        },
        "source_files": {
            "main_train": _describe_files(paths.train_data_paths),
            "main_test": _describe_files(paths.test_data_paths),
            "heldout_test": _describe_files(heldout_paths.test_data_paths) if heldout_paths else [],
        },
        "dataset_metadata": {
            "main": main_metadata,
            "heldout": heldout_metadata,
        },
        "cache_metadata": {
            "main": main_cache,
            "heldout": heldout_cache,
        },
        "loaded_frequency_summaries": loaded_summaries,
        "preprocessed_frequency_summaries": preprocessed_summaries,
    }

    report_path = out_dir / "dataset_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(report), f, indent=2)

    flat_rows = []
    for stage_name, stage_summary in [
        ("loaded", loaded_summaries),
        ("preprocessed", preprocessed_summaries),
    ]:
        for split_name, summaries in (stage_summary or {}).items():
            if summaries is None:
                continue
            for item in summaries:
                row = {
                    "stage": stage_name,
                    "split": split_name,
                    "frequency_hz": item.get("frequency_hz"),
                    "frequency_ghz": item.get("frequency_ghz"),
                    "rows": item.get("rows"),
                    "columns": item.get("columns"),
                    "status": item.get("status"),
                    "missing_targets": ";".join(item.get("missing_targets", []) or []),
                    "missing_features": ";".join(item.get("missing_features", []) or []),
                }
                for target, stats in (item.get("target_summary") or {}).items():
                    for key, value in stats.items():
                        row[f"{target}_{key}"] = value
                flat_rows.append(row)
    pd.DataFrame(flat_rows).to_csv(out_dir / "dataset_frequency_summary.csv", index=False)
    return str(report_path)


def write_runtime_reports(run_dir, out_dir, model_colors=None, plot_show=False):
    run_dir = Path(run_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_results_path = run_dir / "run_results.csv"
    if not run_results_path.exists():
        return None

    df = pd.read_csv(run_results_path)
    if "loop_time_sec" not in df.columns:
        return None
    df["loop_time_sec"] = pd.to_numeric(df["loop_time_sec"], errors="coerce")
    df = df.dropna(subset=["loop_time_sec"])
    if df.empty:
        return None

    by_model = (
        df.groupby(["model_type", "utilization"])["loop_time_sec"]
        .agg(["mean", "std", "min", "max", "sum", "count"])
        .reset_index()
    )
    by_freq = (
        df.groupby(["frequency", "utilization"])["loop_time_sec"]
        .agg(["mean", "std", "min", "max", "sum", "count"])
        .reset_index()
    )
    overall = pd.DataFrame([{
        "loops": int(len(df)),
        "total_loop_time_sec": float(df["loop_time_sec"].sum()),
        "mean_loop_time_sec": float(df["loop_time_sec"].mean()),
        "max_loop_time_sec": float(df["loop_time_sec"].max()),
    }])

    by_model.to_csv(out_dir / "runtime_summary_by_model.csv", index=False)
    by_freq.to_csv(out_dir / "runtime_summary_by_frequency.csv", index=False)
    overall.to_csv(out_dir / "runtime_summary_overall.csv", index=False)

    util_100 = by_model[np.isclose(pd.to_numeric(by_model["utilization"], errors="coerce"), 1.0)]
    if not util_100.empty:
        util_100 = util_100.sort_values("mean")
        colors = [model_colors.get(m, "#4C78A8") if model_colors else "#4C78A8" for m in util_100["model_type"]]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(util_100["model_type"], util_100["mean"], color=colors)
        ax.set_ylabel("Mean loop time (sec)")
        ax.set_xlabel("Model")
        ax.set_title("Runtime by model at 100% utilization")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / "runtime_by_model_U100.png", dpi=180)
        if plot_show:
            plt.show()
        plt.close(fig)

    return str(out_dir / "runtime_summary_by_model.csv")


def write_model_input_report(run_dir, config, out_dir):
    run_dir = Path(run_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_results_path = run_dir / "run_results.csv"
    if not run_results_path.exists():
        return None

    df = pd.read_csv(run_results_path)
    rows = []
    adapter_modes = getattr(config, "ADAPTER_MODES", {})
    frequency_count = int(pd.to_numeric(df.get("frequency"), errors="coerce").dropna().nunique()) if "frequency" in df.columns else len(getattr(config, "freq_list", []) or [])
    for model in sorted(df["model_type"].dropna().astype(str).unique()):
        subset = df[df["model_type"].astype(str) == model]
        sample = subset.iloc[0].to_dict()
        adapter_mode = adapter_modes.get(model, "")
        rows.append({
            "model_type": model,
            "adapter_mode": adapter_mode,
            "input_shape": sample.get("input_shape"),
            "output_shape": sample.get("output_shape"),
            "layers": sample.get("layers"),
            "uses_flat_s_parameters": adapter_mode == "flat",
            "uses_semantic_cci_groups": adapter_mode == "semantic",
            "uses_graph_adapter": adapter_mode == "gnn",
            "hybrid_cci_gnn": model.startswith(("CCIGNN", "CCIEdgeGNN")),
            "uses_topology_token": "-Top" in model or model.endswith("TopNet") or model.endswith("TopNetCon"),
            "frequency_count": frequency_count,
            "target": (getattr(config, "targets", [None]) or [None])[0],
            "cartesian": bool(getattr(config, "cartesian", False)),
        })
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "model_input_summary.csv", index=False)
    with (out_dir / "model_input_summary.json").open("w", encoding="utf-8") as f:
        json.dump(_json_safe(rows), f, indent=2)
    return str(out_dir / "model_input_summary.csv")


def write_inference_latency_reports(run_dir, out_dir, model_colors=None, plot_show=False):
    run_dir = Path(run_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    latency_path = run_dir / "inference_latency.csv"
    if not latency_path.exists():
        return None

    df = pd.read_csv(latency_path)
    numeric_cols = [
        "samples",
        "mean_latency_ms",
        "median_latency_ms",
        "min_latency_ms",
        "max_latency_ms",
        "std_latency_ms",
        "mean_per_sample_ms",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["mean_latency_ms"])
    if df.empty:
        return None

    group_cols = ["model_type", "source", "mode", "utilization"]
    summary = (
        df.groupby(group_cols)
        .agg(
            samples=("samples", "max"),
            mean_latency_ms=("mean_latency_ms", "mean"),
            median_latency_ms=("median_latency_ms", "mean"),
            min_latency_ms=("min_latency_ms", "min"),
            max_latency_ms=("max_latency_ms", "max"),
            mean_per_sample_ms=("mean_per_sample_ms", "mean"),
            runs=("mean_latency_ms", "count"),
        )
        .reset_index()
    )
    summary.to_csv(out_dir / "inference_latency_summary.csv", index=False)

    util_100_single = summary[
        np.isclose(pd.to_numeric(summary["utilization"], errors="coerce"), 1.0)
        & (summary["source"].astype(str) == "test")
        & (summary["mode"].astype(str) == "single")
    ]
    if not util_100_single.empty:
        util_100_single = util_100_single.sort_values("mean_latency_ms")
        colors = [model_colors.get(m, "#4C78A8") if model_colors else "#4C78A8" for m in util_100_single["model_type"]]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(util_100_single["model_type"], util_100_single["mean_latency_ms"], color=colors)
        ax.set_ylabel("Mean single-sample inference latency (ms)")
        ax.set_xlabel("Model")
        ax.set_title("Inference latency by model at 100% utilization")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / "inference_latency_single_U100.png", dpi=180)
        if plot_show:
            plt.show()
        plt.close(fig)

    return str(out_dir / "inference_latency_summary.csv")


def write_rebuttal_reports(run_dir, config, model_colors=None, plot_show=False):
    analysis_dir = Path(run_dir) / "analysis"
    extra_dir = analysis_dir / "rebuttal"
    extra_dir.mkdir(parents=True, exist_ok=True)
    runtime_path = write_runtime_reports(run_dir, extra_dir, model_colors=model_colors, plot_show=plot_show)
    input_path = write_model_input_report(run_dir, config, extra_dir)
    latency_path = write_inference_latency_reports(run_dir, extra_dir, model_colors=model_colors, plot_show=plot_show)
    return {
        "runtime_summary": runtime_path,
        "model_input_summary": input_path,
        "inference_latency_summary": latency_path,
        "output_dir": str(extra_dir),
    }
