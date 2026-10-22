# utils/file_io.py

import json
import math
import os
import glob
import pandas as pd
import tensorflow as tf
import subprocess
import re
import sys
import csv
import numpy as np
import io
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# -------------------------------------------------------------
# Path Dataclass
# -------------------------------------------------------------
@dataclass
class Paths:
    root: str
    dataset_version: str
    dataset_variant: str | None
    dataset_split: str
    dataset_root: str
    train_data_paths: list
    test_data_paths: list
    weights_path: str
    results_path: str
    images_path: str
    run_result_dir: str
    result_log_file: str


# -------------------------------------------------------------
# Create and initialize all paths (Auto-creates directories)
# -------------------------------------------------------------
def resolve_dataset_root(project_root, main_arch, main_topology, dataset_version="legacy", dataset_split="main", dataset_variant=None):
    """Resolve current and versioned dataset layouts."""
    if dataset_version in (None, "", "legacy"):
        return os.path.join(project_root, "dataset", main_arch, dataset_split, main_topology if main_topology else "")
    if dataset_variant:
        return os.path.join(project_root, "dataset", dataset_version, dataset_variant, main_arch, dataset_split, main_topology if main_topology else "")
    return os.path.join(project_root, "dataset", dataset_version, main_arch, dataset_split, main_topology if main_topology else "")


def make_dataset_label(dataset_version="legacy", dataset_variant=None):
    if dataset_version in (None, "", "legacy"):
        label = "legacy"
    else:
        label = safe_slug(dataset_version).replace("_", "")
    if dataset_variant:
        label = f"{label}_{safe_slug(dataset_variant)}"
    return label


def make_run_dir_name(experiment_id, spec_name=None, dataset_version="legacy", dataset_split="main", run_heldout_eval=False, dataset_variant=None):
    """Build a compact run directory name with dataset context."""
    spec_part = safe_slug(spec_name if spec_name else "")
    dataset_part = make_dataset_label(dataset_version, dataset_variant)
    if run_heldout_eval:
        dataset_part = f"{dataset_part}_heldout"
    elif dataset_split and dataset_split != "main":
        dataset_part = f"{dataset_part}_{safe_slug(dataset_split)}"
    parts = [str(experiment_id), dataset_part, spec_part]
    return "_".join(part for part in parts if part)


def make_dataset_context_slug(dataset_version="legacy", dataset_split="main", dataset_variant=None):
    version_part = make_dataset_label(dataset_version, dataset_variant)
    split_part = safe_slug(dataset_split or "main")
    return f"{version_part}_{split_part}"


def make_main_weights_context_slug(dataset_version="legacy", dataset_variant=None):
    """Return the complete dataset identity used below weights/main/."""
    return make_dataset_label(dataset_version, dataset_variant)


def create_paths(main_arch, main_topology, experiment_id=None, spec_name=None, dataset_version="legacy", dataset_split="main", run_heldout_eval=False, dataset_variant=None) -> Paths:
    # root: Project root directory (= Parent to the code folder)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    # Dataset directory (same as original script)
    dataset_root = resolve_dataset_root(project_root, main_arch, main_topology, dataset_version, dataset_split, dataset_variant)
    # CSV paths
    train_data_paths = sorted(glob.glob(os.path.join(dataset_root, "train_*.csv")))
    test_data_paths = sorted(glob.glob(os.path.join(dataset_root, "test_*.csv")))
    # weights: Where to save and restore the trained model
    weights_path = os.path.join(
        project_root,
        "weights",
        "main",
        make_main_weights_context_slug(dataset_version, dataset_variant),
        main_arch,
        main_topology,
    )
    # results: Where to save main modeling results
    results_path = os.path.join(project_root, "results", "main", main_arch)
    # Run directory
    run_result_dir = os.path.join(
        results_path,
        make_run_dir_name(
            experiment_id,
            spec_name=spec_name,
            dataset_version=dataset_version,
            dataset_split=dataset_split,
            run_heldout_eval=run_heldout_eval,
            dataset_variant=dataset_variant,
        ),
    )
    result_log_file = os.path.join(run_result_dir, "_MainModeling-RUN.txt")
    # Keep diagnostics with their experiment so repeated runs cannot overwrite
    # images generated from a different dataset, split, target, or frequency.
    images_path = (
        os.path.join(run_result_dir, "diagnostics")
        if experiment_id is not None
        else os.path.join(project_root, "images", make_dataset_context_slug(dataset_version, dataset_split, dataset_variant), main_arch)
    )
    if experiment_id is not None:
        os.makedirs(run_result_dir, exist_ok=True)
    # Create directories
    for p in [weights_path, results_path, images_path]:
        os.makedirs(p, exist_ok=True)
    return Paths(
        root=project_root,
        dataset_version=dataset_version or "legacy",
        dataset_variant=dataset_variant,
        dataset_split=dataset_split or "main",
        dataset_root=dataset_root,
        train_data_paths=train_data_paths,
        test_data_paths=test_data_paths,
        weights_path=weights_path,
        results_path=results_path,
        images_path=images_path,
        run_result_dir=run_result_dir,
        result_log_file=result_log_file
    )


# -------------------------------------------------------------
# Append text to a file (used for logging model summary)
# -------------------------------------------------------------
def append_text_to_file(text, filepath: str):
    """
    Safely append text to a file.
    Converts None, dicts, lists, objects into strings.
    """
    if text is None:
        text = "<None>"
    if not isinstance(text, str):
        text = str(text)
    p = Path(filepath)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(text if text.endswith("\n") else text + "\n")

# -------------------------------------------------------------
# Write a dictionary to a text file (same behavior as original)
# -------------------------------------------------------------
def saved_dict_writer(path: str, python_dict: dict):
    """
    Save a dictionary to a TXT file in a readable key = value format.

    Direct relocation from your original helper.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for k, v in python_dict.items():
            f.write(f"{k} = {v}\n")


# -------------------------------------------------------------
# Write rows to CSV incrementally (very large-safe)
# -------------------------------------------------------------
def csv_writer(dframe, output_path, prefix="", chunk_size_mb=100):
    rows_per_chunk = estimate_rows_per_chunk(dframe, chunk_size_mb)
    print(f"Total rows: {len(dframe)}")
    print(f"Rows/chunk: ~{rows_per_chunk} (for ~{chunk_size_mb}MB chunks)")
    # ======= write chunks: TRAIN =======
    if len(dframe) == 0:
        outpath = os.path.join(output_path, f"{prefix}_1.csv")
        dframe.head(0).to_csv(outpath, index=False)
        print("Wrote empty chunk:", outpath)
    else:
        chunk_idx = 1
        for start in range(0, len(dframe), rows_per_chunk):
            stop = min(start + rows_per_chunk, len(dframe))
            outpath = os.path.join(output_path, f"{prefix}_{chunk_idx:02d}.csv")
            dframe.iloc[start:stop].to_csv(outpath, index=False)
            print(f"Wrote chunk #{chunk_idx}: rows [{start:7d}:{stop:7d}] -> {outpath}")
            chunk_idx += 1

# ======= estimate rows per ~100MB chunk (CSV-based sample) =======
def estimate_rows_per_chunk(dframe, chunk_mb=100, sample_rows=5000):
    if len(dframe) == 0:
        return 0
    n = min(len(dframe), sample_rows)
    buf = io.StringIO()
    dframe.iloc[:n].to_csv(buf, index=False)
    bytes_per_row = max(1, len(buf.getvalue().encode("utf-8")) / n)
    target_bytes = chunk_mb * 1024 * 1024
    return max(1, int(target_bytes / bytes_per_row))


# -------------------------------------------------------------
# Save predictions for a single frequency (unchanged behavior)
# -------------------------------------------------------------
def save_preds_per_frequency(results_dir, model_type, freq_label, y_pred, yscaler, spec_name, overwrite=True):
    """
    Save predictions for one frequency into its own CSV file:
    <results_dir> / pred_<model_type>_<spec_name>_freq_<freq_label>.csv

    - results_dir: directory where CSVs live
    - model_type: the type of the model (e.g., "GNN")
    - freq_label: the frequency identifier (e.g., 10 or "10GHz")
    - y_pred: array-like predictions in scaled space (N x 1 or N,)
    - yscaler: fitted scaler with inverse_transform
    - spec_name: e.g., "S11" -> column will be "S11_pred"
    - overwrite: if True, replace file; else append
    """
    os.makedirs(results_dir, exist_ok=True)
    col = f"{spec_name}_pred"
    vals = yscaler.inverse_transform(np.asarray(y_pred).reshape(-1, 1)).ravel()
    df = pd.DataFrame({
        "freq": freq_label,
        col: vals,
        "row_idx": np.arange(len(vals))  # optional: helpful for debugging/ordering
    })
    fname = f"pred_{safe_slug(model_type)}_{safe_slug(spec_name)}_freq_{safe_slug(freq_label)}.csv"
    path = os.path.join(results_dir, fname)
    if overwrite or not os.path.exists(path):
        df.to_csv(path, index=False)
    else:
        # Append without duplicating header
        df.to_csv(path, index=False, mode="a", header=False)
    return path

def make_freq_string(freq_list):
    """
    Returns a readable frequency summary string.
    Example: Freqs=[9.81GHz, 10.00GHz, 10.20GHz]
    """
    return "Freqs=[" + ",".join([f"{freq/1e9:.2f}GHz" for freq in freq_list]) + "]"


def print_cuda_info():
    # Log and define the information about the Python and TensorFlow versions, CUDA, and GPU availability.
    print("Versions and CUDA information:", "\nPython:", sys.version, "\nTensorFlow:", tf.__version__, "\nCUDA built:", tf.test.is_built_with_cuda())
    if tf.test.is_built_with_cuda():
        try:
            output = subprocess.check_output(['nvcc', '--version'], encoding='utf-8')
            match = re.search(r'release (\d+\.\d+)', output)
            if match:
                print("CUDA:", match.group(1))
            else:
                print("Could not parse CUDA version from nvcc output.")
        except FileNotFoundError:
            print("nvcc not found. CUDA toolkit may not be installed or not in PATH.")
        print("TensorFlow CUDA:", tf.sysconfig.get_build_info()["cuda_version"])
        print("TensorFlow cuDNN:", tf.sysconfig.get_build_info()["cudnn_version"])
        physical_devices = tf.config.list_physical_devices('GPU')
        print(f"\nPhysical GPUs found: {physical_devices}\n")
        if not physical_devices: # Exit if no GPU found
            sys.exit("\n\nStop: No GPU found")
        # # Enable memory growth for each GPU
        # for device in physical_devices:
        #     try:
        #         # Limiting GPU memory growth (https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth)
        #         tf.config.experimental.set_memory_growth(device, True)
        #         # tf.config.set_memory_growth(device, True) # New version of TensorFlow
        #     except RuntimeError as e:
        #         print(f"Error setting memory growth for {device}: {e}")


def safe_slug(x):
    """Make a filesystem-safe slug for filenames."""
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', str(x))


# -------------------------------------------------------------
# Save CSV for predictions or results (same behavior as original)
# -------------------------------------------------------------
def data_to_csv_saver(csv_path, new_data: dict):
    """
    Append a dict row to a CSV, evolving the schema if needed:
    - Create if missing
    - Add new columns if needed
    - Fill missing columns with ""
    """
    fields = []
    if os.path.exists(csv_path):
        with open(csv_path, newline='', encoding="utf-8") as f:
            fields = csv.DictReader(f).fieldnames or []
    # union of existing + new keys, keeping order
    all_fields = list(dict.fromkeys(fields + list(new_data.keys())))
    # prepare row with all fields present
    row = {f: new_data.get(f, "") for f in all_fields}
    if not fields or set(fields) != set(all_fields):
        # Schema changed or file doesn't exist → rewrite entire file with new header
        rows = []
        if fields:
            # Read existing rows and pad them with any new columns
            with open(csv_path, newline='', encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
                for r in rows:
                    for f in all_fields:
                        r.setdefault(f, "")
        rows.append(row)
        with open(csv_path, "w", newline='', encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_fields)
            writer.writeheader()
            writer.writerows(rows)
    else:
        # Same schema → just append
        with open(csv_path, "a", newline='', encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fields).writerow(row)


def list_and_save_all_files_of_a_folder(target_dir_path, output_file_path):
    '''	This function lists all files in a directory and its subdirectories,
    and saves the output to a specified file.
        Parameters:
        target_dir_path: The path to the directory to start listing files from.
        output_file_path: The file where the output will be saved.
    '''
    print(f"Listing files in {target_dir_path} and saving to {output_file_path}\n")
    with open(output_file_path, 'w') as f:
        for root, dirs, files in os.walk(target_dir_path):
            level = root.replace(target_dir_path, '').count(os.sep)
            indent = ' ' * 4 * (level)
            f.write(f'{indent}{os.path.basename(root)}/\n')
            sub_indent = ' ' * 4 * (level + 1)
            for file in files:
                f.write(f'{sub_indent}{file}\n')


def split_and_write_dataset(df: pd.DataFrame, output_path: str, test_size: float, write: bool):
    """
    Split dataset into train/test and optionally write CSV files.
    Logic intentionally unchanged from original implementation.
    """
    if test_size >= 1.0:
        test_rows = len(df)
    else:
        test_rows = (max(1, int(math.floor(len(df) * test_size))) if len(df) > 0 else 0)
    test_df  = df.iloc[:test_rows].copy()
    train_df = df.iloc[test_rows:].copy()
    print(f"[INFO] Train size: {len(train_df)}, Test size: {len(test_df)}")
    if write:
        print("[INFO] Writing TRAIN dataset chunks")
        csv_writer(train_df, output_path, prefix="train")
        print("[INFO] Writing TEST dataset chunks")
        csv_writer(test_df, output_path, prefix="test")
    else:
        print("[INFO] Write disabled — CSV files not written")
    return train_df, test_df

def write_metadata(
    output_path: str,
    main_arch: str,
    dataset_name: str,
    selected_topologies: list,
    df_origin: pd.DataFrame,
    extra: dict | None = None,
):
    metadata = {
        "architecture": main_arch,
        "dataset_name": dataset_name,
        "topology_indices": selected_topologies,
        "num_rows": int(df_origin.shape[0]),
        "num_columns": int(df_origin.shape[1]),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_root": os.path.join("Files", main_arch, "CSV", "main", "results"),
    }
    if extra:
        metadata.update(extra)
    metadata_path = os.path.join(output_path, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"[INFO] Metadata written to: {metadata_path}")
