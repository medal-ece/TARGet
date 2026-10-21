# code/utils/file_io.py
"""Filesystem and CSV helpers used across the training pipeline."""

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
from pathlib import Path

# Paths used by one experiment run.
@dataclass
class Paths:
    """Project paths resolved for one experiment run."""
    root: str
    dataset_root: str
    train_data_paths: list
    test_data_paths: list
    weights_path: str
    results_path: str
    images_path: str
    run_result_dir: str
    result_log_file: str


# Create the directory layout used by the experiment.
def create_paths(main_arch, main_topology, experiment_id=None, spec_name=None) -> Paths:
    """Resolve and create the directory structure used by one experiment run."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    # Dataset directory (same as original script)
    dataset_root = os.path.join(project_root, "dataset", main_arch, "main", main_topology if main_topology else "")
    # CSV paths
    train_data_paths = sorted(glob.glob(os.path.join(dataset_root, "train_*.csv")))
    test_data_paths = sorted(glob.glob(os.path.join(dataset_root, "test_*.csv")))
    # weights: Where to save and restore the trained model
    weights_path = os.path.join(project_root, "weights", main_arch, main_topology)
    # images: Where to save images/plots
    images_path = os.path.join(project_root, "images")
    # results: Where to save the final results
    results_path = os.path.join(project_root, "results", main_arch)
    # Run directory
    run_result_dir = os.path.join(results_path, f"{experiment_id}_{safe_slug(spec_name if spec_name else '')}")
    result_log_file = os.path.join(run_result_dir, "_MainModeling-RUN.txt")
    if experiment_id is not None:
        os.makedirs(run_result_dir, exist_ok=True)
    # Create directories
    for p in [weights_path, results_path, images_path]:
        os.makedirs(p, exist_ok=True)
    return Paths(
        root=project_root,
        dataset_root=dataset_root,
        train_data_paths=train_data_paths,
        test_data_paths=test_data_paths,
        weights_path=weights_path,
        results_path=results_path,
        images_path=images_path,
        run_result_dir=run_result_dir,
        result_log_file=result_log_file
    )


# Append plain text to a log file.
def append_text_to_file(text, filepath: str):
    """Append text to a file after normalizing non-string inputs."""
    if text is None:
        text = "<None>"
    if not isinstance(text, str):
        text = str(text)
    p = Path(filepath)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(text if text.endswith("\n") else text + "\n")


# Write large DataFrames in smaller CSV chunks.
def csv_writer(dframe, output_path, prefix="", chunk_size_mb=100):
    """Write a DataFrame into multiple CSV chunks sized by an approximate memory target."""
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

# Estimate how many rows fit in one CSV chunk.
def estimate_rows_per_chunk(dframe, chunk_mb=100, sample_rows=5000):
    """Estimate how many rows fit into one CSV chunk of the requested size."""
    if len(dframe) == 0:
        return 0
    n = min(len(dframe), sample_rows)
    buf = io.StringIO()
    dframe.iloc[:n].to_csv(buf, index=False)
    bytes_per_row = max(1, len(buf.getvalue().encode("utf-8")) / n)
    target_bytes = chunk_mb * 1024 * 1024
    return max(1, int(target_bytes / bytes_per_row))


# Save predictions for one frequency to a separate CSV file.
def save_preds_per_frequency(results_dir, model_type, freq_label, y_pred, yscaler, spec_name, overwrite=True):
    """Save predictions for one frequency into a dedicated CSV file."""
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
    """Return a compact human-readable summary of the selected frequencies."""
    return "Freqs=[" + ",".join([f"{freq/1e9:.2f}GHz" for freq in freq_list]) + "]"


def print_cuda_info():
    """Print Python, TensorFlow, CUDA, and GPU availability information."""
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


def safe_slug(x):
    """Make a filesystem-safe slug for filenames."""
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', str(x))


# Append rows to a CSV file while allowing the schema to grow.
def data_to_csv_saver(csv_path, new_data: dict):
    """Append a row to a CSV file while expanding the schema when new keys appear."""
    fields = []
    if os.path.exists(csv_path):
        with open(csv_path, newline='', encoding="utf-8") as f:
            fields = csv.DictReader(f).fieldnames or []
    # union of existing + new keys, keeping order
    all_fields = list(dict.fromkeys(fields + list(new_data.keys())))
    # prepare row with all fields present
    row = {f: new_data.get(f, "") for f in all_fields}
    if not fields or set(fields) != set(all_fields):
        # If the header changes, rewrite the file so all rows share the same columns.
        rows = []
        if fields:
            # Backfill any new columns for rows that were already on disk.
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
        # When the header already matches, we can append directly.
        with open(csv_path, "a", newline='', encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fields).writerow(row)

