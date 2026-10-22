# data/loaders.py

import glob
import hashlib
import json
import sys
import os
import random
import re
from typing import List, Dict, Union, Optional
import numpy as np

import pandas as pd
from prettytable import PrettyTable
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone

from freq_list import all_frequencies_list  # original helper :contentReference[oaicite:2]{index=2}


def frequency_selector(mode: str = 'random', freq_count: Optional[int] = None, manual_freqs: Optional[List[float]] = None, seed_number: int = 42, print_freqs: bool = True) -> List[float]:
    """
    Select frequencies based on the specified mode.
    Args:
        mode: Selection mode - 'random', 'uniform', or 'manual'
        freq_count: Number of frequencies to select (for 'random' mode)
        manual_freqs: List of specific frequencies to use (for 'manual' mode)
        seed_number: Random seed for reproducible results
        print_freqs: Whether to print the selected frequencies
    Returns:
        List of selected frequencies, sorted
    """
    unique_freq_list = all_frequencies_list()
    if mode == 'random':
        if freq_count is None:
            raise ValueError("freq_count must be provided for 'random' mode")
        random.seed(seed_number)
        freq_list = random.sample(unique_freq_list, k=freq_count)
        title = f"Random {freq_count} Unique Frequencies"
    elif mode == 'uniform':
        if freq_count is None or freq_count < 1:
            raise ValueError("freq_count must be provided and >= 1 for 'uniform' mode")
        if freq_count == 1:
            freq_list = [unique_freq_list[0]]
        elif freq_count == 2:
            freq_list = [unique_freq_list[0], unique_freq_list[-1]]
        else:
            # Select freq_count - 2 points between first and last
            indices = np.linspace(0, len(unique_freq_list) - 1, freq_count, dtype=int)
            freq_list = [unique_freq_list[i] for i in indices]
        title = f"Uniform {len(freq_list)} Frequencies"
    elif mode == 'manual':
        if manual_freqs is None:
            raise ValueError("manual_freqs must be provided for 'manual' mode")
        # Validate that all manual frequencies are in the available list
        invalid_freqs = [f for f in manual_freqs if f not in unique_freq_list]
        if invalid_freqs:
            raise ValueError(f"Manual frequencies {invalid_freqs} are not in the available frequency list")
        freq_list = manual_freqs.copy()
        title = f"Manual {len(freq_list)} Frequencies"
    else:
        raise ValueError("mode must be 'random', 'uniform', or 'manual'")
    freq_list.sort()
    indices = [unique_freq_list.index(x) for x in freq_list]
    if print_freqs:
        FreqTab = PrettyTable(["#", "Freq Index", "Freq (GHz)"])
        FreqTab.title = title
        for i, f in enumerate(freq_list):
            FreqTab.add_row([i + 1, indices[i], f"{(f / 1e9):,}"])
        print(FreqTab)
        print("Total unique frequencies:", len(unique_freq_list))
        print("Total selected Frequencies:", len(freq_list), end="\n\n")
    return freq_list


def list_and_save_all_files_of_a_folder(target_dir_path: str, output_file_path: str) -> None:
    """
    List all files in a directory (recursively) and save the tree to a file.

    Directly moved from your original implementation. :contentReference[oaicite:4]{index=4}
    """
    print(f"Listing files in {target_dir_path} and saving to {output_file_path}\n")
    with open(output_file_path, "w", encoding="utf-8") as f:
        for root, dirs, files in os.walk(target_dir_path):
            level = root.replace(target_dir_path, "").count(os.sep)
            indent = " " * 4 * level
            f.write(f"{indent}{os.path.basename(root)}/\n")
            sub_indent = " " * 4 * (level + 1)
            for file in files:
                f.write(f"{sub_indent}{file}\n")


def _fingerprint_files(files_path: List[str]) -> str:
    """Fingerprint the chunk list so stale full-dataset caches are rebuilt."""
    digest = hashlib.sha256()
    for file_path in sorted(files_path):
        stat = os.stat(file_path)
        digest.update(os.path.basename(file_path).encode("utf-8"))
        digest.update(str(stat.st_size).encode("utf-8"))
        digest.update(str(stat.st_mtime_ns).encode("utf-8"))
    return digest.hexdigest()


def _read_full_csv_chunks(files_path: List[str]) -> pd.DataFrame:
    frames = []
    for file_path in files_path:
        df = pd.read_csv(file_path)
        frames.append(df.dropna(axis=1, how="all"))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _read_full_dataframe_with_cache(
    files_path: List[str],
    cache_dir: str | None = None,
    cache_name: str | None = None,
    use_cache: bool = False,
    rebuild_cache: bool = False,
) -> pd.DataFrame:
    if not files_path:
        return pd.DataFrame()
    if not use_cache or not cache_dir:
        return _read_full_csv_chunks(files_path)

    os.makedirs(cache_dir, exist_ok=True)
    cache_stem = cache_name or "dataset"
    cache_path = os.path.join(cache_dir, f"{cache_stem}_all.pkl")
    metadata_path = os.path.join(cache_dir, f"{cache_stem}_all.meta.json")
    fingerprint = _fingerprint_files(files_path)

    if not rebuild_cache and os.path.exists(cache_path) and os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            if metadata.get("fingerprint") == fingerprint:
                print(f"[dataset-cache] Loading {cache_name} cache: {cache_path}")
                return pd.read_pickle(cache_path)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            print(f"[dataset-cache] Ignoring unreadable cache metadata for {cache_name}: {exc}")

    print(f"[dataset-cache] Building {cache_name} cache from {len(files_path)} CSV chunk(s)...")
    df = _read_full_csv_chunks(files_path)
    df.to_pickle(cache_path)
    metadata = {
        "cache_name": cache_name,
        "fingerprint": fingerprint,
        "source_files": [os.path.basename(path) for path in sorted(files_path)],
        "rows": int(len(df)),
        "columns": list(df.columns),
    }
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(f"[dataset-cache] Wrote {cache_name} cache: {cache_path}")
    return df


def _group_by_selected_frequencies(df: pd.DataFrame, freq_list: List[float]) -> Dict[float, pd.DataFrame]:
    data_by_freq = {freq: [] for freq in freq_list}
    if df.empty:
        return {freq: pd.DataFrame() for freq in freq_list}

    freq_set = set(freq_list)
    df = df[df["freq"].isin(freq_set)]
    if not df.empty:
        for freq, group in df.groupby("freq", sort=False):
            if freq in freq_set:
                data_by_freq[freq].append(group)

    for freq in freq_list:
        if data_by_freq[freq]:
            data_by_freq[freq] = pd.concat(data_by_freq[freq], ignore_index=True)
        else:
            data_by_freq[freq] = pd.DataFrame()
    return data_by_freq


def read_csv_files(
    files_path: List[str],
    freq_list: List[float],
    cache_dir: str | None = None,
    cache_name: str | None = None,
    use_cache: bool = False,
    rebuild_cache: bool = False,
) -> Dict[float, pd.DataFrame]:
    """
    Read CSV files and group rows by frequency.
    - files_path: list of CSV paths (train_* or test_*).
    - freq_list: list of frequencies to keep.
    - Returns: dict {freq: concatenated DataFrame for that freq}.
    """
    df = _read_full_dataframe_with_cache(
        files_path,
        cache_dir=cache_dir,
        cache_name=cache_name,
        use_cache=use_cache,
        rebuild_cache=rebuild_cache,
    )
    return _group_by_selected_frequencies(df, freq_list)


def clean_and_convert(df, column="kf"):
    df = df[~df.eq(0).any(1)]  					# Remove rows with at least one zero in any column.
    df.reset_index(drop=True, inplace=True)  	# Reset index and removes the old index
    df[column] = np.log10(df[column])  			# Convert the specified column to logarithmic scale
    return df

def data_preparation(inputs, outputs, df_train, df_test, scaler=StandardScaler()):
    # # Expect df_train and df_test to be pre-concatenated DataFrames for the given freq
    # # Ensure float64 dtype for consistency
    # df_train = df_train.astype('float64')
    # df_test = df_test.astype('float64')    =====>>>> Commenting this part decreased the MAPE!

    # # TODO: why we have kf in the specs?
    # if outputs == ['kf']:
    #     df_train = clean_and_convert(df_train)
    #     df_test = clean_and_convert(df_test)

    # Creates a StandardScaler instance and fits it to x_all
    # df_all = pd.concat([df_test, df_train], ignore_index=True)
    # X_all = df_all[inputs].values
    # y_all = df_all[outputs].values
    # x_all = df_all[inputs]
    # y_all = df_all[outputs]
    # xscaler = scaler.fit(x_all) # Wrong
    # yscaler = scaler.fit(y_all) # Wrong
    # xscaler = scaler.__class__().fit(x_all)
    # yscaler = scaler.__class__().fit(y_all)
    xscaler = clone(scaler).fit(df_train[inputs])
    yscaler = clone(scaler).fit(df_train[outputs])


    x_train = xscaler.transform(df_train[inputs])
    y_train = yscaler.transform(df_train[outputs])
    # X_train = xscaler.transform(df_train[inputs].values)
    # y_train = yscaler.transform(df_train[outputs].values)

    x_test = xscaler.transform(df_test[inputs])
    y_test = yscaler.transform(df_test[outputs])
    # X_test = xscaler.transform(df_test[inputs].values)
    # y_test = yscaler.transform(df_test[outputs].values)
    # 3) Build scaled DataFrames (for GNN)
    df_train_scaled = df_train.copy()
    df_test_scaled  = df_test.copy()

    df_train_scaled.loc[:, inputs] = x_train
    df_train_scaled.loc[:, outputs] = y_train

    df_test_scaled.loc[:, inputs] = x_test
    df_test_scaled.loc[:, outputs] = y_test
    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_test": x_test,
        "y_test": y_test,
        "df_train_scaled": df_train_scaled,
        "df_test_scaled": df_test_scaled,
        "xscaler": xscaler,
        "yscaler": yscaler,
    }

def get_architecture_spec(main_arch: str):
    if main_arch == "phase_shifter":
        arch_features = [
            "s1_s11_abs","s1_s11_ph","s1_s12_abs","s1_s12_ph",
            "s1_s21_abs","s1_s21_ph","s1_s22_abs","s1_s22_ph",
            "s2_s11_abs","s2_s11_ph","s2_s12_abs","s2_s12_ph",
            "s2_s21_abs","s2_s21_ph","s2_s22_abs","s2_s22_ph",
        ]
        arch_targets = ["S11m", "S21m", "S21p", "S22m"]
        to_drop = [
            "s1_s21_abs", "s1_s21_ph", "s1_s22_abs", "s1_s22_ph",
            "s2_s21_abs", "s2_s21_ph", "s2_s22_abs", "s2_s22_ph",
            "S22m",
            "s2"
        ]
        # First sort the old columns:
        desired_col_order = ["freq", "s2"] + arch_features + arch_targets
    else:
        # Other architectures should be Adjusted later
        arch_features = [
            "in_s11_abs","in_s11_ph","in_s12_abs","in_s12_ph",
            "in_s21_abs","in_s21_ph","in_s22_abs","in_s22_ph",
            "out_s11_abs","out_s11_ph","out_s12_abs","out_s12_ph",
            "out_s21_abs","out_s21_ph","out_s22_abs","out_s22_ph",
        ]
        arch_targets = ["GA", "GP", "GT", "NF", "b1f", "kf"]
        to_drop = [
            "in_s21_abs","in_s21_ph","in_s22_abs","in_s22_ph",
            "out_s21_abs","out_s21_ph","out_s22_abs","out_s22_ph",
        ]
        desired_col_order = ["freq", "w"] + arch_features + arch_targets
    return arch_features, arch_targets, to_drop, desired_col_order

def extract_raw_dataset_id(csv_path: str) -> int | None:
    """Extract old raw input-match IDs from names like wmain_4_in_61.csv."""
    stem = os.path.splitext(os.path.basename(csv_path))[0]
    match = re.search(r"_in_(\d+)$", stem)
    return int(match.group(1)) if match else None


def extract_lna_width_and_input_id(csv_path: str) -> tuple[int | None, int | None]:
    """Extract width and input-match ID from names like wmain_4_in_61.csv."""
    stem = os.path.splitext(os.path.basename(csv_path))[0]
    match = re.search(r"^wmain_(\d+)_in_(\d+)$", stem)
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def add_lna_match_ids(df: pd.DataFrame, csv_path: str, expected_out_count: int = 64) -> pd.DataFrame:
    """Add explicit LNA input/output matching-network IDs to one old all/*.csv file.

    The old LNA all files are organized as:
      filename: wmain_<width>_in_<input_id>.csv
      rows: for each frequency, output IDs 1..64 in order.
    """
    width, input_id = extract_lna_width_and_input_id(csv_path)
    if width is None or input_id is None:
        raise ValueError(f"Cannot extract LNA width/input ID from filename: {csv_path}")

    out_ids = df.groupby("freq", sort=False).cumcount() + 1
    max_out = int(out_ids.max()) if len(out_ids) else 0
    if max_out != expected_out_count:
        print(
            f"[WARN] Expected {expected_out_count} output rows per frequency in "
            f"{os.path.basename(csv_path)}, saw max group size {max_out}"
        )

    df = df.copy()
    df["w"] = width
    df["in_topology_id"] = input_id
    df["out_topology_id"] = out_ids.astype(int)
    return df


def load_and_combine_topology_csvs(
    files_root: str,
    topologies: list,
    selected_topologies: list,
    topology_col: str,
    include_file_ids: set[int] | None = None,
    exclude_file_ids: set[int] | None = None,
    add_lna_structural_ids: bool = False,
):
    dfs = []
    total_train = 0
    total_test = 0
    for topo_idx in selected_topologies:
        topo_name = topologies[topo_idx]
        candidate_base = os.path.join(files_root, topo_name)
        if not os.path.isdir(candidate_base):
            print(f"[WARN] Missing topology directory: {candidate_base}")
            continue
        all_dirs = glob.glob(os.path.join(candidate_base, "**", "all"), recursive=True)
        for all_dir in all_dirs:
            train_path = os.path.join(all_dir, "train")
            test_path  = os.path.join(all_dir, "test")
            train_files = sorted(glob.glob(os.path.join(train_path, "*.csv")))
            test_files  = sorted(glob.glob(os.path.join(test_path, "*.csv")))
            if include_file_ids is not None or exclude_file_ids is not None:
                def keep_file(path):
                    file_id = extract_raw_dataset_id(path)
                    if file_id is None:
                        return False
                    if include_file_ids is not None and file_id not in include_file_ids:
                        return False
                    if exclude_file_ids is not None and file_id in exclude_file_ids:
                        return False
                    return True
                train_files = [path for path in train_files if keep_file(path)]
                test_files = [path for path in test_files if keep_file(path)]
            if not train_files and not test_files:
                continue
            print(f"[INFO] Using data for topology {topo_idx} ({topo_name}): {len(train_files)} train, {len(test_files)} test")
            total_train += len(train_files)
            total_test  += len(test_files)
            for csv_path in train_files + test_files:
                df_tmp = pd.read_csv(csv_path)
                if add_lna_structural_ids:
                    df_tmp = add_lna_match_ids(df_tmp, csv_path)
                df_tmp[topology_col] = topo_idx
                dfs.append(df_tmp)
    if not dfs:
        sys.exit("[ERROR] No CSV files found for selected topologies.")
    df_origin = pd.concat(dfs, ignore_index=True)
    print(f"[INFO] Total files read: {total_train} train, {total_test} test")
    print(f"[INFO] Combined dataset shape: {df_origin.shape}")
    return df_origin
