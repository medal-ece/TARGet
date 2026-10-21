# code/data/loaders.py
"""Dataset loading helpers used by the training pipeline."""

import random
from typing import List, Dict, Optional
import numpy as np

import pandas as pd
from prettytable import PrettyTable
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone

from freq_list import all_frequencies_list


def frequency_selector(mode: str = 'random', freq_count: Optional[int] = None, manual_freqs: Optional[List[float]] = None, seed_number: int = 42, print_freqs: bool = True) -> List[float]:
    """Return a sorted list of frequencies selected by random, uniform, or manual mode."""
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
            # Evenly sample across the full available range.
            indices = np.linspace(0, len(unique_freq_list) - 1, freq_count, dtype=int)
            freq_list = [unique_freq_list[i] for i in indices]
        title = f"Uniform {len(freq_list)} Frequencies"
    elif mode == 'manual':
        if manual_freqs is None:
            raise ValueError("manual_freqs must be provided for 'manual' mode")
        # Fail early if the caller requests unsupported frequencies.
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


def read_csv_files(files_path: List[str], freq_list: List[float]) -> Dict[float, pd.DataFrame]:
    """Load CSV files and group their rows into one DataFrame per selected frequency."""
    data_by_freq = {freq: [] for freq in freq_list}
    freq_set = set(freq_list)
    for file_path in files_path:
        df = pd.read_csv(file_path)
        df = df.dropna(axis=1, how="all")
        df = df[df["freq"].isin(freq_set)]
        if df.empty:
            continue
        for freq, group in df.groupby("freq", sort=False):
            if freq in freq_set:
                data_by_freq[freq].append(group)

    for freq in freq_list:
        if data_by_freq[freq]:
            data_by_freq[freq] = pd.concat(data_by_freq[freq], ignore_index=True)
        else:
            # Preserve the key even when a frequency has no matching rows.
            data_by_freq[freq] = pd.DataFrame()
    return data_by_freq


def data_preparation(inputs, outputs, df_train, df_test, scaler=StandardScaler()):
    """Scale train and test splits and return both NumPy arrays and DataFrame copies."""
    xscaler = clone(scaler).fit(df_train[inputs])
    yscaler = clone(scaler).fit(df_train[outputs])

    x_train = xscaler.transform(df_train[inputs])
    y_train = yscaler.transform(df_train[outputs])
    x_test = xscaler.transform(df_test[inputs])
    y_test = yscaler.transform(df_test[outputs])

    # Keep DataFrame copies so graph-based models can still access feature names.
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
    """Return the feature, target, and column-order settings for the selected architecture."""
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
        desired_col_order = ["freq", "s2"] + arch_features + arch_targets
    else:
        # Treat any non-phase-shifter architecture as the LNA layout for now.
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
