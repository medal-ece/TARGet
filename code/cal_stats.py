# code/cal_stats.py
"""Calculate summary statistics across all frequencies for one architecture."""

import glob
from pathlib import Path
import numpy as np
import pandas as pd
from freq_list import all_frequencies_list

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# Update these values before running the script directly.
ARCH = "LNA"  # "LNA" or "phase_shifter"
COLUMNS = None  # None to use defaults for the architecture
if ARCH == "LNA":
    TOPOLOGIES = "combo_0_1"  # None for all topologies
    COLUMNS = ["GA", "GP", "NF"]
else:
    TOPOLOGIES = "combo_0_1_2_3"
    COLUMNS = ["S11m", "S21m", "S22m"]
FREQ_MIN = 150e6
FREQ_MAX = 1e9
OUTPUT_FILENAME = "frequency_stats.csv"

ARCH_DEFAULT_COLUMNS = {
    "LNA": ["GA", "GP", "NF"],
    "phase_shifter": ["S11m", "S21m", "S22m"],
}
# Conversion to dB depends on whether a column represents magnitude or power.
COLUMN_DB_CONVERSION = {"GA": 20, "GP": 20, "NF": 10, "S11m": 20, "S21m": 20, "S22m": 20,}


def build_dataset_paths(project_root: Path, arch: str, raw_topologies: str | None):
    """Resolve dataset directories and collect matching train/test CSV files."""
    dataset_root = project_root / "dataset" / arch / "main"
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    if raw_topologies:
        selected_topologies = [name.strip() for name in raw_topologies.split(",") if name.strip()]
        topology_dirs = [dataset_root / name for name in selected_topologies]
    else:
        topology_dirs = [p for p in dataset_root.iterdir() if p.is_dir()]
    csv_paths = []
    for topo_dir in topology_dirs:
        if not topo_dir.exists():
            print(f"[WARN] Topology directory missing: {topo_dir}")
            continue
        train_paths = sorted(glob.glob(str(topo_dir / "train_*.csv")))
        test_paths = sorted(glob.glob(str(topo_dir / "test_*.csv")))
        csv_paths.extend(train_paths)
        csv_paths.extend(test_paths)
    # Also support CSV files stored directly under the architecture root.
    csv_paths.extend(sorted(glob.glob(str(dataset_root / "train_*.csv"))))
    csv_paths.extend(sorted(glob.glob(str(dataset_root / "test_*.csv"))))
    csv_paths = [str(Path(p).resolve()) for p in csv_paths]
    csv_paths = sorted(set(csv_paths))
    return dataset_root, topology_dirs, csv_paths

def load_and_filter_csvs(file_paths, freq_set):
    """Load CSV files, keep the requested frequencies, and concatenate the results."""
    combined_data = []
    if tqdm:
        progress = tqdm(file_paths, desc="Reading CSVs", unit="file", ncols=120)
    else:
        progress = enumerate(file_paths, start=1)
    for item in progress:
        if tqdm:
            file_path = item
            file_name = Path(file_path).name
            progress.set_postfix_str(f"Reading {file_name}")
        else:
            idx, file_path = item
            file_name = Path(file_path).name
            print(f"  [{idx}/{len(file_paths)}] Reading {file_name}...")
        try:
            df = pd.read_csv(file_path)
        except Exception as exc:
            msg = f"[ERROR] Could not read {file_name}: {exc}"
            if tqdm:
                progress.set_postfix_str(msg)
            else:
                print(f"    {msg}")
            continue
        if df.empty:
            msg = "Skipped empty file."
            if tqdm:
                progress.set_postfix_str(msg)
            else:
                print(f"      {msg}")
            continue
        df = df.dropna(axis=1, how="all")
        if "freq" in df.columns:
            df = df[df["freq"].isin(freq_set)]
            status_msg = f"Filtered to {len(df)} rows"
        else:
            status_msg = f"No freq column found; keeping {len(df)} rows"
        if tqdm:
            progress.set_postfix_str(status_msg)
        else:
            print(f"      {status_msg} after frequency selection.")
        if not df.empty:
            combined_data.append(df)
    if not combined_data:
        return pd.DataFrame()
    return pd.concat(combined_data, ignore_index=True)


def calculate_statistics(arch: str, columns: list[str], raw_topologies: str | None, freq_min: float, freq_max: float):
    """Calculate min, max, mean, and standard deviation for the selected columns."""
    project_root = Path(__file__).parent.parent.resolve()
    dataset_root, topology_dirs, file_paths = build_dataset_paths(project_root, arch, raw_topologies)
    if not file_paths:
        raise FileNotFoundError(f"No CSV files found for architecture '{arch}' under {dataset_root}")
    freq_list = [f for f in all_frequencies_list() if freq_min <= f <= freq_max]
    freq_set = set(freq_list)
    print(f"Architecture: {arch}")
    print(f"Dataset root: {dataset_root}")
    print(f"Topologies: {', '.join([p.name for p in topology_dirs])}")
    print(f"Selected columns: {', '.join(columns)}")
    print(f"Frequency range: {freq_min/1e6:.0f} MHz to {freq_max/1e9:.1f} GHz")
    print(f"Reading {len(file_paths)} CSV files...\n")
    all_data_df = load_and_filter_csvs(file_paths, freq_set)
    if all_data_df.empty:
        raise ValueError("No data available after filtering by frequency and architecture.")
    # Convert selected outputs to dB before summarizing them.
    for col in columns:
        if col in all_data_df.columns:
            multiplier = COLUMN_DB_CONVERSION.get(col, 20)  # default to 20 if not specified
            all_data_df[col] = multiplier * np.log10(np.maximum(all_data_df[col], 1e-15))
    print(f"\nTotal rows combined from all frequencies: {len(all_data_df)}\n")
    stats_funcs = {"MIN": np.min, "MAX": np.max, "AVG": np.mean, "SD": np.std,}
    base_info = {
        "Architecture": arch,
        "Topologies": ", ".join([p.name for p in topology_dirs]),
        "Description": f"All Frequencies ({freq_min/1e6:.0f}MHz - {freq_max/1e9:.1f}GHz)",
        "Total_Samples": len(all_data_df)
    }
    rows = []
    for col in columns:
        row = base_info.copy()
        row["Column"] = col
        if col not in all_data_df.columns:
            print(f"[WARNING] Column '{col}' not found in combined data")
            row.update({"MIN": np.nan, "MAX": np.nan, "AVG": np.nan, "SD": np.nan})
        else:
            valid_values = all_data_df[col].dropna()
            if valid_values.empty:
                print(f"[WARNING] No valid values found for column '{col}'")
                row.update({"MIN": np.nan, "MAX": np.nan, "AVG": np.nan, "SD": np.nan})
            else:
                for stat_name, stat_func in stats_funcs.items():
                    row[stat_name] = float(stat_func(valid_values))
        rows.append(row)
    return pd.DataFrame(rows)


def save_results(results_df: pd.DataFrame, output_filename: str):
    """Append the computed statistics table to the configured output CSV."""
    output_path = Path(__file__).parent.parent.resolve() / "dataset" / output_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        results_df.to_csv(output_path, mode='a', header=False, index=False)
    else:
        results_df.to_csv(output_path, index=False)
    print(f"\nResults appended to: {output_path}")
    return output_path


def display_results(results_df: pd.DataFrame, columns: list[str]):
    """Print the statistics table in a console-friendly layout."""
    print("\n" + "=" * 100)
    print("STATISTICS FOR ALL FREQUENCIES COMBINED (in dB)")
    print("=" * 100)
    if results_df.empty:
        print("No results available.")
        return
    # The first row carries the shared summary metadata.
    first_row = results_df.iloc[0]
    print("\nGeneral summary:")
    print(f"Total Samples: {int(first_row['Total_Samples'])}")
    print(f"Architecture: {first_row.get('Architecture', '')}")
    print(f"Topologies: {first_row.get('Topologies', '')}")
    print(f"Description: {first_row.get('Description', '')}")
    print("\n" + "=" * 100)
    print("DETAILED BREAKDOWN")
    print("=" * 100)
    print("Column breakdown:")
    header = f"{'Column':<12} {'MIN':>12} {'MAX':>12} {'AVG':>12} {'SD':>12}"
    print(header)
    print("-" * len(header))
    for _, row in results_df.iterrows():
        col = row['Column']
        min_val = row.get('MIN', np.nan)
        max_val = row.get('MAX', np.nan)
        avg_val = row.get('AVG', np.nan)
        sd_val = row.get('SD', np.nan)
        print(f"{col:<12} {min_val:12.6f} {max_val:12.6f} {avg_val:12.6f} {sd_val:12.6f}")


if __name__ == "__main__":
    if COLUMNS is None:
        columns = ARCH_DEFAULT_COLUMNS[ARCH]
    elif isinstance(COLUMNS, str):
        columns = [col.strip() for col in COLUMNS.split(",") if col.strip()]
    else:
        columns = COLUMNS  # assume it's a list
    results = calculate_statistics(arch=ARCH, columns=columns, raw_topologies=TOPOLOGIES, freq_min=FREQ_MIN, freq_max=FREQ_MAX)
    display_results(results, columns)
    save_results(results, OUTPUT_FILENAME)
