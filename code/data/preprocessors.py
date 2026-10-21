# code/data/preprocessors.py
"""Preprocessing utilities for cleaning and inspecting per-frequency datasets."""

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from typing import Tuple
from sklearn.preprocessing import StandardScaler

from prettytable import PrettyTable
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA

def add_topology_one_hot(df, topology_col, selected_topologies, prefix="topology_"):
    """
    Add one-hot topology columns while preserving the original topology index.
    For example, topology_index=2 becomes topology_2=1.
    """
    df = df.copy()
    for t in selected_topologies:
        col = f"{prefix}{t}"
        df[col] = (df[topology_col] == t).astype(int)
    return df


# Convert one magnitude/phase pair into x/y components.
def phasor_to_cartesian(abs_values: pd.Series, ph_values: pd.Series) -> Tuple[pd.Series, pd.Series]:
    # Turn magnitude and phase into x/y coordinates for plotting and modeling.
    """
    Convert magnitude/phase pairs to Cartesian real and imaginary components.
    """
    radians = np.deg2rad(ph_values)
    real = abs_values * np.cos(radians)
    imag = abs_values * np.sin(radians)
    return real, imag

# Check column types and coerce bad numeric values when needed.
def df_info_and_type_convert(df, print_info=True, convert_bad_types=True):
    """Print column-level data quality information and optionally coerce non-numeric values."""
    myTable = PrettyTable(["Column", "Unique Count", "NANs", "Infs", "Column Type", "Converting?"])
    myTable.title = f"Columns Specifications (Total Rows: {len(df):,})"
    for col in df.columns:
        type_string = df[col].dtype
        convert_string = "---"
        if convert_bad_types:
            if not pd.api.types.is_numeric_dtype(type_string):
                convert_string = "YES"
                df[col] = pd.to_numeric(df[col], errors="coerce")
        myTable.add_row([col, df[col].nunique(dropna=True), df[col].isna().sum(), df[col].isin([np.inf, -np.inf]).sum(), type_string, convert_string])
    if print_info:
        print(myTable)
    return df



def safe_boxplot(df, save_path, title_prefix):
    """Save a grid of boxplots sized to the number of input columns."""
    n = len(df.columns)
    ncols = 4
    nrows = math.ceil(n / ncols)
    ax = df.plot(kind="box", subplots=True, layout=(nrows, ncols), figsize=(4 * ncols, 3 * nrows))
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def preprocess_data_by_frequency(df, features, specifications, targets, to_drop, main_arch, images_path, images_suffix, config, draw_figures=True, fit_stats=None, return_stats=False):
    """Clean one frequency slice, generate optional diagnostics, and return the processed data."""
    # On the training split we learn the clipping bounds; on the test split we reuse them.
    # fit_stats == None → this is train, learn stats | otherwise → this is test, reuse stats
    is_fit_mode = fit_stats is None

    # Inspect types first so downstream filtering works on numeric values.
    df_type_convert = df_info_and_type_convert(df)
    # Remove exact duplicate rows before feature-level cleanup.
    df_nodups = df_type_convert.drop_duplicates().reset_index(drop=True)

    # Rebuild topology one-hot columns whenever the topology index is available.
    if config.TOPOLOGY_COL in df_nodups.columns:
        df_nodups = add_topology_one_hot(df_nodups, topology_col=config.TOPOLOGY_COL, selected_topologies=config.selected_topologies, prefix=config.TOPOLOGY_PREFIX)
    topology_oh_cols = [f"{config.TOPOLOGY_PREFIX}{t}" for t in config.selected_topologies]

    # Keep topology one-hot columns in the working feature list when present.
    features_expanded = features + [c for c in topology_oh_cols if c not in features]
    
    # Fail fast if the expected topology columns were not created.
    expected_cols = {f"{config.TOPOLOGY_PREFIX}{t}" for t in config.selected_topologies}
    actual_cols = set(df_nodups.columns)
    missing = expected_cols - actual_cols
    if missing:
        raise RuntimeError(f"[TOPOLOGY OHE ERROR] Missing columns after One Hot Encoding: {missing}")

    # Repeat the check when topology metadata is expected by downstream models.
    if config.TOPOLOGY_COL in df_nodups.columns:
        expected_cols = {f"{config.TOPOLOGY_PREFIX}{t}" for t in config.selected_topologies}
        actual_cols = set(df_nodups.columns)
        missing = expected_cols - actual_cols
        if missing:
            raise RuntimeError(f"[TOPOLOGY OHE ERROR] Missing columns after OHE: {missing}")

    # Drop constant columns, but keep topology metadata even in single-topology runs.
    const_cols = []
    for c in df_nodups.columns:
        if df_nodups[c].nunique(dropna=False) <= 1:
            if c == config.TOPOLOGY_COL:
                continue
            if c.startswith(config.TOPOLOGY_PREFIX):
                continue
            const_cols.append(c)
    df_noconstant = df_nodups.drop(columns=const_cols)

    # Shuffle deterministically so repeated experiments stay comparable.
    df_shuffle = df_noconstant.sample(frac=1, random_state=42).reset_index(drop=True)
    # Replace infinities with NaNs so missing-value handling can catch them.
    df_noinf = df_shuffle.replace([np.inf, -np.inf], np.nan)

    # Remove fully empty rows and columns that are mostly missing.
    df_noNan = df_noinf.dropna(how="all")
    df_noNan = df_noNan.dropna(axis=1, thresh=len(df_noNan)*0.5)
    cols_to_remove = [c for c in to_drop if c in df_noNan.columns]
    
    # Keep feature and specification lists aligned with the cleaned DataFrame.
    original_specs = specifications.copy()
    original_features_expanded = features_expanded.copy()
    specifications = [s for s in specifications if s in df_noNan.columns]
    features_expanded = [f for f in features_expanded if f in df_noNan.columns]
    
    # Report requested fields that disappeared during cleaning.
    missing_specs = [s for s in original_specs if s not in specifications]
    missing_features = [f for f in original_features_expanded if f not in features_expanded]
    if missing_specs:
        print(f"[WARNING] Missing specifications (excluded from analysis): {missing_specs}")
    if missing_features:
        print(f"[WARNING] Missing features (excluded from analysis): {missing_features}")
    
    if draw_figures:
        # Use fully valid rows for figures so the plots are easier to interpret.
        df_noNanInAll = df_noNan.dropna().reset_index(drop=True)
        df_nodupcols_all = df_noNanInAll.drop(columns=cols_to_remove)

        # Build correlation views for the full dataset, inputs, and targets.
        corr_full = df_noNanInAll.corr(numeric_only=True)
        corr_nodupcols = df_nodupcols_all.corr(numeric_only=True)
        if features_expanded:
            corr_inputs = df_noNanInAll[features_expanded].corr(numeric_only=True)
        else:
            corr_inputs = pd.DataFrame()
        if specifications:
            corr_y_params = df_noNanInAll[specifications].corr(numeric_only=True)
        else:
            corr_y_params = pd.DataFrame()

        fig, axs = plt.subplots(2, 2, figsize=(20, 16))  # 2x2 grid
        axs = axs.ravel()  # flatten to 1D for easy indexing
        sns.heatmap(corr_full, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, ax=axs[0])
        axs[0].set_title("Correlation Heatmap (Original Data)")
        sns.heatmap(corr_nodupcols, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, ax=axs[1])
        axs[1].set_title("Correlation Heatmap (After Dropping Redundant Columns)")
        if not corr_inputs.empty:
            sns.heatmap(corr_inputs, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, ax=axs[2])
            axs[2].set_title("Correlation Heatmap (Inputs Only)")
        else:
            axs[2].text(0.5, 0.5, "No input features available", ha='center', va='center')
            axs[2].set_title("Correlation Heatmap (Inputs Only)")
        if not corr_y_params.empty:
            sns.heatmap(corr_y_params, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, ax=axs[3])
            axs[3].set_title("Correlation Heatmap (y_params Only)")
        else:
            axs[3].text(0.5, 0.5, "No output specifications available", ha='center', va='center')
            axs[3].set_title("Correlation Heatmap (y_params Only)")
        plt.tight_layout()
        os.makedirs(os.path.join(images_path, "correlation"), exist_ok=True)
        save_path = os.path.join(images_path, "correlation", f"Correlation_heatmap_{main_arch}_{images_suffix}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        # Save distribution plots before outlier handling.
        df_nodupcols_all.hist(figsize=(12, 8), bins=50)
        plt.tight_layout()
        os.makedirs(os.path.join(images_path, "distribution"), exist_ok=True)
        save_path = os.path.join(images_path, "distribution", f"Columns_distribution_{main_arch}_{images_suffix}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        # Save boxplots before clipping or removing outliers.
        os.makedirs(os.path.join(images_path, "boxplot"), exist_ok=True)
        save_path = os.path.join(images_path, "boxplot", f"Columns_Boxplot_{main_arch}_{images_suffix}.png")
        safe_boxplot(df_nodupcols_all, save_path, "BeforeOutliers")

        plt.tight_layout()
        os.makedirs(os.path.join(images_path, "boxplot"), exist_ok=True)
        save_path = os.path.join(images_path, "boxplot", f"Columns_Boxplot_{main_arch}_{images_suffix}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    df_noNanInTargetCols = df_noNan.dropna(subset=targets).reset_index(drop=True)
    non_required_targets = list(set(specifications) - set(targets))
    df_nodupcols = df_noNanInTargetCols.drop(columns=cols_to_remove + non_required_targets)

    # Build several outlier masks so the selection rule can be changed in one place.
    z = (df_nodupcols - df_nodupcols.mean()) / df_nodupcols.std(ddof=0)
    z_thresh = 3.0
    mask_z = z.abs().gt(z_thresh).any(axis=1)

    # The IQR bounds are learned on train data and reused on test data.
    if is_fit_mode:
        Q1 = df_nodupcols.quantile(0.25)
        Q3 = df_nodupcols.quantile(0.75)
        IQR = Q3 - Q1
        lower_iqr = Q1 - 1.5 * IQR
        upper_iqr = Q3 + 1.5 * IQR
    else:
        lower_iqr = fit_stats["lower_iqr"]
        upper_iqr = fit_stats["upper_iqr"]
    mask_iqr = ((df_nodupcols < lower_iqr) | (df_nodupcols > upper_iqr)).any(axis=1)

    # Isolation Forest gives a model-based outlier view.
    X = df_nodupcols.fillna(df_nodupcols.median())
    iso = IsolationForest(n_estimators=200, contamination="auto", random_state=42)
    labels = iso.fit_predict(X)                  # -1 = outlier, 1 = inlier
    mask_if = (labels == -1)

    # Use IQR clipping by default because it is simple and stable.
    method = "iqr"
    if method == "z":
        outliers = mask_z
    elif method == "iqr":
        outliers = mask_iqr
    elif method == "isoforest":
        outliers = mask_if
    elif method == "union":
        outliers = mask_z | mask_iqr | mask_if
    elif method == "intersection":
        outliers = mask_z & mask_iqr

    # Clip by default so sample counts stay stable across frequencies.
    action = "clip"
    df_noOutlier = df_nodupcols.copy()
    if action == "remove":
        df_noOutlier = df_nodupcols[~outliers].copy()
    elif action == "clip":
        for c in (features_expanded + targets):
            df_noOutlier[c] = df_nodupcols[c].clip(lower=lower_iqr[c], upper=upper_iqr[c])
    df_noOutlier_copy = df_noOutlier.copy()

    # Optional domain-specific hard bounds can be enabled here later.
    bound_check = False
    if bound_check:
        # Validity rules — edit as needed
        # Bound checks for numeric columns
        invalid = pd.Series(False, index=df_noOutlier.index)
        numeric_bounds = {}
        for col, (lo, hi) in numeric_bounds.items():
            if col in df_noOutlier.columns:
                s = pd.to_numeric(df_noOutlier[col], errors="coerce")
                invalid |= (s < lo) | (s > hi) | s.isna()
        bad_rows = df_noOutlier[invalid]
        df_noOutlier = df_noOutlier[~invalid]
    
    if draw_figures:
        # Save the post-cleaning distributions for comparison.
        df_noOutlier.hist(figsize=(12, 8), bins=50)
        plt.tight_layout()
        os.makedirs(os.path.join(images_path, "distribution"), exist_ok=True)
        save_path = os.path.join(images_path, "distribution", f"Columns_distribution_{main_arch}_{images_suffix}.png")
        
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        # Save boxplots after outlier handling.
        safe_boxplot(df_noOutlier, save_path, "AfterOutliers")
        plt.tight_layout()
        os.makedirs(os.path.join(images_path, "boxplot"), exist_ok=True)
        save_path = os.path.join(images_path, "boxplot", f"Columns_Boxplot_{main_arch}_{images_suffix}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    stats = {
        "lower_iqr": lower_iqr,
        "upper_iqr": upper_iqr,
    }
    features_used = features_expanded
    if return_stats:
        return df_noOutlier, features_used, stats
    return df_noOutlier, features_used


def PCA_calc(x):
    """Fit PCA on scaled features and print the explained variance ratio."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(x)
    pca = PCA().fit(X_scaled)
    print("\nExplained variance:")
    for i, v in enumerate(pca.explained_variance_ratio_, start=1):
        print(f"Component {i}: {v:.4f}")
    return pca.explained_variance_ratio_


def detect_outliers_z(df: pd.DataFrame, num_cols: list, z_thresh: float = 3.0) -> pd.Series:
    """
    Return a boolean mask of rows flagged by a Z-score rule.
    """
    z = (df[num_cols] - df[num_cols].mean()) / df[num_cols].std(ddof=0)
    mask_z = z.abs().gt(z_thresh).any(axis=1)
    return mask_z

def detect_outliers_iqr(df: pd.DataFrame, num_cols: list, iqr_factor: float = 1.5):
    """
    Compute an IQR-based outlier mask and return the clipping bounds it uses.
    """
    Q1 = df[num_cols].quantile(0.25)
    Q3 = df[num_cols].quantile(0.75)
    IQR = Q3 - Q1
    lower_iqr = Q1 - iqr_factor * IQR
    upper_iqr = Q3 + iqr_factor * IQR
    mask_iqr = ((df[num_cols] < lower_iqr) | (df[num_cols] > upper_iqr)).any(axis=1)
    return mask_iqr, lower_iqr, upper_iqr

def detect_outliers_isoforest(df: pd.DataFrame, num_cols: list, contamination="auto", random_state: int = 42,) -> pd.Series:
    """
    Return an outlier mask produced by Isolation Forest.
    """
    X = df[num_cols].fillna(df[num_cols].median())
    iso = IsolationForest(n_estimators=200, contamination=contamination, random_state=random_state)
    labels = iso.fit_predict(X)   # -1 = outlier, 1 = inlier
    mask_if = (labels == -1)
    return mask_if

def combine_outlier_masks(mask_z: pd.Series | None, mask_iqr: pd.Series | None, mask_if: pd.Series | None, method: str = "iqr") -> pd.Series:
    """
    Combine multiple outlier masks using the requested decision rule.
    """
    if mask_z is None:
        mask_z = pd.Series(False, index=mask_iqr.index if mask_iqr is not None else mask_if.index)
    if mask_iqr is None:
        mask_iqr = pd.Series(False, index=mask_z.index)
    if mask_if is None:
        mask_if = pd.Series(False, index=mask_z.index)

    if method == "z":
        return mask_z
    elif method == "iqr":
        return mask_iqr
    elif method == "isoforest":
        return mask_if
    elif method == "union":
        return mask_z | mask_iqr | mask_if
    elif method == "intersection":
        return mask_z & mask_iqr & mask_if
    else:
        raise ValueError(f"Unknown outlier method: {method}")

def handle_outliers(df: pd.DataFrame, num_cols: list, outliers: pd.Series, lower_iqr: pd.Series | None = None, upper_iqr: pd.Series | None = None, action: str = "clip"):
    """
    Apply either row removal or value clipping to the supplied outlier mask.
    """
    df = df.copy()
    if action == "remove":
        df = df[~outliers].copy()
    elif action == "clip":
        if lower_iqr is None or upper_iqr is None:
            raise ValueError("IQR bounds are required for clipping.")
        for c in num_cols:
            df[c] = df[c].clip(lower=lower_iqr[c], upper=upper_iqr[c])
    else:
        raise ValueError(f"Unknown outlier action: {action}")
    return df
