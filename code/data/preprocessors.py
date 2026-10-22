# data/preprocessors.py

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from typing import Tuple, Dict, List
from sklearn.preprocessing import StandardScaler

from prettytable import PrettyTable
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA

from config import Config  # adjust import if needed

def add_topology_one_hot(df, topology_col, selected_topologies, prefix="topology_"):
    """
    Add one-hot encoded topology columns.
    Example: topology_index=2 → topology_2 = 1
    """
    df = df.copy()
    for t in selected_topologies:
        col = f"{prefix}{t}"
        df[col] = (df[topology_col] == t).astype(int)
    return df




# -----------------------------------------------------------------------------
#  Helper: Convert a single phasor column pair (abs, ph) → Cartesian (real, imag)
# -----------------------------------------------------------------------------
def phasor_to_cartesian(abs_values: pd.Series, ph_values: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """
    Convert magnitude-phase to real-imag components.
    EXACT logic from original implementation.  (Copied, not changed.)
    """
    radians = np.deg2rad(ph_values)
    real = abs_values * np.cos(radians)
    imag = abs_values * np.sin(radians)
    return real, imag


def ensure_cartesian_features(df: pd.DataFrame, required_features: List[str]) -> pd.DataFrame:
    """Create missing *_x/*_y columns from matching *_abs/*_ph phasor columns."""
    df = df.copy()
    bases = set()
    for feature in required_features:
        if feature.endswith("_x") or feature.endswith("_y"):
            bases.add(feature[:-2])
    for base in sorted(bases):
        x_col = f"{base}_x"
        y_col = f"{base}_y"
        abs_col = f"{base}_abs"
        ph_col = f"{base}_ph"
        if (x_col not in df.columns or y_col not in df.columns) and abs_col in df.columns and ph_col in df.columns:
            df[x_col], df[y_col] = phasor_to_cartesian(df[abs_col], df[ph_col])
    return df

# -----------------------------------------------------------------------------
#  Clean DataFrame types and column names
# -----------------------------------------------------------------------------
def df_info_and_type_convert(df, print_info=True, convert_bad_types=True):
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


# -----------------------------------------------------------------------------
#  Remove duplicate rows [considering all columns]
# -----------------------------------------------------------------------------
def remove_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates().reset_index(drop=True)


# -----------------------------------------------------------------------------
#  Remove constant / irrelevant features (columns)
#  Count NaN as a value so truly constant columns (all same or all NaN) are removed
# -----------------------------------------------------------------------------
def remove_constant_columns(df: pd.DataFrame) -> pd.DataFrame:
    const_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    return df.drop(columns=const_cols)


# -----------------------------------------------------------------------------
#  Detect & remove NaN / Inf rows
# -----------------------------------------------------------------------------
def remove_nan_and_inf_rows(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna().reset_index(drop=True)
    return df


# -----------------------------------------------------------------------------
#  Diagnostic Plotting (NO LOGIC CHANGES)
# -----------------------------------------------------------------------------
def plot_hist(df: pd.DataFrame, save_dir: str, prefix: str):
    os.makedirs(save_dir, exist_ok=True)
    for col in df.columns:
        plt.figure(figsize=(6, 4))
        df[col].plot(kind="hist", bins=50, title=f"{prefix} - {col}")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{prefix}_hist_{col}.png"))
        plt.close()


def plot_box(df: pd.DataFrame, save_dir: str, prefix: str):
    os.makedirs(save_dir, exist_ok=True)
    for col in df.columns:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x=df[col])
        plt.title(f"{prefix} - {col}")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{prefix}_box_{col}.png"))
        plt.close()


def plot_correlation(df: pd.DataFrame, save_dir: str, prefix: str):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(), cmap="coolwarm", annot=False)
    plt.title(f"{prefix} - Correlation Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_corr.png"))
    plt.close()


def safe_boxplot(df, save_path, title_prefix):
    n = len(df.columns)
    ncols = 4
    nrows = math.ceil(n / ncols)
    ax = df.plot(kind="box", subplots=True, layout=(nrows, ncols), figsize=(4 * ncols, 3 * nrows))
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def preprocess_data_by_frequency(df, features, specifications, targets, to_drop, main_arch, images_path, images_suffix, config, draw_figures=True, fit_stats=None, return_stats=False, diagnostic_split="data"):
    # fit_stats == None → this is train, learn stats | otherwise → this is test, reuse stats
    is_fit_mode = fit_stats is None
    # 1. Understand the Data
    # Inspect dataset size, column types, distributions.

    # 2. Handle Missing Values
    # Impute: mean, median, mode (Advanced: KNN imputation, regression-based imputation)

    # 3. Remove Duplicates & Bad Samples
    # Drop duplicate rows. Remove constant or irrelevant features. Detect and drop samples with corrupted/invalid values.

    # 4. Outlier Detection & Handling
    # Visualize with boxplots, histograms, scatter plots. Detect using z-score, IQR, or Isolation Forest. Either remove or clip to a threshold.

    # 5. Feature Selection / Dimensionality Reduction   =>  The final step to do.
    # Drop highly correlated features (corr > 0.95).
    # Use domain knowledge (important in RF engineering!).
    # Optionally apply PCA/UMAP if dataset is huge.

    # 6. Encoding Categorical Variables   ===>   If all your input features are already numeric (float, int), you don’t need any encoding step at all.
    # Label encoding (for tree models). One-hot encoding (for linear/NN models). Embeddings for high-cardinality categories.

    # 7. Feature Scaling
    # Standardization (z-score) → for SVM, logistic regression, neural nets. MinMax scaling → for distance-based models (kNN). Leave unscaled → for tree-based models (RF, XGBoost).

    # 8. Feature Engineering
    # Create interaction terms, polynomials, or domain-specific features.

    # 9. Balance the Dataset (if classification)
    # Check class distribution.
    # Oversample (SMOTE, ADASYN) or undersample.
    # Stratified train/test split.

    # 10. Train/Test Split (or Cross-Validation)
    # Always split before scaling to avoid leakage. Typical split: 70–80% train, 20–30% test. Use StratifiedKFold for classification.

    # 11. Pipeline Automation
    # Use sklearn.Pipeline to chain preprocessing + model training.
    # Ensures consistency between train/test.

    if getattr(config, "cartesian", False):
        df = ensure_cartesian_features(df, features)

    # ======= info =======
    df_type_convert = df_info_and_type_convert(df)
    # ======= drop duplicates rows (considering all columns) =======
    df_nodups = df_type_convert.drop_duplicates().reset_index(drop=True)

    # --- Always create topology one-hot columns if topology_index exists ---
    if config.TOPOLOGY_COL in df_nodups.columns:
        df_nodups = add_topology_one_hot(df_nodups, topology_col=config.TOPOLOGY_COL, selected_topologies=config.selected_topologies, prefix=config.TOPOLOGY_PREFIX)
    topology_oh_cols = [f"{config.TOPOLOGY_PREFIX}{t}" for t in config.selected_topologies]

    # Extend the feature list with topology_* columns (for all architectures)
    features_expanded = features + [c for c in topology_oh_cols if c not in features]
    
    # Always ensure that topology one-hot columns were created correctly
    expected_cols = {f"{config.TOPOLOGY_PREFIX}{t}" for t in config.selected_topologies}
    actual_cols = set(df_nodups.columns)
    missing = expected_cols - actual_cols
    if missing:
        raise RuntimeError(f"[TOPOLOGY OHE ERROR] Missing columns after One Hot Encoding: {missing}")

    # Always sanity-check that those columns exist if topology_index exists
    if config.TOPOLOGY_COL in df_nodups.columns:
        expected_cols = {f"{config.TOPOLOGY_PREFIX}{t}" for t in config.selected_topologies}
        actual_cols = set(df_nodups.columns)
        missing = expected_cols - actual_cols
        if missing:
            raise RuntimeError(f"[TOPOLOGY OHE ERROR] Missing columns after OHE: {missing}")

    # ======= Remove constant / irrelevant features (columns) =======
    # NOTE: Count NaN as a value so truly constant columns (all same or all NaN) are removed
    # NOTE: topology one-hot columns are SEMANTIC and must NEVER be dropped, even if constant (e.g. single-topology experiments).
    const_cols = []
    for c in df_nodups.columns:
        if df_nodups[c].nunique(dropna=False) <= 1:
            # ALWAYS keep topology_index (needed by GNN)
            if c == config.TOPOLOGY_COL:
                continue
            # ALWAYS keep topology_* one-hots, even if constant (single-topology runs)
            if c.startswith(config.TOPOLOGY_PREFIX):
                continue
            const_cols.append(c)
    df_noconstant = df_nodups.drop(columns=const_cols)

    # ======= shuffle (deterministic) =======
    df_shuffle = df_noconstant.sample(frac=1, random_state=42).reset_index(drop=True)
    # ===== Convert Infs to NaNs =====
    df_noinf = df_shuffle.replace([np.inf, -np.inf], np.nan)

    # Drop rows/columns if missing heavily.
    df_noNan = df_noinf.dropna(how="all") # Drop rows with all NaNs
    df_noNan = df_noNan.dropna(axis=1, thresh=len(df_noNan)*0.5) # Drop columns with >50% NaNs
    cols_to_remove = [c for c in to_drop if c in df_noNan.columns]
    
    # Filter specifications and features to only include columns that exist in the dataframe
    original_specs = specifications.copy()
    original_features_expanded = features_expanded.copy()
    specifications = [s for s in specifications if s in df_noNan.columns]
    features_expanded = [f for f in features_expanded if f in df_noNan.columns]
    
    # Warn if any specifications/features were excluded
    missing_specs = [s for s in original_specs if s not in specifications]
    missing_features = [f for f in original_features_expanded if f not in features_expanded]
    if missing_specs:
        print(f"[WARNING] Missing specifications (excluded from analysis): {missing_specs}")
    if missing_features:
        print(f"[WARNING] Missing features (excluded from analysis): {missing_features}")
    
    diagnostic_key = f"{main_arch}_{images_suffix}_{diagnostic_split}"

    if draw_figures:
        # We will not use the fully dropped NaNs dataframe, Some rows might be needed for certain outputs.
        # So, we will use df_noNanInAll for figures.
        df_noNanInAll = df_noNan.dropna().reset_index(drop=True) # Drop all NaNs
        # ======= drop duplicated or non-required columns =======
        df_nodupcols_all = df_noNanInAll.drop(columns=cols_to_remove)

        # ======= Correlation analysis =======
        # Full correlation
        corr_full = df_noNanInAll.corr(numeric_only=True)
        # After dropping redundant columns
        corr_nodupcols = df_nodupcols_all.corr(numeric_only=True)
        # Only INPUT_COLS
        if features_expanded:
            corr_inputs = df_noNanInAll[features_expanded].corr(numeric_only=True)
        else:
            corr_inputs = pd.DataFrame()
        # Only y_params
        if specifications:
            corr_y_params = df_noNanInAll[specifications].corr(numeric_only=True)
        else:
            corr_y_params = pd.DataFrame()

        # Plot Heatmap of correlations
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
        save_path = os.path.join(images_path, "correlation", f"Correlation_heatmap_{diagnostic_key}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        # Histograms for distributions
        df_nodupcols_all.hist(figsize=(12, 8), bins=50)
        plt.tight_layout()
        os.makedirs(os.path.join(images_path, "distribution"), exist_ok=True)
        save_path = os.path.join(images_path, "distribution", f"Columns_distribution_{diagnostic_key}_before_outliers.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        # Boxplots for outliers
        # df_nodupcols_all.plot(kind="box", subplots=True, layout=(3, 4), figsize=(12, 8))
        os.makedirs(os.path.join(images_path, "boxplot"), exist_ok=True)
        save_path = os.path.join(images_path, "boxplot", f"Columns_Boxplot_{diagnostic_key}_before_outliers.png")
        safe_boxplot(df_nodupcols_all, save_path, "BeforeOutliers")

    df_noNanInTargetCols = df_noNan.dropna(subset=targets).reset_index(drop=True)
    non_required_targets = list(set(specifications) - set(targets))
    df_nodupcols = df_noNanInTargetCols.drop(columns=cols_to_remove + non_required_targets)
    if not is_fit_mode:
        required_columns = fit_stats["lower_iqr"].index
        missing_required = [c for c in required_columns if c not in df_nodupcols.columns]
        if missing_required:
            raise ValueError(f"Preprocessing input is missing columns required by train-fitted stats: {missing_required}")
        df_nodupcols = df_nodupcols.reindex(columns=required_columns)

    # ---------------- Detect outliers ----------------
    # A) Z-score standardization method
    z = (df_nodupcols - df_nodupcols.mean()) / df_nodupcols.std(ddof=0)
    z_thresh = 3.0
    mask_z = z.abs().gt(z_thresh).any(axis=1)

    # B) IQR method
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

    # C) Isolation Forest (model-based)
    X = df_nodupcols.fillna(df_nodupcols.median())
    iso = IsolationForest(n_estimators=200, contamination="auto", random_state=42)
    labels = iso.fit_predict(X)                  # -1 = outlier, 1 = inlier
    mask_if = (labels == -1)

    # Choose how to flag outliers
    method = "iqr"   # options: "z", "iqr", "isoforest", "union", "intersection"
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

    # ---------------- Handle outliers ----------------
    action = "clip"  # options: "remove" or "clip"
    df_noOutlier = df_nodupcols.copy()
    if action == "remove":
        df_outlier = df_nodupcols[outliers].copy()    # keep for audit
        df_noOutlier = df_nodupcols[~outliers].copy()
    elif action == "clip":
        # Clip (winsorize) to thresholds
        # If using IQR: clip to IQR bounds
        for c in (features_expanded + targets):
            df_noOutlier[c] = df_nodupcols[c].clip(lower=lower_iqr[c], upper=upper_iqr[c])
    df_noOutlier_copy = df_noOutlier.copy()

    # Boundary Check:
    bound_check = False
    if bound_check:
        # Validity rules — edit as needed
        # Bound checks for numeric columns
        invalid = pd.Series(False, index=df_noOutlier.index)  # Build a mask of invalid rows
        numeric_bounds = {}                         # e.g., {"prob": (0.0, 1.0), "age": (0, 120)}
        for col, (lo, hi) in numeric_bounds.items():
            if col in df_noOutlier.columns:
                s = pd.to_numeric(df_noOutlier[col], errors="coerce")
                invalid |= (s < lo) | (s > hi) | s.isna()
        bad_rows = df_noOutlier[invalid]
        df_noOutlier = df_noOutlier[~invalid]
    
    if draw_figures:
        # Histograms for distributions
        df_noOutlier.hist(figsize=(12, 8), bins=50)
        plt.tight_layout()
        os.makedirs(os.path.join(images_path, "distribution"), exist_ok=True)
        save_path = os.path.join(images_path, "distribution", f"Columns_distribution_{diagnostic_key}_after_outliers.png")
        
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        # Boxplots after outlier handling
        os.makedirs(os.path.join(images_path, "boxplot"), exist_ok=True)
        save_path = os.path.join(images_path, "boxplot", f"Columns_Boxplot_{diagnostic_key}_after_outliers.png")
        safe_boxplot(df_noOutlier, save_path, "AfterOutliers")
    stats = {
        "lower_iqr": lower_iqr,
        "upper_iqr": upper_iqr,
    }
    features_used = features_expanded
    if return_stats:
        return df_noOutlier, features_used, stats
    return df_noOutlier, features_used



def PCA_calc(x):
    # ===== Scale + PCA =====
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(x)
    pca = PCA().fit(X_scaled)
    print("\nExplained variance:")
    for i, v in enumerate(pca.explained_variance_ratio_, start=1):
        print(f"Component {i}: {v:.4f}")
    return pca.explained_variance_ratio_


def detect_outliers_z(df: pd.DataFrame, num_cols: list, z_thresh: float = 3.0) -> pd.Series:
    """
    Return a boolean mask of outliers based on Z-score across num_cols.
    """
    # z = (df_nodupcols - df_nodupcols.mean()) / df_nodupcols.std(ddof=0)
    z = (df[num_cols] - df[num_cols].mean()) / df[num_cols].std(ddof=0)
    mask_z = z.abs().gt(z_thresh).any(axis=1)
    return mask_z

def detect_outliers_iqr(df: pd.DataFrame, num_cols: list, iqr_factor: float = 1.5):
    """
    Compute IQR-based outlier mask and the lower/upper bounds used.
    Returns: (mask_iqr, lower_iqr, upper_iqr)
    """
    # Q1 = df_nodupcols.quantile(0.25)
    # Q3 = df_nodupcols.quantile(0.75)
    Q1 = df[num_cols].quantile(0.25)
    Q3 = df[num_cols].quantile(0.75)
    IQR = Q3 - Q1
    lower_iqr = Q1 - iqr_factor * IQR
    upper_iqr = Q3 + iqr_factor * IQR
    mask_iqr = ((df[num_cols] < lower_iqr) | (df[num_cols] > upper_iqr)).any(axis=1)
    return mask_iqr, lower_iqr, upper_iqr

def detect_outliers_isoforest(df: pd.DataFrame, num_cols: list, contamination="auto", random_state: int = 42,) -> pd.Series:
    """
    Model-based outlier detection using IsolationForest.
    """
    X = df[num_cols].fillna(df[num_cols].median())
    iso = IsolationForest(n_estimators=200, contamination=contamination, random_state=random_state)
    labels = iso.fit_predict(X)   # -1 = outlier, 1 = inlier
    mask_if = (labels == -1)
    return mask_if

def combine_outlier_masks(mask_z: pd.Series | None, mask_iqr: pd.Series | None, mask_if: pd.Series | None, method: str = "iqr") -> pd.Series:
    """
    Combine different masks according to the chosen method.
    """
    # start from all-False
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
    Apply the chosen action to outliers: 'remove' or 'clip'.
    Returns a new DataFrame.
    """
    df = df.copy()
    if action == "remove":
        df_outliers = df[outliers].copy()    # you can log/save this if you want
        df = df[~outliers].copy()
    elif action == "clip":
        if lower_iqr is None or upper_iqr is None:
            raise ValueError("IQR bounds are required for clipping.")
        for c in num_cols:
            df[c] = df[c].clip(lower=lower_iqr[c], upper=upper_iqr[c])
    else:
        raise ValueError(f"Unknown outlier action: {action}")
    return df


def preprocess_dataset(df: pd.DataFrame, num_cols: list, to_drop: list, topology_col: str, random_state: int = 42):
    """
    Preprocess dataset.
    Returns:
        df_clean  : fully processed dataframe (used for training / writing)
        df_noinf  : dataframe with infs replaced by NaNs but BEFORE NaN-based row/column dropping
                    (used for PCA diagnostics only)
    """
    # --- basic cleaning ---
    df = df.copy()
    # remove duplicates (considering all columns)
    df_no_dups = df.drop_duplicates().reset_index(drop=True)
    # Remove constant / irrelevant features (columns)
    # Count NaN as a value so truly constant columns (all same or all NaN) are removed
    const_cols = [c for c in df_no_dups.columns if df_no_dups[c].nunique(dropna=False) <= 1 and c != topology_col]
    df_no_const = df_no_dups.drop(columns=const_cols)
    # shuffle (deterministic)
    df_shuffle = df_no_const.sample(frac=1, random_state=random_state).reset_index(drop=True)
    # Convert Infs to NaNs
    df_noinf = df_shuffle.replace([np.inf, -np.inf], np.nan)
    # Drop rows that are all NaN
    df_no_nan_step_1 = df_noinf.dropna(how="all")
    # Drop columns with too many NaNs (>50%)
    df_no_nan_step_2 = df_no_nan_step_1.dropna(axis=1, thresh=int(0.5 * len(df_no_nan_step_1)))
    # We will not save the fully dropped NaNs dataframe, Some rows might be needed for certain outputs.
    # So, we will use df_no_nan_step_3 for figures and df_no_nan_step_2 for further processing.
    df_no_nan_step_3 = df_no_nan_step_2.dropna().reset_index(drop=True) # Drop all NaNs
    # Drop duplicated or non-required (architecture-specific) columns
    df_no_dup_cols = df_no_nan_step_3.drop(columns=to_drop, errors="ignore")
    # Keep only numeric columns that actually exist
    num_cols = [c for c in num_cols if c in df_no_dup_cols.columns]
    # --- outlier detection ---
    mask_z = detect_outliers_z(df_no_dup_cols, num_cols)
    mask_iqr, lower_iqr, upper_iqr = detect_outliers_iqr(df_no_dup_cols, num_cols)
    mask_if = detect_outliers_isoforest(df_no_dup_cols, num_cols)
    method = "iqr"     # options: "z", "iqr", "isoforest", "union", "intersection"
    outliers = combine_outlier_masks(mask_z, mask_iqr, mask_if, method=method)
    action = "clip"   # options: "remove" or "clip"
    df_outliers = handle_outliers(df, num_cols=num_cols, outliers=outliers, lower_iqr=lower_iqr, upper_iqr=upper_iqr, action=action)
    print(f"[INFO] Preprocessing complete. Shape: {df.shape}")
    return df_noinf, df_no_nan_step_2, df_no_nan_step_3, df_outliers
