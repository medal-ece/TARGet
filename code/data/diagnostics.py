import os
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from data.preprocessors import PCA_calc, phasor_to_cartesian
from utils.plotting import PCA_plot, scatter_plotter


def plot_input_scatterplots(data_by_freq_tr, data_by_freq_te, freq_list, save_root, main_arch, cartesian=False):
    sc_save_path = os.path.join(save_root, "scatterplots")
    os.makedirs(sc_save_path, exist_ok=True)

    yellow_red = LinearSegmentedColormap.from_list("yellow_red", ["yellow", "red"])
    # input_cols = ["s1_s11", "s1_s12", "s2_s11", "s2_s12"]

    for freq in freq_list:
        # Decide S-parameter prefixes and whether S21 plots are available
        if main_arch == "phase_shifter":
            input_cols = ["s1_s11", "s1_s12", "s2_s11", "s2_s12"]
        elif main_arch == "LNA":
            input_cols = ["in_s11", "in_s12", "out_s11", "out_s12"]
        else:
            print(f"[plot_input_scatterplots] Unknown main_arch='{main_arch}' at {freq/1e9} GHz; skipping.")
            continue
        # -------------------------
        # Process S11, S12 groups
        # -------------------------
        labels = ["TrainDS", "TestDS"]
        for i, df in enumerate([data_by_freq_tr[freq], data_by_freq_te[freq]]):
            for var in input_cols:
                abs_col = f"{var}_abs"
                ph_col  = f"{var}_ph"
                # skip if missing even one column
                if abs_col not in df or ph_col not in df:
                    print(f"Skipping {var} on {labels[i]} df at {freq/1e9}GHz due to missing columns.")
                    continue
                # Convert to cartesian
                x, y = phasor_to_cartesian(df[abs_col], df[ph_col])
                df_new = pd.DataFrame({"x": x, "y": y})
                # repetition count
                df_new["reps"] = df_new.groupby(["x", "y"])["x"].transform("size")
                # print unique counts
                row_counts = df_new.value_counts().reset_index(name="count")
                # print(f"{var} Parameter details ({labels[i]} @ {freq/1e9}GHz):")
                # print(f"Len: {len(df_new)}\nUnique Counts:\n{row_counts}\n")
                # scatter plot
                scatter_plotter(df_new["x"], df_new["y"], save_path=sc_save_path, name=f"{str(freq/1e9)}GHz_{labels[i]}_{var}", color_values=df_new["reps"], cmap=yellow_red)
                # --- overwrite dataset if Cartesian mode ---
                if cartesian:
                    df[f"{var}_x"] = df_new["x"]
                    df[f"{var}_y"] = df_new["y"]
                    df.drop(columns=[abs_col, ph_col], inplace=True)
        if main_arch == "phase_shifter":
            # S21 visualization
            x, y = phasor_to_cartesian(data_by_freq_tr[freq]["S21m"], data_by_freq_tr[freq]["S21p"])
            df_new = pd.DataFrame({"x": x, "y": y})
            df_new["reps"] = df_new.groupby(["x", "y"])["x"].transform("size")
            row_counts = df_new.value_counts().reset_index(name="count")
            # print(f"S21 Parameter details ({labels[i]} @ {freq/1e9}GHz):")
            # print(f"Len: {len(df_new)}\nUnique Counts:\n{row_counts}\n")
            scatter_plotter(df_new["x"], df_new["y"], save_path=sc_save_path, name=f"{freq/1e9}GHz_{labels[i]}_S21", color_values=df_new["reps"], cmap=yellow_red)


def run_pca_diagnostics(df_noinf: pd.DataFrame, feature_list: list, images_path: str, main_arch: str, dataset_name: str):
    """
    Run PCA diagnostics using df_noinf (pre-NaN-dropping dataframe).
    Logic intentionally unchanged.
    """
    assert df_noinf is not None, "df_noinf must be provided for PCA diagnostics"
    # ===== PCA: all features =====
    X_all = df_noinf[feature_list].copy()
    pca_all = PCA_calc(X_all)
    PCA_plot(pca_all, images_path, f"PCA_{main_arch}_{dataset_name}_total")
    # ===== PCA: S1 features =====
    X_s1 = df_noinf[feature_list[:4]].copy()
    pca_s1 = PCA_calc(X_s1)
    PCA_plot(pca_s1, images_path, f"PCA_{main_arch}_{dataset_name}_S1")
    # ===== PCA: S2 features =====
    X_s2 = df_noinf[feature_list[4:]].copy()
    pca_s2 = PCA_calc(X_s2)
    PCA_plot(pca_s2, images_path, f"PCA_{main_arch}_{dataset_name}_S2")
