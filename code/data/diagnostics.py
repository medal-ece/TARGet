# code/data/diagnostics.py
"""Diagnostics for inspecting raw RF inputs before training."""

import os
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from data.preprocessors import phasor_to_cartesian
from utils.plotting import scatter_plotter


def plot_input_scatterplots(data_by_freq_tr, data_by_freq_te, freq_list, save_root, main_arch, cartesian=False):
    """Plot per-frequency input scatterplots and optionally convert stored inputs to Cartesian form."""
    sc_save_path = os.path.join(save_root, "scatterplots")
    os.makedirs(sc_save_path, exist_ok=True)

    yellow_red = LinearSegmentedColormap.from_list("yellow_red", ["yellow", "red"])

    for freq in freq_list:
        # Choose the input groups that exist for the active architecture.
        if main_arch == "phase_shifter":
            input_cols = ["s1_s11", "s1_s12", "s2_s11", "s2_s12"]
        elif main_arch == "LNA":
            input_cols = ["in_s11", "in_s12", "out_s11", "out_s12"]
        else:
            print(f"[plot_input_scatterplots] Unknown main_arch='{main_arch}' at {freq/1e9} GHz; skipping.")
            continue
        labels = ["TrainDS", "TestDS"]
        for i, df in enumerate([data_by_freq_tr[freq], data_by_freq_te[freq]]):
            for var in input_cols:
                abs_col = f"{var}_abs"
                ph_col  = f"{var}_ph"
                # Skip incomplete phasor pairs instead of failing the whole frequency.
                if abs_col not in df or ph_col not in df:
                    print(f"Skipping {var} on {labels[i]} df at {freq/1e9}GHz due to missing columns.")
                    continue
                x, y = phasor_to_cartesian(df[abs_col], df[ph_col])
                df_new = pd.DataFrame({"x": x, "y": y})
                # Count repeated points so denser regions show up clearly in the color scale.
                df_new["reps"] = df_new.groupby(["x", "y"])["x"].transform("size")
                scatter_plotter(df_new["x"], df_new["y"], save_path=sc_save_path, name=f"{str(freq/1e9)}GHz_{labels[i]}_{var}", color_values=df_new["reps"], cmap=yellow_red)
                # Overwrite the stored columns only when the pipeline expects Cartesian inputs.
                if cartesian:
                    df[f"{var}_x"] = df_new["x"]
                    df[f"{var}_y"] = df_new["y"]
                    df.drop(columns=[abs_col, ph_col], inplace=True)
        if main_arch == "phase_shifter":
            # The phase-shifter data also exposes S21 for a quick visual sanity check.
            x, y = phasor_to_cartesian(data_by_freq_tr[freq]["S21m"], data_by_freq_tr[freq]["S21p"])
            df_new = pd.DataFrame({"x": x, "y": y})
            df_new["reps"] = df_new.groupby(["x", "y"])["x"].transform("size")
            scatter_plotter(df_new["x"], df_new["y"], save_path=sc_save_path, name=f"{freq/1e9}GHz_{labels[i]}_S21", color_values=df_new["reps"], cmap=yellow_red)

