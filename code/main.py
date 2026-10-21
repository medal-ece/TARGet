# code/main.py
"""Entry point for dataset preparation, model training, and result analysis."""
import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
# Reduce TensorFlow log noise during long experiment runs.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow C++ backend logs (INFO & WARNING)
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

from config import Config
from models.factory import get_topology
from data.loaders import frequency_selector, read_csv_files, get_architecture_spec
from data.diagnostics import plot_input_scatterplots
from data.preprocessors import preprocess_data_by_frequency

from training.experiment import run_experiments
from analysis.plots import plot_utilization_from_logs
from analysis.table import print_results_table
from analysis.core import load_runs, aggregate_metrics
from utils.file_io import create_paths, print_cuda_info, make_freq_string, append_text_to_file

import getpass
import sys, time

import logging
import warnings
warnings.filterwarnings("ignore")
logging.getLogger("tensorflow").setLevel(logging.ERROR)


def main():
    """Run the full experiment workflow from data loading through analysis."""
    experiment_id = time.strftime("%Y%m%d_%H%M%S")
    user = getpass.getuser()
    config = Config()
    coordinate_string = "_xy" if config.cartesian else ""
    coord = "Cartesian" if config.cartesian else "Polar"
    topologies, feature_list, spec_list = get_topology(config.main_arch, cartesian=config.cartesian)
    selected_topologies = config.selected_topologies
    is_combo = len(selected_topologies) > 1
    if is_combo:
        config.main_topology = "combo_" + "_".join(map(str, selected_topologies))
    else:
        config.main_topology = topologies[selected_topologies[0]]

    config.features = feature_list
    config.specifications = spec_list
    # config.targets = [spec_list[0]]
    original_input_cols, original_output_cols, to_drop, desired_col_order = get_architecture_spec(config.main_arch)
    config.to_drop = to_drop
    paths = create_paths(config.main_arch, config.main_topology, experiment_id, config.targets[0])

    print("\n=== PATHS ===")
    print(f"Dataset Root: {paths.dataset_root}")
    print(f"Results Dir:  {paths.run_result_dir}")
    print(f"Log File:     {paths.result_log_file}")

    # Select experiment frequencies from the configured mode.
    if config.freq_mode == "random":
        config.freq_list = frequency_selector(mode='random', freq_count=config.freq_count)
    elif config.freq_mode == "uniform":
        config.freq_list = frequency_selector(mode='uniform', freq_count=config.freq_count)
    elif config.freq_mode == "manual":
        config.freq_list = frequency_selector(mode='manual', manual_freqs=config.manual_freqs)
    else:
        raise ValueError(f"Invalid freq_mode: {config.freq_mode}. Must be 'random', 'uniform', or 'manual'")
    
    freq_string = make_freq_string(config.freq_list)
    print(freq_string)
    append_text_to_file(freq_string + "\n", paths.result_log_file)

    # Load and group the raw CSV data before preprocessing.
    print("\nLoading CSV files...")
    READ_DS_start = time.time()
    data_by_freq_tr = read_csv_files(paths.train_data_paths, config.freq_list)
    data_by_freq_te = read_csv_files(paths.test_data_paths, config.freq_list)
    print(f"Reading the dataset time duration: {(time.time() - READ_DS_start):.1f} s\n")

    plot_input_scatterplots(data_by_freq_tr, data_by_freq_te, config.freq_list, paths.images_path, config.main_arch, config.cartesian)
    if config.preprocess:
        for freq in config.freq_list:
            print(f"\n\n{'------'*10}\nPreprocessing for Frequency: {str(freq / 1e9)} GHz\n")
            img_sfx = f"{config.main_topology}_{freq/1e9}GHz{coordinate_string}"
            df_tr = data_by_freq_tr[freq]
            df_te = data_by_freq_te[freq]

            print(f"\nTrain Dataframe Info:\nShape: {df_tr.shape}\n")
            df_tr, features_used, prep_stats = preprocess_data_by_frequency(df_tr, config.features, config.specifications, config.targets, config.to_drop, config.main_arch, paths.images_path, img_sfx, config, True, fit_stats=None, return_stats=True)
            config.features = features_used
            print(f"\nTest Dataframe Info:\nShape: {df_te.shape}\n")
            df_te, _ = preprocess_data_by_frequency(df_te, config.features, config.specifications, config.targets, config.to_drop, config.main_arch, paths.images_path, img_sfx, config, True, fit_stats=prep_stats, return_stats=False)
            data_by_freq_tr[freq] = df_tr
            data_by_freq_te[freq] = df_te
    run_experiments(
        experiment_id=experiment_id,
        config=config,
        paths=paths,
        data_by_freq_tr=data_by_freq_tr,
        data_by_freq_te=data_by_freq_te,
        coord=coord,
        coordinate_string=coordinate_string,
        user=user,
    )
    if config.run_analysis_after_experiment:
        MODEL_BASE_COLORS = config.MODEL_BASE_COLORS
        csv_path = os.path.join(paths.run_result_dir, "all_repetitions.csv")
        pred_csv_path = os.path.join(paths.run_result_dir, "all_predictions.csv")
        analysis_out = os.path.join(paths.run_result_dir, "analysis")
        os.makedirs(analysis_out, exist_ok=True)
        df = load_runs(csv_path)
        agg = aggregate_metrics(df, group_cols=["model_type", "frequency", "utilization"])
        agg.to_csv(os.path.join(analysis_out, "aggregated_metrics.csv"), index=False)
        plot_utilization_from_logs(csv_path, pred_csv_path, analysis_out, experiment_id, title="", MODEL_BASE_COLORS=MODEL_BASE_COLORS, plot_show=config.plot_show)
        print_results_table(csv_path, experiment_id)

if __name__ == "__main__":
    print_cuda_info()
    try:
        main()
        status = "SUCCESS"
        msg = "Experiment finished successfully"
    except Exception as e:
        status = "FAILED"
        msg = str(e)
        raise
    finally:
        print(f"\nExperiment Status: {status}")
    sys.exit(0)
   
