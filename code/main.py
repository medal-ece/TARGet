# main.py
import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
# "0" = all logs, "1" = filter INFO, "2" = filter INFO+WARNING, "3" = filter INFO+WARNING+ERROR
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow C++ backend logs (INFO & WARNING)
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

# Self imports
from config import Config
from models.factory import get_topology
from data.loaders import frequency_selector, read_csv_files, get_architecture_spec
from data.diagnostics import plot_input_scatterplots
from data.preprocessors import preprocess_data_by_frequency

from training.experiment import run_experiments
from analysis.plots import plot_utilization_from_logs, plot_heldout_from_logs
from analysis.table import print_results_table
from analysis.core import load_runs, aggregate_metrics
from analysis.reports import (
    summarize_data_by_frequency,
    write_experiment_dataset_report,
    write_rebuttal_reports,
)

# Standard imports
# for new version of keras:
# from keras.src.legacy.saving import legacy_h5_format
import getpass
import sys, time

# Ignore all warnings
import logging
import warnings
warnings.filterwarnings("ignore")   # Suppress Python warnings
logging.getLogger("tensorflow").setLevel(logging.ERROR)     # 3) Suppress TensorFlow Python logger
import subprocess


from utils.file_io import create_paths, print_cuda_info, make_freq_string, append_text_to_file

def main(config=None, experiment_id=None):
    if experiment_id is None:
        experiment_id = time.strftime("%Y%m%d_%H%M%S")
    user = getpass.getuser()
    if config is None:
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
    # Paths
    paths = create_paths(
        config.main_arch,
        config.main_topology,
        experiment_id,
        config.targets[0],
        dataset_version=config.dataset_version,
        dataset_split=config.dataset_split,
        dataset_variant=config.dataset_variant,
        run_heldout_eval=config.run_heldout_eval,
    )
    heldout_paths = None
    if config.run_heldout_eval:
        heldout_paths = create_paths(
            config.main_arch,
            config.main_topology,
            experiment_id=None,
            spec_name=config.targets[0],
            dataset_version=config.dataset_version,
            dataset_split=config.heldout_dataset_split,
            dataset_variant=config.dataset_variant,
        )

    print("\n=== PATHS ===")
    print(f"Dataset Version: {paths.dataset_version}")
    if paths.dataset_variant:
        print(f"Dataset Variant: {paths.dataset_variant}")
    print(f"Dataset Split:   {paths.dataset_split}")
    print(f"Dataset Root: {paths.dataset_root}")
    if heldout_paths is not None:
        print(f"Heldout Root: {heldout_paths.dataset_root}")
    print(f"Results Dir:  {paths.run_result_dir}")
    print(f"Log File:     {paths.result_log_file}")

    if not paths.train_data_paths:
        raise FileNotFoundError(
            "No train_*.csv files found for the configured main dataset root: "
            f"{paths.dataset_root}. Check dataset_version, dataset_variant, dataset_split, "
            "main_arch, and selected_topologies."
        )
    if not paths.test_data_paths:
        raise FileNotFoundError(
            "No test_*.csv files found for the configured main dataset root: "
            f"{paths.dataset_root}. Check dataset_version, dataset_variant, dataset_split, "
            "main_arch, and selected_topologies."
        )
    if heldout_paths is not None and not heldout_paths.test_data_paths:
        raise FileNotFoundError(
            "No test_*.csv files found for the configured heldout dataset root: "
            f"{heldout_paths.dataset_root}. Check heldout_dataset_split and dataset_variant."
        )

    # list_and_save_all_files_of_a_folder("./Files", "./directory_listing.txt")

    # all_files = paths.test_data_paths + paths.train_data_paths # The order matters. Don't change the order.
    # print(f"\nCSV files Names ({len(all_files)} files in total):")
    # print("Test files:\n\t" + ',\n\t'.join(paths.test_data_paths))
    # print("Train files:\n\t" + ',\n\t'.join(paths.train_data_paths), "\n")

    # Select frequencies based on config
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

    # Load raw CSV data and select the desired columns:
    # Probably for LNA_Cascode we'll use 'wmain_4_in_1.csv' instead of 's1_1.csv'
    print("\nLoading CSV files...")
    READ_DS_start = time.time()
    dataset_cache_dir = os.path.join(paths.dataset_root, ".cache")
    data_by_freq_tr = read_csv_files(
        paths.train_data_paths,
        config.freq_list,
        cache_dir=dataset_cache_dir,
        cache_name="train",
        use_cache=config.use_dataset_cache,
        rebuild_cache=config.rebuild_dataset_cache,
    )
    data_by_freq_te = read_csv_files(
        paths.test_data_paths,
        config.freq_list,
        cache_dir=dataset_cache_dir,
        cache_name="test",
        use_cache=config.use_dataset_cache,
        rebuild_cache=config.rebuild_dataset_cache,
    )
    data_by_freq_heldout = None
    if heldout_paths is not None:
        heldout_cache_dir = os.path.join(heldout_paths.dataset_root, ".cache")
        data_by_freq_heldout = read_csv_files(
            heldout_paths.test_data_paths,
            config.freq_list,
            cache_dir=heldout_cache_dir,
            cache_name="test",
            use_cache=config.use_dataset_cache,
            rebuild_cache=config.rebuild_dataset_cache,
        )
    print(f"Reading the dataset time duration: {(time.time() - READ_DS_start):.1f} s\n")
    dataset_read_time_sec = time.time() - READ_DS_start
    loaded_dataset_summaries = {
        "train": summarize_data_by_frequency(data_by_freq_tr, config, "train"),
        "test": summarize_data_by_frequency(data_by_freq_te, config, "test"),
        "heldout": summarize_data_by_frequency(data_by_freq_heldout, config, "heldout") if data_by_freq_heldout is not None else None,
    }

    # Only make these diagnostic plots for phase shifter architecture
    plot_input_scatterplots(data_by_freq_tr, data_by_freq_te, config.freq_list, paths.images_path, config.main_arch, config.cartesian)
    # # Adjust features and to_drop if Cartesian mode is enabled
    # if config.cartesian:
    #     if config.main_arch == "phase_shifter":
    #         config.features = [
    #             "s1_s11_x", "s1_s11_y", "s1_s12_x", "s1_s12_y",
    #             "s2_s11_x", "s2_s11_y", "s2_s12_x", "s2_s12_y",
    #         ]
    #         config.to_drop = [
    #             "s1_s21_x", "s1_s21_y", "s1_s22_x", "s1_s22_y",
    #             "s2_s21_x", "s2_s21_y", "s2_s22_x", "s2_s22_y",
    #             "S22m", "s2",
    #         ]
    #     elif config.main_arch == "LNA":
    #         config.features = [
    #             'in_s11_x', 'in_s11_y', 'in_s12_x', 'in_s12_y',
    #             'out_s11_x', 'out_s11_y', 'out_s12_x', 'out_s12_y',
    #         ]
    #         config.to_drop = [
    #             "in_s21_x","in_s21_y","in_s22_x","in_s22_y",
    #             "out_s21_x","out_s21_y","out_s22_x","out_s22_y",
    #         ]

    # ---------------------
    # Preprocess per frequency
    # ---------------------
    if config.preprocess:
        for freq in config.freq_list:
            print(f"\n\n{'------'*10}\nPreprocessing for Frequency: {str(freq / 1e9)} GHz\n")
            img_sfx = f"{config.main_topology}_{freq/1e9}GHz{coordinate_string}"
            df_tr = data_by_freq_tr[freq]
            df_te = data_by_freq_te[freq]

            print(f"\nTrain Dataframe Info:\nShape: {df_tr.shape}\n")
            # print(f"{df_tr.info()}\n")
            # print(f"{df_tr.describe(include='all').T}\n")
            df_tr, features_used, prep_stats = preprocess_data_by_frequency(df_tr, config.features, config.specifications, config.targets, config.to_drop, config.main_arch, paths.images_path, img_sfx, config, True, fit_stats=None, return_stats=True, diagnostic_split="train")
            config.features = features_used
            print(f"\nTest Dataframe Info:\nShape: {df_te.shape}\n")
            # print(f"{df_te.info()}\n")
            # print(f"{df_te.describe(include='all').T}\n")
            df_te, _ = preprocess_data_by_frequency(df_te, config.features, config.specifications, config.targets, config.to_drop, config.main_arch, paths.images_path, img_sfx, config, True, fit_stats=prep_stats, return_stats=False, diagnostic_split="test")
            if data_by_freq_heldout is not None:
                df_ho = data_by_freq_heldout[freq]
                print(f"\nHeldout Dataframe Info:\nShape: {df_ho.shape}\n")
                df_ho, _ = preprocess_data_by_frequency(df_ho, config.features, config.specifications, config.targets, config.to_drop, config.main_arch, paths.images_path, img_sfx, config, True, fit_stats=prep_stats, return_stats=False, diagnostic_split="heldout")
                data_by_freq_heldout[freq] = df_ho
            # Analyzing and cleaning the data:
            data_by_freq_tr[freq] = df_tr
            data_by_freq_te[freq] = df_te

    preprocessed_dataset_summaries = {
        "train": summarize_data_by_frequency(data_by_freq_tr, config, "train"),
        "test": summarize_data_by_frequency(data_by_freq_te, config, "test"),
        "heldout": summarize_data_by_frequency(data_by_freq_heldout, config, "heldout") if data_by_freq_heldout is not None else None,
    }
    dataset_report_path = write_experiment_dataset_report(
        out_dir=paths.run_result_dir,
        experiment_id=experiment_id,
        config=config,
        paths=paths,
        heldout_paths=heldout_paths,
        loaded_summaries=loaded_dataset_summaries,
        preprocessed_summaries=preprocessed_dataset_summaries,
        dataset_read_time_sec=dataset_read_time_sec,
    )
    print(f"Dataset report: {dataset_report_path}")

    run_experiments(
        experiment_id=experiment_id,
        config=config,
        paths=paths,
        data_by_freq_tr=data_by_freq_tr,
        data_by_freq_te=data_by_freq_te,
        coord=coord,
        coordinate_string=coordinate_string,
        user=user,
        data_by_freq_heldout=data_by_freq_heldout,
    )
    # Analysis from logs
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
        heldout_csv_path = os.path.join(paths.run_result_dir, "heldout_results.csv")
        heldout_pred_csv_path = os.path.join(paths.run_result_dir, "all_heldout_predictions.csv")
        if os.path.exists(heldout_csv_path):
            plot_heldout_from_logs(
                normal_csv_path=csv_path,
                normal_pred_csv_path=pred_csv_path,
                heldout_csv_path=heldout_csv_path,
                heldout_pred_csv_path=heldout_pred_csv_path,
                out_dir=analysis_out,
                experiment_id=experiment_id,
                title="",
                MODEL_BASE_COLORS=MODEL_BASE_COLORS,
                plot_show=config.plot_show,
                util=1.0,
            )
        rebuttal_outputs = write_rebuttal_reports(
            paths.run_result_dir,
            config,
            model_colors=MODEL_BASE_COLORS,
            plot_show=config.plot_show,
        )
        print(f"Rebuttal report artifacts: {rebuttal_outputs['output_dir']}")
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
