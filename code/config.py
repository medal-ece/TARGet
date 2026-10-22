# code/config.py
"""Central experiment configuration."""

from dataclasses import dataclass, field

@dataclass
class Config:
    """Default settings for data preparation, training, and analysis."""

    # General run settings.
    cartesian: bool = True
    preprocess: bool = True
    train_model: bool = True
    test_model: bool = True
    save_model: bool = True
    write_to_csv: bool = True
    run_analysis_after_experiment = True
    plot_during_experiment: bool = True
    plot_show: bool = True
    debug: bool = True
    # Model families to train for each experiment.
    model_types: list = field(default_factory=lambda: ["FCN", "CCI", "GNN", "CCIGNN-Sep"])  # CCIGNN-Sep is TARGet
    # Utilization levels are expressed as fractions of the available training set.
    utilization: list = field(default_factory=lambda: [0.01, 0.03, 0.05, 0.2, 0.5, 1.0])
    # Repeat each configuration to capture variability across runs.
    repetition: int = 1
    freq_count: int = 40
    freq_mode: str = "uniform"  # "random", "uniform", or "manual"
    manual_freqs: list = field(default_factory=list)  # List of specific frequencies for manual mode
    freq_list: list = field(default_factory=lambda: [9800000000])

    workflow: str = "main"
    measure_inference_latency: bool = True
    inference_latency_warmup: int = 2
    inference_latency_repeats: int = 10
    inference_latency_batch_size: int = 512
    dataset_version: str = "v1"
    dataset_variant: str | None = None
    dataset_split: str = "main"
    heldout_dataset_split: str = "heldout"
    run_heldout_eval: bool = False
    lna_heldout_in_match_topologies: list = field(default_factory=lambda: ["inf_cap_ind"])
    lna_heldout_out_match_topologies: list = field(default_factory=lambda: ["inf_cap_ind"])
    lna_heldout_match_rule: str = "either_side"
    use_dataset_cache: bool = True
    rebuild_dataset_cache: bool = False

    # Shared model and optimizer settings.
    main_arch: str = "LNA" # LNA, phase_shifter
    TOPOLOGY_COL = "topology_index"
    TOPOLOGY_PREFIX = "topology_"
    # Filled later by main.py after topology selection.
    main_topology: str = None
    batch_size: int = 512
    ep_count: int = 500
    val_ratio: float = 0.2
    early_stop_monitor: str = "val_loss"
    early_stop_patience: int = 40
    min_delta: float = 1e-5
    optimizer_name: str = "adam"
    learning_rate: float = 0.001
    scaler_name: str = "standard"
    loss_name: str = "mse"
    # Metrics tracked during training.
    metrics_names: list = field(default_factory=lambda: ["mae", "mape", "rmse", "nmae"])
    # Metrics expected in saved logs and downstream analysis.
    METRICS = ["mae", "mape", "rmse", "nmae"]

    # Data and feature settings.
    base_seed = 42
    features: list = field(default_factory=list)
    specifications: list = field(default_factory=list)
    # targets: list = field(default_factory=list)
    if main_arch == "phase_shifter":
        selected_topologies = [0, 1, 2, 3]  # [0], [1], [0, 3], [1, 2, 3], etc.
        targets: list = field(default_factory=lambda: ["S11m"]) # ["S11m", "S21m", "S21p", "S22m"]
    elif main_arch == "LNA":
        selected_topologies = [0, 1]        # [0], [1], [0, 1]
        targets: list = field(default_factory=lambda: ["GA"]) # ["GA", "GP", "NF"]
    to_drop: list = field(default_factory=list)

    # Adapter modes per model family.
    ADAPTER_MODES = {
        "FCN":                  "flat",       # same adapter family, no topo features
        "CCI":                  "semantic",
        "GNN":                  "gnn",
        "CCIGNN-Sep":           "semantic",   # hybrid CCI + GNN
    }

    # Plotting configuration.
    metrics_to_plot: list = field(default_factory=lambda: ["loss"])
    MODEL_BASE_COLORS = {
        "FCN":              "tab:cyan",             # cyan
        "CCI":              "#5556F0",              # custom blue (darker than default tab:blue for better contrast)
        "GNN":              "#2ca02c",              # custom green (similar to tab:green but slightly darker for better contrast)
        "CCIGNN-Sep":       "tab:red",              # red
    }

