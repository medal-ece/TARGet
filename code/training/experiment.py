# training/experiment.py

# Self imports
from models.factory import get_shape, model_selector, create_model
from training.state import RunState
from training.recorder import record_repetition
from training.metrics import get_scaler, get_optimizer, get_regression_loss, get_regression_metrics, normalized_mae
from training.trainer import save_trained_model, compute_test_metrics_dict
from utils.plotting import history_plot
from utils.seed import set_global_seed
from utils.file_io import append_text_to_file, save_preds_per_frequency, data_to_csv_saver
from data.adapters import get_data_adapter
from models.factory import topology_policy

# Standard imports
import os, time
import numpy as np
import pandas as pd
from quantiphy import Quantity
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from keras.models import load_model
from keras.utils import plot_model
from spektral.layers import GCNConv, GlobalAvgPool
import keras


def train_once(model_type, model_parameters, prefix_string, repeat_string, x_train, y_train, x_val, y_val, config, paths, saving_name):
    # Defining and preparing the model for compilation:
    # Metrics:
    mets = get_regression_metrics(config.metrics_names)
    # Optimizer:
    opt = get_optimizer(config.optimizer_name, learning_rate=config.learning_rate, clipnorm=1.0) # clipnorm tames occasional spikes
    # Loss Function:
    loss = get_regression_loss(config.loss_name)
    # Define the callbacks
    # reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor=early_stop_monitor, factor=0.5, patience=12, min_lr=1e-6, verbose=0)
    # ckpt = tf.keras.callbacks.ModelCheckpoint('best.ckpt', monitor='val_loss', mode='min', save_best_only=True, save_weights_only=True, verbose=0)
    early_stop = keras.callbacks.EarlyStopping(monitor=config.early_stop_monitor, patience=config.early_stop_patience, verbose=0, mode='min', min_delta=config.min_delta, restore_best_weights=True)
    # callbacks = [early_stop, reduce_lr, ckpt]
    callbacks = [early_stop]
    # Initialize the model
    model = create_model(model_type, model_parameters, config, f"{prefix_string}{repeat_string}")
    # Compile
    model.compile(optimizer=opt, loss=loss, metrics=mets)
    x_tr = x_train
    y_tr = y_train
    history = model.fit(x_tr, y_tr, validation_data=(x_val, y_val), shuffle=True, epochs=config.ep_count, batch_size=config.batch_size, verbose=0, callbacks=callbacks)

    # Plot training history
    history_plot(history, config.metrics_to_plot, paths.run_result_dir, saving_name)
    return model


def test_once(model, x_test, y_test, config):
    # Predict
    y_pred = model.predict(x_test, verbose=0)
    # Calculating metrics:
    metrics_out = compute_test_metrics_dict(y_test, y_pred, config.metrics_names)
    return metrics_out, y_pred

def train_phase(model_type, model_parameters, prefix_string, repeat_string, x_tr_k, y_tr_k, x_val, y_val, config, paths, saving_name, state: RunState):
    model_saved = False
    model_trained = train_once(model_type, model_parameters, prefix_string, repeat_string, x_tr_k, y_tr_k, x_val, y_val, config, paths, saving_name)
    if not state.model_printed:
        # Plot model architecture ("TB": vertical plot; "LR": horizontal plot)
        plot_model(model_trained, to_file=os.path.join(paths.run_result_dir, f"Model_{model_type}_{config.main_arch}.png"), show_shapes=True, show_dtype=True, show_layer_names=True, rankdir="TB", show_layer_activations=True)
        # Print the model summary
        stringlist = []
        model_trained.summary(print_fn=lambda x: stringlist.append(x))
        summary_str = "\n".join(stringlist)
        tqdm.write(f"{summary_str}\n\n")
        append_text_to_file(summary_str, paths.result_log_file)
        state.model_printed = True
    if config.save_model and (not os.path.exists(os.path.join(paths.weights_path, f"{saving_name}.keras"))):
        # If the model is not already saved, save it
        save_log = save_trained_model(model_trained, paths.weights_path, model_name=saving_name)
        tqdm.write(save_log)
        model_saved = True
    return model_trained, model_saved


def test_phase(x_test, y_test, config, paths, saving_name, model_trained, util, freq):
    if config.train_model:
        model_to_test = model_trained
    else:
        load_path = os.path.join(paths.weights_path, f"{saving_name}.keras")
        model_to_test = load_model(load_path)
    metrics_out, y_pred = test_once(model_to_test, x_test, y_test, config)
    metrics_string = ", ".join([f"{m.upper()}: {metrics_out[m]:.4f}" for m in config.metrics_names])
    tqdm.write(f"Test on {Quantity(f'{freq} Hz')} & {util*100} %: {metrics_string}\n")
    return metrics_out, y_pred


def build_model_names(model_type, freq, util, r, config, model_parameters, nodes_phrase, coordinate_string):
    repeat_string = f"_r{r}" if config.repetition > 1 else ""
    util_string = f"U{str(util).rjust(3, '_')}"
    target_concat_name = '_'.join(config.targets)
    prefix_string = f"{model_type}_{target_concat_name}_{str(freq / 1e9)}_{util_string}{coordinate_string}"
    model_name_base = f"{prefix_string}_main_model"
    def _io_inputs(mp):
        # prefer new schema
        if isinstance(mp, dict) and "io" in mp:
            io = mp["io"]
            if "inputs" in io:
                return io["inputs"]
            if "input_groups" in io:
                return io["input_groups"]
            if "cci_input_groups" in io:
                return io["cci_input_groups"]
        # fallback legacy
        return mp.get("input_params")

    def _io_outputs(mp):
        if isinstance(mp, dict) and "io" in mp and "outputs" in mp["io"]:
            return mp["io"]["outputs"]
        return mp.get("output_params")
    if model_type in ["CCIGNN", "CCIGNN-Con", "CCIGNN-Sep", "CCIGNN-SepCon"]:
        # Hybrid shapes: [cci groups] + graph tensors (X, A)
        io = model_parameters.get("io", {})
        cci_groups = io.get("cci_input_groups", [])   # should be the cci groups
        late_features = io.get("late_features", [])
        N = model_parameters["N"]
        F = model_parameters["F"]
        # Structured shape (for logging / debugging)
        input_shape = {
            "cci": [len(g) for g in cci_groups],
            "late": len(late_features),
            "X": (N, F),
            "A": (N, N),
        }
        late_part = f"_late-{input_shape['late']}" if input_shape["late"] > 0 else ""
        # Filesystem-safe string version (for filenames)
        input_shape_str = (f"cci-{'+'.join(map(str, input_shape['cci']))}{late_part}_X-{N}x{F}_A-{N}x{N}")
    else:
        inp = _io_inputs(model_parameters)
        input_shape = get_shape(inp)
        input_shape_str = str(input_shape)
    outp = _io_outputs(model_parameters)
    output_shape = get_shape(outp)
    saving_name = f"{model_name_base}_[I{input_shape_str}, O{output_shape}]_{nodes_phrase}"
    return prefix_string, repeat_string, saving_name, input_shape_str, output_shape


def record_prediction_samples(
    path,
    *,
    experiment_id,
    user,
    model_type,
    architecture,
    topology,
    coordinate,
    frequency,
    utilization,
    repetition,
    seed,
    y_true,
    y_pred,
):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}")
    rows = []
    ts = time.time()
    for sample_id, (yt, yp) in enumerate(zip(y_true, y_pred)):
        rows.append({
            "timestamp": ts,
            "experiment_id": experiment_id,
            "schema_version": "v1",
            "user": user,
            "model_type": model_type,
            "architecture": architecture,
            "topology": topology,
            "coordinate": coordinate,
            "frequency": frequency,
            "utilization": utilization,
            "repetition": repetition,
            "seed": seed,
            "sample_id": sample_id,
            "y_true": float(yt),
            "y_pred": float(yp),
        })
    df = pd.DataFrame(rows)
    write_header = not os.path.exists(path)
    df.to_csv(path, mode="a", header=write_header, index=False)


def run_utilization_block(experiment_id, freq, util, model_parameters, nodes_phrase, x_train, y_train, x_test, y_test, model_type, config, paths, pbar, train_time_weight, test_time_weight, coordinate_string, coord, user, yscaler, state: RunState):
    # tqdm.write(str(f"{np.shape(x_train)}, {np.shape(x_test)}"))
    for r in range(config.repetition):
        set_global_seed(config.base_seed + r)
        val_frac = config.val_ratio
        idx = np.arange(len(y_train))
        tr_pool_idx, val_idx = train_test_split(idx, test_size=val_frac, shuffle=True, random_state=config.base_seed + r,)
        if isinstance(x_train, list):
            x_train_pool = [arr[tr_pool_idx] for arr in x_train]
            x_val = [arr[val_idx] for arr in x_train]
        else:
            x_train_pool = x_train[tr_pool_idx]
            x_val = x_train[val_idx]
        y_train_pool = y_train[tr_pool_idx]
        y_val = y_train[val_idx]

        k = max(1, int(util * len(y_train_pool)))   # how many samples
        if model_type in ["CCI", "CCI-Top", "CCI-TopNet", "CCI-TopNetCon", "CCI-Sep", "CCI-SepTopNet"]:
            # It's similar to below, but for CCI-NN we have two circuit components
            # Handle arbitrary number of CCI inputs (base groups + optional topology group)
            x_tr_k = [arr[:k] for arr in x_train_pool]
        elif model_type in ["GNN",  "GNN-ATT", "CCIGNN", "CCIGNN-Con", "CCIGNN-Sep", "CCIGNN-SepCon"]:
            # It's similar to above, but for GNN we have two inputs: X and A
            # x_tr_k = [x_train[0][:k], x_train[1][:k]]
            x_tr_k = [arr[:k] for arr in x_train_pool]
        else:
            x_tr_k = x_train_pool[:k]
        y_tr_k = y_train_pool[:k]
        loop_start_time = time.time()
        prefix_string, repeat_string, saving_name, input_shape_str, output_shape = build_model_names(model_type, freq, util, r, config, model_parameters, nodes_phrase, coordinate_string)
        model_saved = False
        model_trained = None
        # Training Phase
        if config.train_model:
            model_trained, model_saved = train_phase(model_type, model_parameters, prefix_string, repeat_string, x_tr_k, y_tr_k, x_val, y_val, config, paths, saving_name, state)
        pbar.update(train_time_weight if config.train_model else 0)
        # Testing Phase:
        if config.test_model:
            metrics_out, y_pred = test_phase(x_test, y_test, config, paths, saving_name, model_trained, util, freq)
            raw_record = {
                # identity
                "timestamp": time.time(),
                "experiment_id": experiment_id,
                "schema_version": "v1",
                "user": user,
                "model_type": model_type,
                "architecture": config.main_arch,
                "topology": config.main_topology,
                "coordinate": coord, 
                # experiment axes
                "frequency": freq,
                "utilization": util,
                "repetition": r,
                "seed": config.base_seed + r,
                # model config
                "optimizer": config.optimizer_name,
                "learning_rate": config.learning_rate,
                "batch_size": config.batch_size,
                "scaler": config.scaler_name,
                "loss": config.loss_name,
                "adapter_mode": config.ADAPTER_MODES[model_type]

            }
            # Add computed metrics like "mae": metrics_out["mae"],
            raw_record.update(metrics_out)
            record_repetition(raw_record, path=os.path.join(paths.run_result_dir, "all_repetitions.csv"))
            y_true_to_save = y_test
            y_pred_to_save = y_pred

            if yscaler is not None:
                y_true_to_save = yscaler.inverse_transform(np.asarray(y_test).reshape(-1, 1)).reshape(-1)
                y_pred_to_save = yscaler.inverse_transform(np.asarray(y_pred).reshape(-1, 1)).reshape(-1)
            else:
                y_true_to_save = np.asarray(y_test).reshape(-1)
                y_pred_to_save = np.asarray(y_pred).reshape(-1)
            record_prediction_samples(
                path=os.path.join(paths.run_result_dir, "all_predictions.csv"),
                experiment_id=experiment_id,
                user=user,
                model_type=model_type,
                architecture=config.main_arch,
                topology=config.main_topology,
                coordinate=coord,
                frequency=freq,
                utilization=util,
                repetition=r,
                seed=config.base_seed + r,
                y_true=y_true_to_save,
                y_pred=y_pred_to_save,
            )
            if r == (config.repetition-1):
                if config.write_to_csv:
                    path = save_preds_per_frequency(results_dir=paths.run_result_dir, model_type=model_type, freq_label=str(freq / 1e9), y_pred=y_pred, yscaler=yscaler, spec_name=config.targets[0], overwrite=True)
                    tqdm.write(f"Prediction file saved: {path}")
            def _save_new_model():
                save_log = save_trained_model(model_trained, paths.weights_path, model_name=saving_name)
                tqdm.write(save_log)
            # Save the model after testing if not saved before:
            if config.save_model and config.train_model and (not model_saved):
                # if saved is True, it's saved in this loop, so no need to save again
                load_path = os.path.join(paths.weights_path, f"{saving_name}.keras")
                if os.path.exists(load_path):
                    try:
                        current_saved_model = load_model(load_path, custom_objects={"normalized_mae": normalized_mae, "GCNConv": GCNConv, "GlobalAvgPool": GlobalAvgPool})
                        y_pred_current_saved_model = current_saved_model.predict(x_test, verbose=0)
                        mae_saved = mean_absolute_error(y_test, y_pred_current_saved_model)
                        mae_new   = metrics_out["mae"]
                        if mae_saved < mae_new:
                            tqdm.write(f"Current saved model has better MAE ({mae_saved:.6f}) than the new one ({mae_new:.6f}). Not overwriting.")
                        else:
                            tqdm.write(f"New model has better MAE ({mae_new:.6f}) than the current saved one ({mae_saved:.6f}). Overwriting.")
                            _save_new_model()
                    except Exception as e:
                        tqdm.write(f"Warning: could not load existing model at {load_path} ({type(e).__name__}: {e}). Overwriting with new model.")
                        _save_new_model()
                else:
                    _save_new_model()


        pbar.update(test_time_weight if config.test_model else 0)
        loop_time = time.time()-loop_start_time
        if util == 1:
            state.loop_time_list.append(loop_time)
        tqdm.write(f"Time = {Quantity(f'{loop_time:.1f} s')}\n")
        if config.test_model:
            results_to_save = {
                'date': time.strftime('%Y-%m-%d'), 'time': time.strftime('%H:%M:%S'), 'user': user,
                'architecture': config.main_arch, 'topology': config.main_topology, 'coordinate': coord,
                'model_type': model_type, 'input_shape': input_shape_str, 'output_shape': output_shape, 'layers': nodes_phrase,
                'optimizer': config.optimizer_name, 'learning_rate': config.learning_rate, 'scaler': config.scaler_name, 'loss_function': config.loss_name, 'metrics': '_'.join(config.metrics_names),
                'frequency': freq, 'utilization': util, 'batch_size': config.batch_size, 'validation_ratio': config.val_ratio, 'loop_time_sec': round(loop_time, 2),
            }
            results_to_save.update(metrics_out)
            data_to_csv_saver(os.path.join(paths.run_result_dir, f"run_results.csv"), results_to_save)


def run_frequency_block(
        *,
        experiment_id,
        freq,
        model_type,
        model_parameters,
        nodes_phrase,
        scaler,
        data_by_freq_tr,
        data_by_freq_te,
        config,
        use_topology_features,
        paths,
        pbar,
        train_time_weight,
        test_time_weight,
        coordinate_string,
        coord,
        user,
        state: RunState
    ):
    # Data preparation via adapters (model-agnostic)
    adapter = get_data_adapter(model_type, config, use_topology_features=use_topology_features)
    # Make sure topology one-hot features do not leak into models that should not see them.
    if model_type.endswith("-noTop"):
        io = model_parameters.get("io", {})
        if "inputs" in io:
            effective_inputs = io["inputs"]
        elif "input_groups" in io:
            effective_inputs = [f for group in io["input_groups"] for f in group]
        elif "cci_input_groups" in io:
            effective_inputs = [f for group in io["cci_input_groups"] for f in group]
        else:
            effective_inputs = config.features  # temporary fallback
        leaked = [f for f in effective_inputs if f.startswith(config.TOPOLOGY_PREFIX)]
        if leaked:
            raise RuntimeError(f"[SANITY ERROR] {model_type} received topology features: {leaked}")


    adapter_out = adapter.prepare(df_train=data_by_freq_tr[freq], df_test=data_by_freq_te[freq], config=config, model_parameters=model_parameters, scaler=scaler)
    x_train = adapter_out.x_train
    y_train = adapter_out.y_train
    x_test  = adapter_out.x_test
    y_test  = adapter_out.y_test
    yscaler = adapter_out.yscaler
    
    if config.debug:
        io = model_parameters.get("io", {})
        if model_type in ("FCN", "FCN-Top", "FCN-SepTopNet"):
            assert isinstance(x_train, np.ndarray), f"[{model_type}] expected dense ndarray input"
        elif model_type in ("CCI", "CCI-Top", "CCI-TopNet", "CCI-TopNetCon", "CCI-Sep"):
            expected_groups = None
            expected_late = 0
            if "input_groups" in io:
                expected_groups = len(io["input_groups"])
                expected_topology = 1 if io.get("topology_features") else 0
                expected_late = 1 if io.get("late_features") else 0
            elif "input_params" in model_parameters:
                expected_groups = len(model_parameters["input_params"])
            assert isinstance(x_train, list), f"[{model_type}] expected list input"
            assert expected_groups is not None, f"[{model_type}] could not infer expected input groups"
            expected_total = expected_groups + expected_topology + expected_late
            assert len(x_train) == expected_total, (f"[{model_type}] got {len(x_train)} inputs, expected {expected_total} ({expected_groups} groups + {expected_topology} topology + {expected_late} late)")
        elif model_type == "CCI-SepTopNet":
            expected_groups = len(io.get("input_groups", []))
            expected_topology = 1 if io.get("topology_features") else 0
            expected_late = 1 if io.get("late_features") else 0
            assert isinstance(x_train, list), f"[{model_type}] expected list input"
            expected_total = expected_groups + expected_topology + expected_late
            assert len(x_train) == expected_total, (f"[{model_type}] got {len(x_train)} inputs, expected {expected_total} ({expected_groups} groups + {expected_topology} topology + {expected_late} late)")
        elif model_type in ("GNN", "GNN-ATT"):
            assert isinstance(x_train, list), f"[{model_type}] expected graph input list"
            assert len(x_train) == 2, f"[{model_type}] expected [X, A]"
        elif model_type in ["CCIGNN", "CCIGNN-Con", "CCIGNN-Sep", "CCIGNN-SepCon"]:
            expected_cci_groups = None
            expected_late = 0
            if "cci_input_groups" in io:
                expected_cci_groups = len(io["cci_input_groups"])
                expected_late = 1 if io.get("late_features") else 0
            elif "cci_input_groups" in model_parameters:
                expected_cci_groups = len(model_parameters["cci_input_groups"])
            assert isinstance(x_train, list), "[CCI-GNN] expected hybrid input list"
            assert expected_cci_groups is not None, "[CCI-GNN] could not infer expected CCI groups"
            expected_total = expected_cci_groups + expected_late + 2
            assert len(x_train) == expected_total, (f"[CCI-GNN] expected {expected_total} inputs ([{expected_cci_groups} CCI groups] + {expected_late} late + X + A), got {len(x_train)}")

    for util in config.utilization:
        run_utilization_block(
            experiment_id=experiment_id,
            freq=freq,
            util=util,
            model_parameters=model_parameters,
            nodes_phrase=nodes_phrase,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            model_type=model_type,
            config=config,
            paths=paths,
            pbar=pbar,
            train_time_weight=train_time_weight,
            test_time_weight=test_time_weight,
            coordinate_string=coordinate_string,
            coord=coord,
            user=user,
            yscaler=yscaler,
            state=state
        )

def run_experiments(experiment_id, config, paths, data_by_freq_tr, data_by_freq_te, coord, coordinate_string, user):
    total_time = time.time()
    for model_type in config.model_types:
        # Model-local feature policy
        orig_features = config.features.copy()
        policy = topology_policy(model_type)
        use_topology_features = policy["use_topology_features"]
        model_features = orig_features.copy()
        if not use_topology_features:
            # Strip topology one-hot features ONLY for this model
            model_features = [f for f in model_features if not f.startswith(config.TOPOLOGY_PREFIX)]
        # Build model AFTER topology decision
        model_parameters, nodes_phrase = model_selector(model_type, config, use_topology_features=use_topology_features, features=model_features, targets=config.targets)
        # scaler:
        scaler = get_scaler(config.scaler_name)
        # New training loop:
        program_starts = time.time()
        print(f"{model_type} model experiment started ...")
        # Final results variables:
        state = RunState()
        train_time_weight = 7
        test_time_weight = 2
        each_util_time_weight = config.repetition * ((train_time_weight * config.train_model) + (test_time_weight * config.test_model))
        total_progress = len(config.freq_list) * len(config.utilization) * each_util_time_weight
        with tqdm(total=total_progress, desc=f"{model_type} Progress", dynamic_ncols=True) as pbar:
            # Loop over the frequencies:
            for freq_i, freq in enumerate(config.freq_list):      # Reading the data from csv files and only the determined columns
                tqdm.write(f"\n\n{'------'*10}\nFreq #{freq_i+1}: {str(freq / 1e9)} GHz\n")
                run_frequency_block(
                    experiment_id=experiment_id,
                    freq=freq,
                    model_type=model_type,
                    model_parameters=model_parameters,
                    nodes_phrase=nodes_phrase,
                    scaler=scaler,
                    data_by_freq_tr=data_by_freq_tr,
                    data_by_freq_te=data_by_freq_te,
                    config=config,
                    use_topology_features=use_topology_features,
                    paths=paths,
                    pbar=pbar,
                    train_time_weight=train_time_weight,
                    test_time_weight=test_time_weight,
                    coordinate_string=coordinate_string,
                    coord=coord,
                    user=user,
                    state=state
                )
        model_time = time.time() - program_starts
        print(f"{model_type} model experiment completed in {Quantity(model_time)}.\n\n")

    total_time = time.time() - total_time
    print(f"All experiments completed in {Quantity(total_time)}.\n\n")
