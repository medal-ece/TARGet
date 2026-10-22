# training/trainer.py

import json
import os
import time
import numpy as np
from utils.metrics import smape

from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

# -------------------------------------------------------------------------
# Compute regression metrics
# -------------------------------------------------------------------------
def compute_test_metrics_dict(y_true, y_pred, metrics):
    """
    Compute test metrics defined in config.metrics.
    Returns: dict(metric_name -> value)
    """
    out = {}
    for m in metrics:
        if m == "mae":
            out["mae"] = mean_absolute_error(y_true, y_pred)
        elif m == "mape":
            out["mape"] = mean_absolute_percentage_error(y_true, y_pred) * 100.0
        elif m == "rmse":
            out["rmse"] = np.sqrt(mean_squared_error(y_true, y_pred))
            # out["rmse"] = mean_squared_error(y_true, y_pred, squared=False)
        elif m == "smape":
            out["smape"] = smape(y_true, y_pred)
        elif m == "nmae":
            # Normalized MAE: divide MAE by (max - min) of the true targets
            y_min = float(np.min(y_true))
            y_max = float(np.max(y_true))
            denom = y_max - y_min
            if denom <= 0:
                out["nmae"] = 0.0
            else:
                mae = out.get("mae", mean_absolute_error(y_true, y_pred))
                out["nmae"] = mae / denom
        else:
            raise ValueError(f"Unsupported test metric: {m}")
    return out


# -------------------------------------------------------------------------
# Compute evaluation metrics (IDENTICAL TO YOUR ORIGINAL BEHAVIOR)
# -------------------------------------------------------------------------
def evaluate_model(model, x_test, y_test, config):
    y_pred = model.predict(x_test)
    # mae, mape, rmse = compute_test_metrics_dict(y_test, y_pred)
    metrics_out = compute_test_metrics_dict(y_test, y_pred, config.metrics)
    return metrics_out, y_pred


# -------------------------------------------------------------------------
# Save Keras model to disk
# -------------------------------------------------------------------------
def save_trained_model(model, path, model_name, metadata=None):
    os.makedirs(path, exist_ok=True)
    full_path = os.path.join(path, f"{model_name}.keras")
    model.save(full_path)
    if metadata is not None:
        meta_path = os.path.join(path, f"{model_name}.meta.json")
        metadata_out = {
            "schema_version": "model_weight_meta_v1",
            "saved_at_unix": time.time(),
            "model_file": os.path.basename(full_path),
            **metadata,
        }
        with open(meta_path, "w", encoding="utf-8") as handle:
            json.dump(metadata_out, handle, indent=2)
    return f"Model '{model_name}' saved to: {full_path}"
