# code/training/trainer.py
"""Training-time helpers for evaluation and model persistence."""

import os
import numpy as np
from utils.metrics import smape

from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

# Evaluate predictions with the metrics requested in the config.
def compute_test_metrics_dict(y_true, y_pred, metrics):
    """Compute the requested regression metrics for a test prediction batch."""
    out = {}
    for m in metrics:
        if m == "mae":
            out["mae"] = mean_absolute_error(y_true, y_pred)
        elif m == "mape":
            out["mape"] = mean_absolute_percentage_error(y_true, y_pred) * 100.0
        elif m == "rmse":
            out["rmse"] = np.sqrt(mean_squared_error(y_true, y_pred))
        elif m == "smape":
            out["smape"] = smape(y_true, y_pred)
        elif m == "nmae":
            # Normalize MAE by the target range for the current evaluation slice.
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


# Save a trained Keras model to disk.
def save_trained_model(model, path, model_name):
    """Save a trained Keras model under the configured weights directory."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    full_path = os.path.join(path, f"{model_name}.keras")
    model.save(full_path)
    return f"Model '{model_name}' saved to: {full_path}"
