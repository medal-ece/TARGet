# code/utils/metrics.py
"""Small metric utilities shared by training and plotting helpers."""
import tensorflow as tf

def smape(y_true, y_pred, epsilon=1e-8):
    """Return symmetric MAPE as a percentage."""
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    epsilon = tf.cast(epsilon, y_pred.dtype)
    numerator = tf.abs(y_pred - y_true)
    denominator = tf.abs(y_true) + tf.abs(y_pred) + epsilon
    return float(tf.reduce_mean(2.0 * numerator / denominator) * 100.0)

ALIAS = {
    "mae": tf.keras.metrics.MeanAbsoluteError(),
    "mse": tf.keras.metrics.MeanSquaredError(),
    "rmse": tf.keras.metrics.RootMeanSquaredError(),
    "mape": tf.keras.metrics.MeanAbsolutePercentageError(),
    "logcosh": "log_cosh_error",
    # r2 intentionally not used in this project
    "cosine": "cosine_similarity",
    "mre": "mean_relative_error",
    "msle": "mean_squared_logarithmic_error",
    "smape": smape,
}
