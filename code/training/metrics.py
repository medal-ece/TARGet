# code/training/metrics.py
"""Factories for losses, metrics, optimizers, and scalers used during training."""

from utils.metrics import smape

import tensorflow as tf
from keras import losses, metrics
from keras.optimizers import (SGD, RMSprop, Adam, Adadelta, Adagrad, Adamax, Nadam, Ftrl)
from sklearn.preprocessing import (StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, Normalizer, PowerTransformer, QuantileTransformer)

__all__ = [
    "get_regression_loss",
    "get_regression_metrics",
    "get_optimizer",
    "get_scaler",
    "smape",
    "smape_loss",
    "normalized_mae",
]

def normalized_mae(y_true, y_pred, eps=1e-8):
    """Compute MAE normalized by the target range within the current batch."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    abs_err = tf.abs(y_true - y_pred)
    mae = tf.reduce_mean(abs_err)
    y_min = tf.reduce_min(y_true)
    y_max = tf.reduce_max(y_true)
    denom = tf.maximum(y_max - y_min, eps)
    return mae / denom


def get_regression_loss(name: str, **kwargs):
    """Return a Keras regression loss instance from a project-level alias."""
    key = (name or "").strip().lower()
    if key in {"mse", "mean_squared_error"}:
        return losses.MeanSquaredError(**kwargs)
    elif key in {"mae", "mean_absolute_error", "l1"}:
        return losses.MeanAbsoluteError(**kwargs)
    elif key in {"mape", "mean_absolute_percentage_error", "percent"}:
        return losses.MeanAbsolutePercentageError(**kwargs)
    elif key in {"msle", "mean_squared_log_error", "rmsle"}:
        return losses.MeanSquaredLogarithmicError(**kwargs)
    elif key in {"logcosh", "log_cosh"}:
        return losses.LogCosh(**kwargs)
    elif key in {"huber", "smooth_l1"}:
        return losses.Huber(**kwargs)
    elif key in {"cosine", "cosine_similarity"}:
        # In loss form, cosine similarity is returned with a negative sign.
        return losses.CosineSimilarity(**kwargs)
    else:
        # Fall back to MSE if the alias is unknown.
        return losses.MeanSquaredError(**kwargs)
    

def get_regression_metrics(names, **common_kwargs):
    """Return one or more Keras metric objects from the configured aliases."""
    def _ctor(key):
        key = (key or "").strip().lower()
        if key in {"mae", "mean_absolute_error", "l1"}:
            return metrics.MeanAbsoluteError
        if key in {"mse", "mean_squared_error"}:
            return metrics.MeanSquaredError
        if key in {"rmse", "root_mean_squared_error"}:
            return metrics.RootMeanSquaredError
        if key in {"mape", "mean_absolute_percentage_error"}:
            return metrics.MeanAbsolutePercentageError
        if key in {"msle", "mean_squared_log_error", "rmsle"}:
            return metrics.MeanSquaredLogarithmicError
        if key in {"logcosh", "log_cosh_error"}:
            return metrics.LogCoshError
        if key in {"cosine", "cosine_similarity"}:
            return metrics.CosineSimilarity
        if key in {"mre", "mean_relative_error"}:
            if "normalizer" not in common_kwargs:
                raise ValueError("MeanRelativeError requires a 'normalizer' kwarg.")
            return metrics.MeanRelativeError
        if key in {"r2", "r2_score", "r²"}:
            raise ValueError("R2 metric is intentionally disabled in this project.")
        # Unknown aliases fall back to MSE to keep the config forgiving.
        return metrics.MeanSquaredError
    # Match the return type to the caller's input form.
    if isinstance(names, str):
        names = [names]
        single = True
    else:
        single = False

    out = []
    for name in names:
        key = (name or "").strip().lower()
        if key in {"nmae", "normalized_mae", "norm_mae"}:
            out.append(normalized_mae)
        else:
            out.append(_ctor(key)(**common_kwargs))
    return out[0] if single else out


def get_optimizer(name: str, **kwargs):
    """Return an optimizer instance from a short project-level name."""
    opt = (name or "").strip().lower()
    if opt == "sgd":
        # Plain SGD is kept here mainly for controlled comparisons.
        return SGD(**kwargs)
    elif opt == "rmsprop":
        # RMSprop can help when training is sensitive to noisy gradients.
        return RMSprop(**kwargs)
    elif opt == "adam":
        # Adam is the default optimizer used by this project.
        return Adam(**kwargs)
    elif opt == "adadelta":
        # Adadelta is available for experiments that prefer adaptive step sizes.
        return Adadelta(**kwargs)
    elif opt == "adagrad":
        # Adagrad is included for comparison runs with sparse-style updates.
        return Adagrad(**kwargs)
    elif opt == "adamax":
        # Adamax is the infinity-norm variant of Adam.
        return Adamax(**kwargs)
    elif opt == "nadam":
        # Nadam adds Nesterov-style momentum on top of Adam.
        return Nadam(**kwargs)
    elif opt == "ftrl":
        # FTRL is kept for experiments that need a sparse-friendly optimizer.
        return Ftrl(**kwargs)
    else:
        return Adam(**kwargs)


def get_scaler(name, **kwargs):
    """Return a feature scaler instance from the configured project alias."""
    name = name.lower()
    if name == "standard":
        # Standard scaling is the default preprocessing choice in this project.
        return StandardScaler(**kwargs)
    elif name == "minmax":
        # Min-max scaling keeps every feature in a bounded range.
        return MinMaxScaler(**kwargs)
    elif name == "maxabs":
        # MaxAbsScaler preserves sign while rescaling by the largest magnitude.
        return MaxAbsScaler(**kwargs)
    elif name == "robust":
        # Robust scaling uses median and IQR, which helps when outliers are strong.
        return RobustScaler(**kwargs)
    elif name == "normalizer_l1":
        # Normalize each sample to unit L1 norm.
        return Normalizer(norm="l1", **kwargs)
    elif name == "normalizer_l2":
        # Normalize each sample to unit L2 norm.
        return Normalizer(norm="l2", **kwargs)
    elif name == "normalizer_max":
        # Normalize each sample by its largest absolute value.
        return Normalizer(norm="max", **kwargs)
    elif name == "power_yeo":
        # Yeo-Johnson can help when the raw feature distribution is highly skewed.
        return PowerTransformer(method="yeo-johnson", **kwargs)
    elif name == "power_boxcox":
        # Box-Cox is available when all features are strictly positive.
        return PowerTransformer(method="box-cox", **kwargs)
    elif name == "quantile_normal":
        # Map features to a normal distribution through their empirical quantiles.
        return QuantileTransformer(output_distribution="normal", random_state=0, **kwargs)
    elif name == "quantile_uniform":
        # Map features to a uniform distribution through their empirical quantiles.
        return QuantileTransformer(output_distribution="uniform", random_state=0, **kwargs)
    else:
        return StandardScaler(**kwargs)


def smape_loss(y_true, y_pred):
    """Expose SMAPE with the same call signature as a Keras loss."""
    return smape(y_true, y_pred)
