# code/utils/seed.py
"""Reproducibility helpers."""

import os
import random
import numpy as np
import tensorflow as tf

def set_global_seed(seed: int):
    """Seed Python, NumPy, and TensorFlow random generators in one place."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
