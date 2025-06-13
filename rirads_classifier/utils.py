"""
Utility functions for RI-RADS text classification.
"""

import os
import sys
import json
import random
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np
import tensorflow as tf
import pandas as pd

class LoggingSetup:
    """Setup logging configuration to suppress verbose outputs."""

    @staticmethod
    def setup_logging():
        """Configure logging to suppress unnecessary outputs."""
        # TensorFlow logging
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
        os.environ["GRPC_VERBOSITY"] = "ERROR"
        os.environ["GLOG_minloglevel"] = "2"

        # Suppress Abseil logs
        try:
            import absl.logging
            absl.logging.get_absl_handler().python_handler.stream = open(os.devnull, 'w')
            absl.logging.set_verbosity(absl.logging.FATAL)
            absl.logging.set_stderrthreshold(absl.logging.FATAL)
        except ImportError:
            pass

        # TensorFlow specific logging
        tf.get_logger().setLevel('ERROR')
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

        # Setup basic logging for our script
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )




def unpack_x_y_sample_weight(data):
    """Unpack x, y, and sample_weight from data."""
    if isinstance(data, tuple):
        if len(data) == 1:
            return data[0], None, None
        elif len(data) == 2:
            return data[0], data[1], None
        elif len(data) == 3:
            return data[0], data[1], data[2]
    else:
        return data, None, None

# Apply the monkey-patch
if hasattr(tf.keras.utils, 'unpack_x_y_sample_weight'):
    # Already exists, no need to patch
    pass
else:
    # Add the missing function
    tf.keras.utils.unpack_x_y_sample_weight = unpack_x_y_sample_weight

# Also try to add it to keras directly if it's imported
try:
    import keras
    if not hasattr(keras.utils, 'unpack_x_y_sample_weight'):
        keras.utils.unpack_x_y_sample_weight = unpack_x_y_sample_weight
except ImportError:
    pass


