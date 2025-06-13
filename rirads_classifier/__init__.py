"""
RI-RADS Text Classification Package

A deep learning-based system for classifying radiology reports into
RI-RADS (Radiology Imaging Reporting and Data System) categories.
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Import main components
from .rirads_classifier import RIRADSClassifier
from .config import TrainingConfig, InferenceConfig, get_config
from .data_processing import DataProcessor, SlidingWindowProcessor
from .evaluation import EvaluationMetrics, ReportGenerator
from .utils import (
    setup_logging,
    suppress_tf_logging,
    set_random_seeds,
    create_sample_data,
    validate_data_format
)

# Define public API
__all__ = [
    "RIRADSClassifier",
    "TrainingConfig",
    "InferenceConfig", 
    "get_config",
    "DataProcessor",
    "SlidingWindowProcessor",
    "EvaluationMetrics",
    "ReportGenerator",
    "setup_logging",
    "suppress_tf_logging",
    "set_random_seeds",
    "create_sample_data",
    "validate_data_format",
]
