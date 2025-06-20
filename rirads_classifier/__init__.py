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
from .llm_classifier import RIRADSLLMClassifier, get_recommended_llm_models, create_llm_config
from .config import TrainingConfig, InferenceConfig, get_config, list_available_configs
from .data_processing import DataProcessor, SlidingWindowProcessor
from .evaluation import EvaluationMetrics, ReportGenerator
from .monitoring import MetricsLogger, EarlyStoppingMonitor, AlertSystem
from .utils import (
    setup_logging,
    suppress_tf_logging,
    set_random_seeds,
    create_sample_data,
    validate_data_format,
    get_gpu_info,
    format_time,
    get_memory_usage
)

# Define public API
__all__ = [
    "RIRADSClassifier",
    "RIRADSLLMClassifier",
    "get_recommended_llm_models",
    "create_llm_config",
    "TrainingConfig",
    "InferenceConfig", 
    "get_config",
    "list_available_configs",
    "DataProcessor",
    "SlidingWindowProcessor",
    "EvaluationMetrics",
    "ReportGenerator",
    "MetricsLogger",
    "EarlyStoppingMonitor",
    "AlertSystem",
    "setup_logging",
    "suppress_tf_logging",
    "set_random_seeds",
    "create_sample_data",
    "validate_data_format",
    "get_gpu_info",
    "format_time",
    "get_memory_usage",
]
