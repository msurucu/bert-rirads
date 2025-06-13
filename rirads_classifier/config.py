"""
Configuration management for RI-RADS text classification.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import json
from pathlib import Path
import pandas as pd

@dataclass
class TrainingConfig:
    """Configuration class for training parameters."""

    # Model and data paths
    model_name: str = "google-bert/bert-base-multilingual-cased"
    data_dir: str = "datas/"
    output_dir: str = "outputs/"
    train_file: str = "train.json"
    val_file: str = "val.json"
    test_file: str = "test.json"
    ext_test_file: str = "ek_test_pred_250403.json"

    # Training parameters
    batch_size: int = 8
    num_epochs: int = 15
    learning_rate: float = 5e-6
    weight_decay: float = 0.0
    warmup_steps: int = 0
    max_length: int = 512
    seed: int = 42

    # Sliding window parameters
    use_sliding_window: bool = True
    min_chunk_size: int = 50
    window_stride_ratio: float = 0.5  # 50% overlap

    # Optimizer parameters
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0

    # Class weights for imbalanced data
    class_importance: List[int] = None

    def __post_init__(self):
        if self.class_importance is None:
            self.class_importance = [3000, 3000, 10, 2, 1]

@dataclass 
class InferenceConfig:
    """Configuration for inference/prediction."""

    model_path: str
    batch_size: int = 32
    max_length: int = 512
    use_sliding_window: bool = True
    window_stride_ratio: float = 0.5
    device: str = "auto"

    def __post_init__(self):
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")


RIRADS_CONFIGS = {
    "standard": TrainingConfig(
        num_epochs=15,
        batch_size=8,
        learning_rate=5e-6,
    ),
    "fast": TrainingConfig(
        num_epochs=5,
        batch_size=16,
        learning_rate=1e-5,
    ),
    "high_quality": TrainingConfig(
        num_epochs=30,
        batch_size=4,
        learning_rate=2e-6,
        warmup_ratio=0.2,
    ),
}


def get_config(name: str = "standard") -> TrainingConfig:
    """Get a predefined configuration by name."""
    if name not in RIRADS_CONFIGS:
        raise ValueError(f"Unknown config: {name}. Available: {list(RIRADS_CONFIGS.keys())}")
    return RIRADS_CONFIGS[name]
