"""
Configuration management for RI-RADS text classification.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import json
from pathlib import Path
import pandas as pd
import logging

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
        self.validate()
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        errors = []
        
        # Validate paths
        if not self.data_dir:
            errors.append("data_dir cannot be empty")
        
        if not self.output_dir:
            errors.append("output_dir cannot be empty")
        
        # Validate model parameters
        if self.batch_size <= 0:
            errors.append(f"batch_size must be positive, got {self.batch_size}")
        
        if self.num_epochs <= 0:
            errors.append(f"num_epochs must be positive, got {self.num_epochs}")
        
        if self.learning_rate <= 0:
            errors.append(f"learning_rate must be positive, got {self.learning_rate}")
        
        if self.max_length <= 0 or self.max_length > 8192:
            errors.append(f"max_length must be between 1 and 8192, got {self.max_length}")
        
        # Validate sliding window parameters
        if self.window_stride_ratio <= 0 or self.window_stride_ratio > 1:
            errors.append(f"window_stride_ratio must be between 0 and 1, got {self.window_stride_ratio}")
        
        if self.min_chunk_size <= 0:
            errors.append(f"min_chunk_size must be positive, got {self.min_chunk_size}")
        
        # Validate class importance
        if len(self.class_importance) != 5:
            errors.append(f"class_importance must have 5 values, got {len(self.class_importance)}")
        
        if any(w <= 0 for w in self.class_importance):
            errors.append("All class_importance values must be positive")
        
        # Validate optimizer parameters
        if not (0 <= self.adam_beta1 < 1):
            errors.append(f"adam_beta1 must be in [0, 1), got {self.adam_beta1}")
        
        if not (0 <= self.adam_beta2 < 1):
            errors.append(f"adam_beta2 must be in [0, 1), got {self.adam_beta2}")
        
        if self.adam_epsilon <= 0:
            errors.append(f"adam_epsilon must be positive, got {self.adam_epsilon}")
        
        if errors:
            raise ValueError(f"Configuration validation failed:\n" + "\n".join(f"- {error}" for error in errors))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """Create config from dictionary."""
        # Filter out unknown fields
        known_fields = {field.name for field in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in known_fields}
        return cls(**filtered_dict)
    
    def save(self, filepath: Path) -> None:
        """Save configuration to JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, filepath: Path) -> 'TrainingConfig':
        """Load configuration from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

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
        self.validate()
    
    def validate(self) -> None:
        """Validate inference configuration."""
        errors = []
        
        # Validate model path
        if not self.model_path:
            errors.append("model_path cannot be empty")
        elif not Path(self.model_path).exists():
            errors.append(f"Model not found at {self.model_path}")
        
        # Validate parameters
        if self.batch_size <= 0:
            errors.append(f"batch_size must be positive, got {self.batch_size}")
        
        if self.max_length <= 0 or self.max_length > 8192:
            errors.append(f"max_length must be between 1 and 8192, got {self.max_length}")
        
        if self.window_stride_ratio <= 0 or self.window_stride_ratio > 1:
            errors.append(f"window_stride_ratio must be between 0 and 1, got {self.window_stride_ratio}")
        
        if self.device not in ["auto", "cpu", "gpu"]:
            errors.append(f"device must be 'auto', 'cpu', or 'gpu', got {self.device}")
        
        if errors:
            raise ValueError(f"Inference configuration validation failed:\n" + "\n".join(f"- {error}" for error in errors))


RIRADS_CONFIGS = {
    "standard": {
        "num_epochs": 15,
        "batch_size": 8,
        "learning_rate": 5e-6,
        "warmup_steps": 100,
    },
    "fast": {
        "num_epochs": 5,
        "batch_size": 16,
        "learning_rate": 1e-5,
        "warmup_steps": 50,
    },
    "high_quality": {
        "num_epochs": 30,
        "batch_size": 4,
        "learning_rate": 2e-6,
        "warmup_steps": 200,
        "weight_decay": 0.01,
    },
    "medical_bert": {
        "model_name": "emilyalsentzer/Bio_ClinicalBERT",
        "num_epochs": 20,
        "batch_size": 6,
        "learning_rate": 3e-6,
        "warmup_steps": 150,
        "weight_decay": 0.01,
    },
    "debug": {
        "num_epochs": 2,
        "batch_size": 2,
        "learning_rate": 1e-4,
        "warmup_steps": 10,
    },
}


def get_config(name: str = "standard") -> TrainingConfig:
    """Get a predefined configuration by name."""
    if name not in RIRADS_CONFIGS:
        raise ValueError(f"Unknown config: {name}. Available: {list(RIRADS_CONFIGS.keys())}")
    
    # Create base config and update with preset values
    config = TrainingConfig()
    preset_values = RIRADS_CONFIGS[name]
    
    # Update config with preset values
    for key, value in preset_values.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            logging.warning(f"Unknown config parameter in preset '{name}': {key}")
    
    return config


def list_available_configs() -> Dict[str, Dict[str, Any]]:
    """List all available configuration presets."""
    return RIRADS_CONFIGS.copy()


def create_config_from_args(**kwargs) -> TrainingConfig:
    """Create configuration from keyword arguments."""
    return TrainingConfig(**kwargs)
