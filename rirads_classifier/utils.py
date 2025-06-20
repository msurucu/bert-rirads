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

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """Setup comprehensive logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file to write logs to
    """
    # Configure format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup root logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


def suppress_tf_logging():
    """Suppress TensorFlow verbose logging."""
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


def set_random_seeds(seed: int = 42):
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)




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


def create_sample_data(data_dir: Path, num_samples: int = 100) -> None:
    """Create sample data for testing.
    
    Args:
        data_dir: Directory to create sample data in
        num_samples: Number of samples per split
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample radiology report templates
    templates = [
        "Normal chest X-ray with no acute findings. Heart size normal.",
        "Chest CT shows small nodule in right upper lobe, benign appearing.",
        "MRI brain shows no acute abnormalities. No masses or lesions identified.",
        "Suspicious mass in left breast measuring 2.1 cm with irregular margins.",
        "Large heterogeneous mass in right kidney with areas of necrosis, highly suspicious for malignancy."
    ]
    
    # Create data for each split
    for split in ['train', 'val', 'test']:
        samples = []
        for i in range(num_samples):
            label = str((i % 5) + 1)  # RI-RADS 1-5
            template_idx = int(label) - 1
            text = f"Case {i+1}: {templates[template_idx]} Patient ID: {1000+i}."
            
            samples.append({
                'sentence1': text,
                'label': label
            })
        
        # Save to JSON file
        output_file = data_dir / f"{split}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in samples:
                json.dump(sample, f, ensure_ascii=False)
                f.write('\n')
        
        print(f"Created {output_file} with {len(samples)} samples")


def validate_data_format(filepath: Path) -> bool:
    """Validate data file format.
    
    Args:
        filepath: Path to data file
        
    Returns:
        True if valid, False otherwise
    """
    try:
        filepath = Path(filepath)
        if not filepath.exists():
            print(f"  ✗ File not found: {filepath}")
            return False
            
        with open(filepath, 'r', encoding='utf-8') as f:
            line_count = 0
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        data = json.loads(line)
                        
                        # Check required fields
                        if 'sentence1' not in data:
                            print(f"  ✗ Line {line_num}: Missing 'sentence1' field")
                            return False
                            
                        if 'label' not in data:
                            print(f"  ✗ Line {line_num}: Missing 'label' field")
                            return False
                            
                        # Validate label range
                        try:
                            label = int(data['label'])
                            if label < 1 or label > 5:
                                print(f"  ✗ Line {line_num}: Invalid label {label} (must be 1-5)")
                                return False
                        except ValueError:
                            print(f"  ✗ Line {line_num}: Label must be numeric")
                            return False
                            
                        line_count += 1
                        
                    except json.JSONDecodeError as e:
                        print(f"  ✗ Line {line_num}: Invalid JSON - {e}")
                        return False
                        
        print(f"  ✓ Valid format with {line_count} samples")
        return True
        
    except Exception as e:
        print(f"  ✗ Error validating file: {e}")
        return False


def calculate_class_distribution(filepath: Path) -> Dict[str, int]:
    """Calculate class distribution in dataset.
    
    Args:
        filepath: Path to data file
        
    Returns:
        Dictionary mapping labels to counts
    """
    distribution = {}
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    label = str(data['label'])
                    distribution[label] = distribution.get(label, 0) + 1
                    
    except Exception as e:
        print(f"Error calculating distribution: {e}")
        
    return distribution


def save_config(config: Dict, filepath: Path) -> None:
    """Save configuration to JSON file.
    
    Args:
        config: Configuration dictionary
        filepath: Output file path
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert any Path objects to strings for JSON serialization
    config_copy = {}
    for key, value in config.items():
        if isinstance(value, Path):
            config_copy[key] = str(value)
        else:
            config_copy[key] = value
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(config_copy, f, indent=2, ensure_ascii=False)


def load_config(filepath: Path) -> Dict:
    """Load configuration from JSON file.
    
    Args:
        filepath: Config file path
        
    Returns:
        Configuration dictionary
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_gpu_info() -> Dict[str, Any]:
    """Get GPU information.
    
    Returns:
        Dictionary with GPU details
    """
    gpu_info = {
        'available': False,
        'count': 0,
        'devices': []
    }
    
    try:
        physical_devices = tf.config.list_physical_devices('GPU')
        gpu_info['available'] = len(physical_devices) > 0
        gpu_info['count'] = len(physical_devices)
        
        for i, device in enumerate(physical_devices):
            try:
                details = tf.config.experimental.get_device_details(device)
                gpu_info['devices'].append({
                    'id': i,
                    'name': device.name,
                    'details': details
                })
            except:
                gpu_info['devices'].append({
                    'id': i,
                    'name': device.name,
                    'details': 'Details unavailable'
                })
                
    except Exception as e:
        print(f"Error getting GPU info: {e}")
        
    return gpu_info


def format_time(seconds: float) -> str:
    """Format time duration in human-readable format.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage information.
    
    Returns:
        Dictionary with memory statistics
    """
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size
            'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
            'percent': process.memory_percent()
        }
    except ImportError:
        return {'error': 'psutil not available'}
    except Exception as e:
        return {'error': str(e)}


def ensure_dir(path: Path) -> Path:
    """Ensure directory exists, create if it doesn't.
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


