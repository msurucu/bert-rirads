#!/usr/bin/env python
"""
RI-RADS Proje Kurulum Scripti

Bu script, mevcut projenizi yeni yapıya dönüştürür.
"""

import os
import shutil
from pathlib import Path
from datetime import datetime

# Dosya içerikleri
FILES = {
    "rirads_classifier/__init__.py": '''"""
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
''',

    ".gitignore": '''# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
.venv

# Data and outputs
data/
outputs/
*.json
*.csv
*.xlsx
!data/sample/*.json

# Model files
*.h5
*.pkl
*.pt
*.pth
saved_model/
checkpoints/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Jupyter
.ipynb_checkpoints/
*.ipynb

# Logs
*.log
logs/

# Testing
.pytest_cache/
.coverage
htmlcov/
''',

    "LICENSE": '''MIT License

Copyright (c) 2024 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
''',

    "CONTRIBUTING.md": '''# Contributing to RI-RADS Scoring Tool

We welcome contributions to the RI-RADS Scoring Tool! 

## How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Code Style

- Follow PEP 8
- Use type hints where appropriate
- Add docstrings to all functions and classes
- Run `black` for code formatting
- Run `flake8` for linting

## Testing

- Add tests for new features
- Ensure all tests pass before submitting PR
- Aim for >80% code coverage

## Reporting Issues

- Use GitHub Issues
- Provide detailed description
- Include steps to reproduce
- Add system information
'''
}


def create_directory_structure():
    """Proje dizin yapısını oluştur."""
    directories = [
        "rirads_classifier",
        "data",
        "data/sample",
        "outputs",
        "configs",
        "notebooks",
        "tests",
        "docs"
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")


def backup_original_file():
    """Orijinal dosyayı yedekle."""
    original_file = "llm_claude_006.py"
    if Path(original_file).exists():
        backup_name = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{original_file}"
        shutil.copy(original_file, backup_name)
        print(f"✓ Original file backed up as: {backup_name}")
        return True
    return False


def split_original_code():
    """Orijinal kodu modüllere ayır."""
    # Orijinal dosyayı oku
    original_file = "llm_claude_006.py"
    if not Path(original_file).exists():
        print(f"❌ Original file {original_file} not found!")
        return False

    with open(original_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Config modülünü oluştur - TrainingConfig sınıfını çıkar
    config_start = content.find("@dataclass\nclass TrainingConfig:")
    config_end = content.find("class LoggingSetup:")

    if config_start != -1 and config_end != -1:
        # Import kısmını ekle
        config_content = '''"""
Configuration management for RI-RADS text classification.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import json
from pathlib import Path
import pandas as pd

''' + content[config_start:config_end].strip()

        # Config modülüne ek sınıfları ekle
        config_content += '''

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
'''

        with open("rirads_classifier/config.py", 'w', encoding='utf-8') as f:
            f.write(config_content)
        print("✓ Created: rirads_classifier/config.py")

    # TextClassifier'ı RIRADSClassifier olarak ayır
    classifier_start = content.find("class TextClassifier:")
    if classifier_start != -1:
        # Import bölümünü hazırla
        classifier_imports = '''"""
RI-RADS Text Classification System

Main classifier module for training and evaluating RI-RADS scoring models.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import tensorflow as tf
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    TFAutoModelForSequenceClassification,
    create_optimizer,
    set_seed
)

from .config import TrainingConfig, InferenceConfig
from .data_processing import DataProcessor, SlidingWindowProcessor
from .evaluation import EvaluationMetrics, ReportGenerator
from .utils import setup_logging, suppress_tf_logging

'''

        # TextClassifier'ı RIRADSClassifier olarak değiştir
        classifier_content = content[classifier_start:].replace("TextClassifier", "RIRADSClassifier")

        # Ana sınıfı yaz
        with open("rirads_classifier/rirads_classifier.py", 'w', encoding='utf-8') as f:
            f.write(classifier_imports + classifier_content.split("\n\ndef main()")[0])
        print("✓ Created: rirads_classifier/rirads_classifier.py")

    # Utility fonksiyonlarını ayır
    utils_content = '''"""
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

'''

    # LoggingSetup ve diğer yardımcı fonksiyonları ekle
    logging_start = content.find("class LoggingSetup:")
    logging_end = content.find("class TextClassifier:")

    if logging_start != -1 and logging_end != -1:
        utils_content += content[logging_start:logging_end]

    # unpack_x_y_sample_weight fonksiyonunu ekle
    unpack_start = content.find("def unpack_x_y_sample_weight")
    unpack_end = content.find("@dataclass")

    if unpack_start != -1 and unpack_end != -1:
        utils_content += "\n\n" + content[unpack_start:unpack_end]

    with open("rirads_classifier/utils.py", 'w', encoding='utf-8') as f:
        f.write(utils_content)
    print("✓ Created: rirads_classifier/utils.py")

    return True


def create_additional_files():
    """Ek dosyaları oluştur."""
    for filename, content in FILES.items():
        filepath = Path(filename)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"✓ Created: {filename}")


def main():
    """Ana kurulum fonksiyonu."""
    print("RI-RADS Proje Kurulum Başlıyor...")
    print("=" * 50)

    # 1. Dizin yapısını oluştur
    print("\n1. Dizin yapısı oluşturuluyor...")
    create_directory_structure()

    # 2. Orijinal dosyayı yedekle
    print("\n2. Orijinal dosya yedekleniyor...")
    backup_original_file()

    # 3. Kodu modüllere ayır
    print("\n3. Kod modüllere ayrılıyor...")
    if not split_original_code():
        print("❌ Kod ayrıştırma başarısız!")
        return

    # 4. Ek dosyaları oluştur
    print("\n4. Ek dosyalar oluşturuluyor...")
    create_additional_files()

    # 5. Kurulum tamamlandı
    print("\n" + "=" * 50)
    print("✅ Proje kurulumu tamamlandı!")
    print("\nYapılması gerekenler:")
    print("1. README.md, requirements.txt ve diğer artifact'ları manuel olarak kaydedin")
    print("2. config.py dosyasındaki 'import pandas as pd' satırını kontrol edin")
    print("3. Modüller arası import'ları test edin")
    print("4. 'python train_rirads.py --create-sample-data' ile test verisi oluşturun")


if __name__ == "__main__":
    main()