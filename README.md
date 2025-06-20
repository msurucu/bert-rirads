# RI-RADS Text Classification Tool

A deep learning-based text classification system for RI-RADS (Radiology Imaging Reporting and Data System) scoring using transformer models. This tool leverages BERT multilingual models to classify radiology reports into appropriate RI-RADS categories.

## 🎯 Purpose

This tool automates the classification of radiology reports into RI-RADS categories (1-5), helping radiologists standardize their reporting and improve diagnostic consistency. The system uses state-of-the-art transformer models to understand medical text and assign appropriate scores.

## 📋 Features

- **Multi-class Classification**: Supports 5 RI-RADS categories
- **Multilingual Support**: Uses BERT multilingual models for international compatibility
- **Long Text Handling**: Implements sliding window approach for lengthy radiology reports
- **Class Imbalance Handling**: Custom class weights for imbalanced medical datasets
- **Comprehensive Evaluation**: Confusion matrices, classification reports, and per-class metrics
- **External Dataset Validation**: Supports evaluation on separate test sets

## 🏗️ Architecture

```
├── rirads_classifier/
│   ├── __init__.py
│   ├── config.py          # Configuration management
│   ├── data_loader.py     # Data loading and preprocessing
│   ├── model.py           # Model definition and training
│   ├── evaluation.py      # Evaluation metrics and visualization
│   └── utils.py           # Utility functions
├── data/
│   ├── train.json         # Training dataset
│   ├── val.json           # Validation dataset
│   └── test.json          # Test dataset
├── outputs/               # Model outputs and results
├── notebooks/             # Jupyter notebooks for analysis
├── requirements.txt       # Python dependencies
├── setup.py              # Package setup
└── README.md             # This file
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/rirads-scoring-tool.git
cd rirads-scoring-tool

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from rirads_classifier import RIRADSClassifier, TrainingConfig

# Initialize configuration
config = TrainingConfig(
    model_name="google-bert/bert-base-multilingual-cased",
    batch_size=8,
    num_epochs=15,
    learning_rate=5e-6
)

# Create and train classifier
classifier = RIRADSClassifier(config)
classifier.train()

# Make predictions
prediction = classifier.predict("Your radiology report text here...")
print(f"RI-RADS Score: {prediction}")
```

### Data Format

The tool expects JSON files with the following structure:

```json
{
  "sentence1": "Radiology report text...",
  "label": "1"  // RI-RADS category (1-5)
}
```

## 📊 RI-RADS Categories

| Category | Description | Clinical Significance |
|----------|-------------|----------------------|
| RI-RADS 1 | Negative | No findings |
| RI-RADS 2 | Benign | Benign findings |
| RI-RADS 3 | Probably Benign | <2% malignancy risk |
| RI-RADS 4 | Suspicious | 2-95% malignancy risk |
| RI-RADS 5 | Highly Suspicious | >95% malignancy risk |

## ⚙️ Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | bert-base-multilingual-cased | Transformer model to use |
| `batch_size` | 8 | Training batch size |
| `num_epochs` | 15 | Number of training epochs |
| `learning_rate` | 5e-6 | Learning rate for optimizer |
| `max_length` | 512 | Maximum sequence length |
| `use_sliding_window` | True | Enable sliding window for long texts |
| `window_stride_ratio` | 0.5 | Overlap ratio for sliding window |

## 📈 Performance Metrics

The tool provides comprehensive evaluation metrics:

- **Accuracy**: Overall and per-class accuracy
- **Confusion Matrix**: Visual representation of predictions
- **Classification Report**: Precision, recall, F1-score for each category
- **External Validation**: Performance on unseen datasets

## 🔧 Advanced Features

### Sliding Window for Long Reports

The tool automatically handles long radiology reports using a sliding window approach:

```python
config = TrainingConfig(
    use_sliding_window=True,
    window_stride_ratio=0.5,  # 50% overlap
    min_chunk_size=50         # Minimum tokens per chunk
)
```

### Class Weight Adjustment

Handle imbalanced datasets with custom class weights:

```python
config = TrainingConfig(
    class_importance=[3000, 3000, 10, 2, 1]  # Weights for RI-RADS 1-5
)
```

## 📝 Training Your Own Model

1. Prepare your dataset in the required JSON format
2. Place files in the `data/` directory
3. Configure training parameters
4. Run the training script:

```bash
python train_rirads.py --config config.yaml
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Format code
black rirads_classifier/

# Lint code
flake8 rirads_classifier/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Thanks to the Hugging Face team for the transformers library
- Medical professionals who provided domain expertise
- Contributors to the open-source radiology community

## 📚 References

1. ACR BI-RADS® Atlas, 5th Edition
2. [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
3. [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers)

## 📧 Contact

For questions or support, please open an issue on GitHub or contact rasiterenbuyuktoka@hotmail.com, islerya@gmail.com, msurucu@gmail.com

---

**Note**: This tool is for research purposes and should not replace professional medical judgment. Always consult with qualified radiologists for clinical decisions.