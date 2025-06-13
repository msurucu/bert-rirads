#!/usr/bin/env python
"""
RI-RADS Classifier - Example Usage

This script demonstrates how to use the RI-RADS classifier for
training and prediction tasks.
"""

import sys
from pathlib import Path

# Add parent directory to path if running as script
sys.path.append(str(Path(__file__).parent))

from rirads_classifier import (
    RIRADSClassifier,
    TrainingConfig,
    InferenceConfig,
    create_sample_data,
    validate_data_format
)


def example_1_quick_start():
    """Example 1: Quick start with sample data."""
    print("=" * 60)
    print("Example 1: Quick Start with Sample Data")
    print("=" * 60)

    # Create sample data
    data_dir = Path("data/sample")
    print(f"\n1. Creating sample data in {data_dir}...")
    create_sample_data(data_dir, num_samples=50)

    # Validate data format
    print("\n2. Validating data format...")
    validate_data_format(data_dir / "train.json")

    # Configure training
    print("\n3. Setting up configuration...")
    config = TrainingConfig(
        data_dir=str(data_dir),
        output_dir="outputs/example1",
        num_epochs=3,  # Quick training for demo
        batch_size=4,
        learning_rate=5e-6
    )
    print(config)

    # Initialize and train
    print("\n4. Training model...")
    classifier = RIRADSClassifier(config)
    classifier.train()

    # Make predictions
    print("\n5. Making predictions...")
    test_texts = [
        "No suspicious findings observed. Normal breast tissue.",
        "Irregular mass with spiculated margins detected. Highly suspicious.",
        "Small cyst noted, appears benign with smooth borders."
    ]

    predictions = classifier.predict(test_texts)

    print("\nResults:")
    for text, pred in zip(test_texts, predictions):
        print(f"Text: {text[:50]}...")
        print(f"Prediction: RI-RADS {pred}\n")


def example_2_custom_configuration():
    """Example 2: Custom configuration with medical BERT."""
    print("\n" + "=" * 60)
    print("Example 2: Custom Configuration")
    print("=" * 60)

    # Create custom configuration
    config = TrainingConfig(
        model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
        data_dir="data/",
        output_dir="outputs/example2",
        num_epochs=10,
        batch_size=8,
        learning_rate=3e-6,
        warmup_ratio=0.1,
        use_sliding_window=True,
        window_stride_ratio=0.5,
        class_weights=[3.0, 2.5, 1.5, 1.0, 1.0]  # Custom weights
    )

    # Save configuration
    config_path = Path("configs/medical_bert_config.json")
    config_path.parent.mkdir(exist_ok=True)
    config.save(config_path)
    print(f"\nConfiguration saved to {config_path}")

    # Load and display
    loaded_config = TrainingConfig.load(config_path)
    print(f"\nLoaded configuration:")
    print(loaded_config)


def example_3_inference():
    """Example 3: Load trained model for inference."""
    print("\n" + "=" * 60)
    print("Example 3: Inference with Trained Model")
    print("=" * 60)

    # Assuming we have a trained model
    model_path = "outputs/best_model"

    if not Path(model_path).exists():
        print(f"\nModel not found at {model_path}")
        print("Please train a model first using Example 1")
        return

    # Load model for inference
    print(f"\nLoading model from {model_path}...")
    config = InferenceConfig(model_path=model_path)
    classifier = RIRADSClassifier.load_for_inference(model_path, config)

    # Example radiology reports
    reports = [
        """
        Bilateral mammography performed. The breast tissue is heterogeneously dense,
        which may obscure small masses. No dominant mass, suspicious calcifications,
        or architectural distortion identified. Skin and nipples appear normal.
        IMPRESSION: Negative mammogram. BI-RADS Category 1.
        """,

        """
        Right breast ultrasound shows an oval, circumscribed, hypoechoic mass
        measuring 8 x 5 mm at 2 o'clock position, 3 cm from the nipple. 
        The mass demonstrates posterior acoustic enhancement consistent with
        a simple cyst. No internal vascularity on Doppler examination.
        IMPRESSION: Simple cyst right breast. BI-RADS Category 2 - Benign finding.
        """,

        """
        Left breast MRI reveals a 12 mm enhancing mass at 10 o'clock position.
        The mass shows smooth margins and homogeneous enhancement. Kinetic curve
        analysis shows persistent enhancement pattern. No additional suspicious
        enhancement elsewhere. 
        IMPRESSION: Probably benign finding. Recommend 6-month follow-up. BI-RADS Category 3.
        """,

        """
        Targeted ultrasound of palpable right breast lump reveals an irregular,
        hypoechoic mass measuring 18 x 14 mm with indistinct margins and
        posterior acoustic shadowing. The mass shows increased vascularity on
        color Doppler. Associated skin thickening noted.
        IMPRESSION: Suspicious finding. Tissue diagnosis recommended. BI-RADS Category 4C.
        """,

        """
        Diagnostic mammography demonstrates a 3.5 cm spiculated mass in the
        upper outer quadrant of the left breast with associated pleomorphic
        calcifications. Multiple enlarged axillary lymph nodes identified,
        the largest measuring 2.2 cm with loss of fatty hilum.
        IMPRESSION: Highly suggestive of malignancy. BI-RADS Category 5.
        """
    ]

    # Get predictions with probabilities
    print("\nClassifying radiology reports...")
    predictions, probabilities = classifier.predict(reports, return_probs=True)

    # Display results
    print("\nClassification Results:")
    print("-" * 60)

    for i, (report, pred, probs) in enumerate(zip(reports, predictions, probabilities)):
        print(f"\nReport {i + 1}:")
        print(f"Text preview: {report.strip()[:100]}...")
        print(f"Predicted RI-RADS: {pred}")
        print("Confidence scores:")
        for j, prob in enumerate(probs):
            print(f"  RI-RADS {j + 1}: {prob:.3f}")
        print()


def example_4_batch_processing():
    """Example 4: Batch processing of reports."""
    print("\n" + "=" * 60)
    print("Example 4: Batch Processing")
    print("=" * 60)

    # Create a batch of reports
    import json

    batch_file = Path("data/batch_reports.json")
    batch_file.parent.mkdir(exist_ok=True)

    # Sample batch data
    batch_data = [
        {"id": "001", "text": "No abnormalities detected. Normal mammogram."},
        {"id": "002", "text": "Benign-appearing calcifications in both breasts."},
        {"id": "003", "text": "Small nodule with smooth borders, likely fibroadenoma."},
        {"id": "004", "text": "Irregular mass with microcalcifications, suspicious."},
        {"id": "005", "text": "Spiculated mass with skin retraction, highly suspicious."},
    ]

    # Save batch file
    with open(batch_file, 'w') as f:
        json.dump(batch_data, f, indent=2)

    print(f"Created batch file: {batch_file}")

    # Process batch (simulation - would use predict_rirads.py in practice)
    print("\nTo process this batch, run:")
    print(f"python predict_rirads.py <model_path> --file {batch_file} --output results.csv")


def main():
    """Run all examples."""
    print("RI-RADS Classifier - Usage Examples")
    print("=" * 60)

    # Uncomment the examples you want to run

    # example_1_quick_start()  # Basic training example
    example_2_custom_configuration()  # Configuration examples
    # example_3_inference()  # Inference example
    example_4_batch_processing()  # Batch processing example

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()