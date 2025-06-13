#!/usr/bin/env python
"""
RI-RADS Prediction Script

Script for making predictions using a trained RI-RADS classifier.
Supports both single text and batch prediction from files.
"""

import argparse
import sys
import json
from pathlib import Path
from typing import List, Optional
import pandas as pd

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from rirads_classifier import RIRADSClassifier
from rirads_classifier.config import InferenceConfig
from rirads_classifier.evaluation import ReportGenerator


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Make predictions using trained RI-RADS classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model path
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to trained model directory"
    )

    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)

    input_group.add_argument(
        "--text",
        type=str,
        help="Single text to classify"
    )

    input_group.add_argument(
        "--file",
        type=str,
        help="File containing texts to classify (JSON or TXT)"
    )

    input_group.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive mode - enter texts to classify"
    )

    # Output options
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for predictions (CSV format)"
    )

    parser.add_argument(
        "--show-probabilities",
        action="store_true",
        help="Show probability scores for each category"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for predictions"
    )

    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "gpu"],
        default="auto",
        help="Device to use for inference"
    )

    return parser.parse_args()


def load_texts_from_file(filepath: Path) -> List[str]:
    """
    Load texts from file.

    Args:
        filepath: Path to input file

    Returns:
        List of texts
    """
    filepath = Path(filepath)

    if filepath.suffix == '.json':
        # Load from JSON
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Handle different JSON formats
        if isinstance(data, list):
            if isinstance(data[0], dict):
                # List of dictionaries
                if 'sentence1' in data[0]:
                    texts = [item['sentence1'] for item in data]
                elif 'text' in data[0]:
                    texts = [item['text'] for item in data]
                else:
                    raise ValueError("JSON objects must have 'sentence1' or 'text' field")
            else:
                # List of strings
                texts = data
        else:
            raise ValueError("JSON file must contain a list")

    elif filepath.suffix == '.txt':
        # Load from text file (one text per line)
        with open(filepath, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]

    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")

    return texts


def print_prediction_result(
        text: str,
        prediction: str,
        probabilities: Optional[pd.Series] = None,
        category_names: Optional[List[str]] = None
):
    """
    Print formatted prediction result.

    Args:
        text: Input text
        prediction: Predicted category
        probabilities: Probability scores
        category_names: Category descriptions
    """
    print("\n" + "=" * 60)
    print("TEXT:")
    print("-" * 60)

    # Truncate long texts
    if len(text) > 200:
        print(text[:200] + "...")
    else:
        print(text)

    print("\n" + "-" * 60)
    print(f"PREDICTION: RI-RADS {prediction}")

    if category_names and prediction in category_names:
        print(f"CATEGORY: {category_names[prediction]}")

    if probabilities is not None:
        print("\nPROBABILITIES:")
        for category, prob in probabilities.items():
            bar_length = int(prob * 20)
            bar = "█" * bar_length + "░" * (20 - bar_length)
            print(f"  RI-RADS {category}: [{bar}] {prob:.3f}")


def interactive_mode(classifier, config, show_probabilities):
    """
    Run interactive prediction mode.

    Args:
        classifier: Trained classifier
        config: Inference configuration
        show_probabilities: Whether to show probabilities
    """
    print("\nInteractive RI-RADS Classification")
    print("Enter radiology report text (or 'quit' to exit):")
    print("=" * 60)

    # Load category names if available
    label_mappings_path = Path(config.model_path) / "label_mappings.json"
    category_names = None

    if label_mappings_path.exists():
        with open(label_mappings_path, 'r') as f:
            mappings = json.load(f)
            if 'categories' in mappings:
                category_names = {str(i + 1): name for i, name in enumerate(mappings['categories'])}

    while True:
        print("\n> ", end="", flush=True)
        text = input().strip()

        if text.lower() in ['quit', 'exit', 'q']:
            print("Exiting...")
            break

        if not text:
            print("Please enter some text.")
            continue

        # Make prediction
        if show_probabilities:
            predictions, probs = classifier.predict(text, return_probs=True)
            prediction = predictions[0]
            prob_dict = {str(i + 1): probs[0][i] for i in range(len(probs[0]))}
            prob_series = pd.Series(prob_dict)

            print_prediction_result(text, prediction, prob_series, category_names)
        else:
            prediction = classifier.predict(text)[0]
            print_prediction_result(text, prediction, None, category_names)


def main():
    """Main prediction function."""
    args = parse_arguments()

    # Load inference configuration
    config = InferenceConfig(
        model_path=args.model_path,
        batch_size=args.batch_size,
        device=args.device
    )

    # Load classifier
    print(f"Loading model from {args.model_path}...")
    try:
        classifier = RIRADSClassifier.load_for_inference(args.model_path, config)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # Interactive mode
    if args.interactive:
        interactive_mode(classifier, config, args.show_probabilities)
        return

    # Single text prediction
    if args.text:
        texts = [args.text]

    # File-based prediction
    elif args.file:
        print(f"Loading texts from {args.file}...")
        try:
            texts = load_texts_from_file(args.file)
            print(f"Loaded {len(texts)} texts")
        except Exception as e:
            print(f"Error loading file: {e}")
            sys.exit(1)

    # Make predictions
    print(f"\nMaking predictions...")

    if args.show_probabilities:
        predictions, probabilities = classifier.predict(texts, return_probs=True)
    else:
        predictions = classifier.predict(texts)
        probabilities = None

    # Display results
    if args.text:
        # Single text - detailed output
        if args.show_probabilities:
            prob_dict = {str(i + 1): probabilities[0][i] for i in range(len(probabilities[0]))}
            prob_series = pd.Series(prob_dict)
            print_prediction_result(texts[0], predictions[0], prob_series)
        else:
            print_prediction_result(texts[0], predictions[0])

    else:
        # Multiple texts - summary output
        print(f"\nPredictions complete!")

        # Create results DataFrame
        results_df = pd.DataFrame({
            'text': texts,
            'prediction': predictions
        })

        if probabilities is not None:
            # Add probability columns
            for i in range(probabilities.shape[1]):
                results_df[f'prob_rirads_{i + 1}'] = probabilities[:, i]
            results_df['confidence'] = probabilities.max(axis=1)

        # Print summary
        print("\nSummary:")
        print("-" * 40)
        for category, count in results_df['prediction'].value_counts().sort_index().items():
            percentage = (count / len(results_df)) * 100
            print(f"RI-RADS {category}: {count} ({percentage:.1f}%)")

        if 'confidence' in results_df.columns:
            print(f"\nAverage confidence: {results_df['confidence'].mean():.3f}")

        # Save results if requested
        if args.output:
            output_path = Path(args.output)
            results_df.to_csv(output_path, index=False)
            print(f"\nResults saved to {output_path}")

            # Also generate report if we have the components
            if 'categories' in classifier.__dict__:
                report_gen = ReportGenerator(classifier.config, output_path.parent)
                report_gen.generate_prediction_report(
                    texts,
                    predictions,
                    probabilities,
                    output_path.with_suffix('.report.csv')
                )


if __name__ == "__main__":
    main()