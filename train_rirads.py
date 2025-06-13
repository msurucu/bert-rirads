#!/usr/bin/env python
"""
RI-RADS Classifier Training Script

Main script for training the RI-RADS text classification model.
Supports both command-line and programmatic usage.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from rirads_classifier import RIRADSClassifier
from rirads_classifier.config import TrainingConfig, get_config
from rirads_classifier.utils import create_sample_data, validate_data_format


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train RI-RADS text classification model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Configuration options
    parser.add_argument(
        "--config",
        type=str,
        choices=["standard", "fast", "high_quality", "medical_bert"],
        default="standard",
        help="Predefined configuration to use"
    )

    parser.add_argument(
        "--config-file",
        type=str,
        help="Path to custom configuration JSON file"
    )

    # Data options
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/",
        help="Directory containing training data"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/",
        help="Directory for saving outputs"
    )

    # Model options
    parser.add_argument(
        "--model-name",
        type=str,
        help="Transformer model to use (overrides config)"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size (overrides config)"
    )

    parser.add_argument(
        "--num-epochs",
        type=int,
        help="Number of epochs (overrides config)"
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Learning rate (overrides config)"
    )

    # Utility options
    parser.add_argument(
        "--create-sample-data",
        action="store_true",
        help="Create sample data for testing"
    )

    parser.add_argument(
        "--validate-data",
        action="store_true",
        help="Validate data format before training"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print configuration and exit without training"
    )

    return parser.parse_args()


def main():
    """Main training function."""
    # Parse arguments
    args = parse_arguments()

    # Create sample data if requested
    if args.create_sample_data:
        print("Creating sample data...")
        create_sample_data(Path(args.data_dir))
        print("Sample data created. You can now run training.")
        return

    # Load configuration
    if args.config_file:
        print(f"Loading configuration from {args.config_file}")
        config = TrainingConfig.load(args.config_file)
    else:
        print(f"Using {args.config} configuration")
        config = get_config(args.config)

    # Override configuration with command-line arguments
    if args.data_dir:
        config.data_dir = args.data_dir
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.model_name:
        config.model_name = args.model_name
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.num_epochs:
        config.num_epochs = args.num_epochs
    if args.learning_rate:
        config.learning_rate = args.learning_rate

    # Validate data if requested
    if args.validate_data:
        print("\nValidating data files...")
        data_dir = Path(config.data_dir)

        for split in ['train', 'val', 'test']:
            filepath = data_dir / f"{split}.json"
            print(f"\n{split}.json:")
            if filepath.exists():
                validate_data_format(filepath)
            else:
                print(f"  ✗ File not found: {filepath}")

        if not args.dry_run:
            response = input("\nContinue with training? (y/n): ")
            if response.lower() != 'y':
                print("Training cancelled.")
                return

    # Print configuration
    print("\nConfiguration:")
    print("-" * 40)
    print(config)

    # Dry run - exit without training
    if args.dry_run:
        print("\nDry run complete. Exiting without training.")
        return

    # Validate configuration
    try:
        config.validate()
    except Exception as e:
        print(f"\nConfiguration error: {e}")
        sys.exit(1)

    # Initialize classifier
    print("\nInitializing RI-RADS classifier...")
    classifier = RIRADSClassifier(config)

    # Train model
    print("\nStarting training...")
    try:
        classifier.train()
        print("\nTraining completed successfully!")
        print(f"Results saved to: {classifier.output_dir}")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()