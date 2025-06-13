"""
Evaluation metrics and reporting for RI-RADS classification.

This module provides comprehensive evaluation metrics, visualization,
and report generation for the RI-RADS scoring system.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve
)
from datetime import datetime


class EvaluationMetrics:
    """Handles evaluation metrics calculation and visualization."""

    def __init__(self, config, output_dir: Path):
        """
        Initialize evaluation metrics handler.

        Args:
            config: Training configuration
            output_dir: Directory for saving outputs
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.logger = logging.getLogger(__name__)

        # Set plot style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")

    def calculate_metrics(
            self,
            y_true: np.ndarray,
            y_pred: np.ndarray,
            y_probs: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Calculate comprehensive evaluation metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_probs: Prediction probabilities (optional)

        Returns:
            Dictionary of metrics
        """
        metrics = {}

        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['macro_f1'] = f1_score(y_true, y_pred, average='macro')
        metrics['weighted_f1'] = f1_score(y_true, y_pred, average='weighted')

        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None
        )

        for i, category in enumerate(self.config.rirads_categories):
            if i < len(precision):
                metrics[f'precision_class_{category}'] = precision[i]
                metrics[f'recall_class_{category}'] = recall[i]
                metrics[f'f1_class_{category}'] = f1[i]
                metrics[f'support_class_{category}'] = support[i]

        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()

        # AUC if probabilities available
        if y_probs is not None and len(np.unique(y_true)) == 2:
            metrics['auc'] = roc_auc_score(y_true, y_probs[:, 1])

        return metrics

    def plot_confusion_matrix(
            self,
            y_true: np.ndarray,
            y_pred: np.ndarray,
            save_path: Path,
            class_names: Optional[List[str]] = None
    ):
        """
        Plot and save confusion matrix.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Path to save the plot
            class_names: Names of classes
        """
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Create figure
        plt.figure(figsize=(10, 8))

        # Plot heatmap
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names or self.config.rirads_categories,
            yticklabels=class_names or self.config.rirads_categories,
            square=True,
            cbar_kws={'label': 'Count'}
        )

        plt.title('RI-RADS Classification Confusion Matrix', fontsize=16)
        plt.xlabel('Predicted Category', fontsize=12)
        plt.ylabel('True Category', fontsize=12)

        # Add percentage annotations
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                percentage = cm_normalized[i, j] * 100
                text_color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
                plt.text(
                    j + 0.5, i + 0.7,
                    f'{percentage:.1f}%',
                    ha='center', va='center',
                    color=text_color,
                    fontsize=8
                )

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Confusion matrix saved to {save_path}")

    def plot_roc_curves(
            self,
            y_true: np.ndarray,
            y_probs: np.ndarray,
            save_path: Path
    ):
        """
        Plot ROC curves for multi-class classification.

        Args:
            y_true: True labels
            y_probs: Prediction probabilities
            save_path: Path to save the plot
        """
        plt.figure(figsize=(10, 8))

        # Compute ROC curve for each class
        n_classes = y_probs.shape[1]

        for i in range(n_classes):
            # Create binary labels for this class
            y_true_binary = (y_true == i).astype(int)

            if len(np.unique(y_true_binary)) > 1:
                fpr, tpr, _ = roc_curve(y_true_binary, y_probs[:, i])
                auc = roc_auc_score(y_true_binary, y_probs[:, i])

                plt.plot(
                    fpr, tpr,
                    label=f'{self.config.category_names[i]} (AUC = {auc:.3f})',
                    linewidth=2
                )

        # Plot diagonal
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1)

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves for RI-RADS Categories', fontsize=16)
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"ROC curves saved to {save_path}")

    def plot_training_history(
            self,
            history: List[Dict],
            save_path: Path
    ):
        """
        Plot training history over epochs.

        Args:
            history: List of epoch results
            save_path: Path to save the plot
        """
        # Convert to DataFrame
        df = pd.DataFrame(history)

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Plot loss
        axes[0, 0].plot(df['epoch'], df['train_loss'], label='Train', marker='o')
        axes[0, 0].plot(df['epoch'], df['val_loss'], label='Validation', marker='s')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Plot accuracy
        axes[0, 1].plot(df['epoch'], df['train_accuracy'], label='Train', marker='o')
        axes[0, 1].plot(df['epoch'], df['val_accuracy'], label='Validation', marker='s')
        axes[0, 1].plot(df['epoch'], df['test_accuracy'], label='Test', marker='^')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Accuracy Over Time')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Plot per-class metrics if available
        if 'f1_class_1' in df.columns:
            for i, category in enumerate(self.config.rirads_categories):
                col_name = f'f1_class_{category}'
                if col_name in df.columns:
                    axes[1, 0].plot(df['epoch'], df[col_name],
                                    label=f'Class {category}', marker='o')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('F1 Score')
            axes[1, 0].set_title('Per-Class F1 Scores')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

        # Learning rate schedule (if available)
        if 'learning_rate' in df.columns:
            axes[1, 1].plot(df['epoch'], df['learning_rate'], marker='o')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_title('Learning Rate Schedule')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].set_yscale('log')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Training history plot saved to {save_path}")


class ReportGenerator:
    """Generates comprehensive evaluation reports."""

    def __init__(self, config, output_dir: Path):
        """
        Initialize report generator.

        Args:
            config: Training configuration
            output_dir: Directory for saving reports
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.logger = logging.getLogger(__name__)

    def save_classification_report(
            self,
            y_true: np.ndarray,
            y_pred: np.ndarray,
            save_path: Path,
            class_names: Optional[List[str]] = None
    ):
        """
        Save detailed classification report.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Path to save report
            class_names: Names of classes
        """
        # Generate classification report
        report = classification_report(
            y_true,
            y_pred,
            target_names=class_names or self.config.rirads_categories,
            output_dict=True
        )

        # Create formatted text report
        text_report = classification_report(
            y_true,
            y_pred,
            target_names=class_names or self.config.rirads_categories
        )

        # Save text report
        with open(save_path, 'w') as f:
            f.write("RI-RADS Classification Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(text_report)
            f.write("\n\nDetailed Metrics:\n")
            f.write("-" * 30 + "\n")

            # Add additional metrics
            accuracy = accuracy_score(y_true, y_pred)
            f.write(f"Overall Accuracy: {accuracy:.4f}\n")

            # Per-class accuracy
            cm = confusion_matrix(y_true, y_pred)
            per_class_acc = cm.diagonal() / cm.sum(axis=1)
            for i, (category, acc) in enumerate(zip(self.config.rirads_categories, per_class_acc)):
                f.write(f"Class {category} Accuracy: {acc:.4f}\n")

        # Save JSON report
        json_path = save_path.with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2)

        self.logger.info(f"Classification report saved to {save_path}")

    def generate_final_report(self, results_history: List[Dict]):
        """
        Generate comprehensive final training report.

        Args:
            results_history: List of epoch results
        """
        report_path = self.output_dir / "final_report.md"

        with open(report_path, 'w') as f:
            # Header
            f.write("# RI-RADS Classifier Training Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Configuration summary
            f.write("## Configuration\n\n")
            f.write(f"- **Model:** {self.config.model_name}\n")
            f.write(f"- **Batch Size:** {self.config.batch_size}\n")
            f.write(f"- **Learning Rate:** {self.config.learning_rate}\n")
            f.write(f"- **Epochs:** {self.config.num_epochs}\n")
            f.write(f"- **Max Length:** {self.config.max_length}\n")
            f.write(f"- **Sliding Window:** {'Enabled' if self.config.use_sliding_window else 'Disabled'}\n")
            if self.config.use_sliding_window:
                f.write(f"  - Window Stride: {self.config.window_stride_ratio * 100}%\n")
                f.write(f"  - Min Chunk Size: {self.config.min_chunk_size}\n")
            f.write("\n")

            # Training results
            f.write("## Training Results\n\n")

            # Best epoch
            best_epoch = max(results_history, key=lambda x: x['val_accuracy'])
            f.write(f"### Best Epoch: {best_epoch['epoch']}\n\n")
            f.write(f"- **Validation Accuracy:** {best_epoch['val_accuracy']:.4f}\n")
            f.write(f"- **Test Accuracy:** {best_epoch['test_accuracy']:.4f}\n")
            f.write(f"- **Validation Loss:** {best_epoch['val_loss']:.4f}\n\n")

            # Final epoch
            final_epoch = results_history[-1]
            f.write(f"### Final Epoch: {final_epoch['epoch']}\n\n")
            f.write(f"- **Validation Accuracy:** {final_epoch['val_accuracy']:.4f}\n")
            f.write(f"- **Test Accuracy:** {final_epoch['test_accuracy']:.4f}\n")
            f.write(f"- **Validation Loss:** {final_epoch['val_loss']:.4f}\n\n")

            # RI-RADS Categories
            f.write("## RI-RADS Categories\n\n")
            f.write("| Category | Description | Clinical Significance |\n")
            f.write("|----------|-------------|-----------------------|\n")
            for i, (cat, name) in enumerate(zip(self.config.rirads_categories,
                                                self.config.category_names)):
                f.write(f"| {cat} | {name} | ")
                if i == 0:
                    f.write("No findings |\n")
                elif i == 1:
                    f.write("Benign findings |\n")
                elif i == 2:
                    f.write("<2% malignancy risk |\n")
                elif i == 3:
                    f.write("2-95% malignancy risk |\n")
                elif i == 4:
                    f.write(">95% malignancy risk |\n")

            f.write("\n## Files Generated\n\n")
            f.write("- `best_model/`: Best performing model checkpoint\n")
            f.write("- `final_model/`: Final model after all epochs\n")
            f.write("- `training_history.png`: Training metrics visualization\n")
            f.write("- `confusion_matrix_*.png`: Confusion matrices for each epoch\n")
            f.write("- `classification_report_*.txt`: Detailed metrics for each epoch\n")
            f.write("- `config.json`: Complete configuration used for training\n")

        self.logger.info(f"Final report saved to {report_path}")

    def generate_prediction_report(
            self,
            texts: List[str],
            predictions: List[str],
            probabilities: Optional[np.ndarray] = None,
            save_path: Optional[Path] = None
    ) -> pd.DataFrame:
        """
        Generate report for predictions on new texts.

        Args:
            texts: Input texts
            predictions: Predicted categories
            probabilities: Prediction probabilities
            save_path: Optional path to save report

        Returns:
            DataFrame with predictions
        """
        # Create DataFrame
        df = pd.DataFrame({
            'text': texts,
            'predicted_category': predictions
        })

        # Add probabilities if available
        if probabilities is not None:
            for i, category in enumerate(self.config.rirads_categories):
                df[f'prob_class_{category}'] = probabilities[:, i]

            # Add confidence score (max probability)
            df['confidence'] = probabilities.max(axis=1)

        # Add category descriptions
        category_map = dict(zip(self.config.rirads_categories, self.config.category_names))
        df['category_description'] = df['predicted_category'].map(category_map)

        # Save if path provided
        if save_path:
            df.to_csv(save_path, index=False)
            self.logger.info(f"Prediction report saved to {save_path}")

            # Also save summary
            summary_path = save_path.with_suffix('.summary.txt')
            with open(summary_path, 'w') as f:
                f.write("RI-RADS Prediction Summary\n")
                f.write("=" * 40 + "\n\n")
                f.write(f"Total predictions: {len(df)}\n\n")
                f.write("Category distribution:\n")
                for category, count in df['predicted_category'].value_counts().sort_index().items():
                    percentage = (count / len(df)) * 100
                    f.write(f"  {category} ({category_map[category]}): {count} ({percentage:.1f}%)\n")

                if 'confidence' in df.columns:
                    f.write(f"\nAverage confidence: {df['confidence'].mean():.3f}\n")
                    f.write(f"Min confidence: {df['confidence'].min():.3f}\n")
                    f.write(f"Max confidence: {df['confidence'].max():.3f}\n")

        return df


class ModelExplainer:
    """Provides model interpretability features."""

    def __init__(self, model, tokenizer, config):
        """
        Initialize model explainer.

        Args:
            model: Trained model
            tokenizer: Tokenizer
            config: Configuration
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.logger = logging.getLogger(__name__)

    def get_attention_weights(self, text: str) -> np.ndarray:
        """
        Get attention weights for input text.

        Args:
            text: Input text

        Returns:
            Attention weights
        """
        # This is a placeholder - actual implementation would depend on model architecture
        self.logger.info("Attention visualization not implemented yet")
        return np.array([])

    def get_important_tokens(self, text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Get most important tokens for classification.

        Args:
            text: Input text
            top_k: Number of top tokens to return

        Returns:
            List of (token, importance) tuples
        """
        # This is a placeholder - could use integrated gradients or similar
        self.logger.info("Token importance not implemented yet")
        return []