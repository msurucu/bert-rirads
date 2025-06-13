"""
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

class RIRADSClassifier:
    """Main class for text classification training and evaluation."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.datasets = None
        self.tf_datasets = {}
        self.label2id = {}
        self.id2label = {}
        self.output_dir = None
        self.logger = logging.getLogger(__name__)

        # Setup logging and random seeds
        LoggingSetup.setup_logging()
        self._set_random_seeds()
        self._setup_output_directory()

    def _set_random_seeds(self):
        """Set random seeds for reproducibility."""
        set_seed(self.config.seed)
        tf.random.set_seed(self.config.seed)
        np.random.seed(self.config.seed)
        random.seed(self.config.seed)

    def _setup_output_directory(self):
        """Create unique output directory."""
        output_base = Path(self.config.output_dir)
        output_base.mkdir(parents=True, exist_ok=True)

        model_name = self.config.model_name.split("/")[1]
        existing_dirs = glob.glob(f"{self.config.output_dir}output_{model_name}*")
        count = len(existing_dirs)

        while True:
            new_dir = f"{self.config.output_dir}output_{model_name}_{count}"
            if not os.path.exists(new_dir):
                break
            count += 1

        self.output_dir = Path(new_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Output directory: {self.output_dir}")

    def load_data(self):
        """Load and preprocess datasets."""
        self.logger.info("Loading datasets...")

        data_files = {
            'train': self.config.data_dir + self.config.train_file,
            'validation': self.config.data_dir + self.config.val_file,
            'test': self.config.data_dir + self.config.test_file,
        }

        try:
            self.datasets = load_dataset('json', data_files=data_files,)
            self.logger.info(f"Loaded datasets: {list(self.datasets.keys())}")
        except Exception as e:
            self.logger.error(f"Failed to load datasets: {e}")
            raise

    def setup_tokenizer_and_labels(self):
        """Initialize tokenizer and label mappings."""
        self.logger.info(f"Loading tokenizer: {self.config.model_name}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                batch_size=self.config.batch_size,
            )
        except Exception as e:
            self.logger.error(f"Failed to load tokenizer: {e}")
            raise

        # Setup label mappings
        label_list = self.datasets["train"].unique("label")
        label_list.sort()
        self.label2id = {label: i for i, label in enumerate(label_list)}
        self.id2label = {i: label for label, i in self.label2id.items()}

        self.logger.info(f"Label mappings: {self.id2label}")

    def tokenize_datasets(self):
        """Tokenize all datasets."""
        self.logger.info("Tokenizing datasets...")

        def tokenize_function(examples):
            # Handle both single and batch inputs
            if isinstance(examples['sentence1'], str):
                sentences = [examples['sentence1']]
                labels = [examples['label']]
            else:
                sentences = examples['sentence1']
                labels = examples['label']

            all_input_ids = []
            all_attention_masks = []
            all_labels = []
            all_sentences = []

            for sentence, label in zip(sentences, labels):
                # First, tokenize without truncation to check length
                tokens = self.tokenizer(
                    sentence,
                    truncation=False,
                    padding=False,
                    return_tensors=None
                )

                input_ids = tokens['input_ids']
                attention_mask = tokens['attention_mask']

                # If within limit, use as is
                if len(input_ids) <= self.config.max_length:
                    all_input_ids.append(input_ids)
                    all_attention_masks.append(attention_mask)
                    all_labels.append(self.label2id[label] if label != -1 else -1)
                    all_sentences.append(sentence)
                else:
                    # Apply sliding window for long texts
                    self.logger.info(f"Applying sliding window to text with {len(input_ids)} tokens")

                    stride = int(self.config.max_length * self.config.window_stride_ratio)
                    for start_idx in range(0, len(input_ids), stride):
                        end_idx = min(start_idx + self.config.max_length, len(input_ids))

                        # Skip if remaining chunk is too small
                        if end_idx - start_idx < self.config.min_chunk_size:
                            continue

                        chunk_input_ids = input_ids[start_idx:end_idx]
                        chunk_attention_mask = attention_mask[start_idx:end_idx]

                        all_input_ids.append(chunk_input_ids)
                        all_attention_masks.append(chunk_attention_mask)
                        all_labels.append(self.label2id[label] if label != -1 else -1)
                        all_sentences.append(sentence[start_idx:end_idx])
                        # If we've reached the end, break
                        if end_idx >= len(input_ids):
                            break

            return {
                'input_ids': all_input_ids,
                'attention_mask': all_attention_masks,
                'label': all_labels,
                'sentence1': all_sentences,
            }

        # Apply tokenization with remove_columns=False to preserve original data
        self.datasets = self.datasets.map(
            tokenize_function,
            batched=True,
            batch_size=1,  # Process one at a time for sliding window
            remove_columns=['sentence1', 'label']
        )

        # Print label distribution
        train_df = pd.DataFrame(self.datasets["train"])
        self.logger.info(f"Label distribution:\n{train_df['label'].value_counts()}")

        # Log statistics about sliding window application
        original_counts = {}
        expanded_counts = {}
        for split in ["train", "validation", "test"]:
            original_count = len(self.datasets[split].unique("label"))
            expanded_count = len(self.datasets[split])
            original_counts[split] = original_count
            expanded_counts[split] = expanded_count

            if expanded_count > original_count:
                self.logger.info(f"{split} set: {original_count} -> {expanded_count} samples "
                               f"(+{expanded_count - original_count} from sliding window)")

    def setup_model_and_datasets(self):
        """Initialize model and prepare TensorFlow datasets."""
        self.logger.info("Setting up model and TensorFlow datasets...")

        # Check for GPU availability
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            try:
                tf.config.experimental.set_memory_growth(physical_devices[0], True)
                device = "/gpu:0"
                self.logger.info("Using GPU for training")
            except:
                device = "/cpu:0"
                self.logger.warning("GPU setup failed, using CPU")
        else:
            device = "/cpu:0"
            self.logger.info("No GPU found, using CPU")

        strategy = tf.distribute.OneDeviceStrategy(device=device)

        with strategy.scope():
            self.model = TFAutoModelForSequenceClassification.from_pretrained(
                self.config.model_name,
                num_labels=len(self.id2label)
            )

            dataset_options = tf.data.Options()
            dataset_options.experimental_distribute.auto_shard_policy = (
                tf.data.experimental.AutoShardPolicy.OFF
            )

            for split in ["train", "validation", "test"]:
                shuffle = (split == "train")

                try:
                    # Try the standard method first
                    tf_dataset = self.model.prepare_tf_dataset(
                        self.datasets[split],
                        shuffle=shuffle,
                        batch_size=self.config.batch_size,
                        tokenizer=self.tokenizer,
                    )
                    self.tf_datasets[split] = tf_dataset.with_options(dataset_options)
                except Exception as e:
                    self.logger.warning(f"Standard dataset preparation failed for {split}: {e}")
                    # Fallback to manual dataset creation
                    self.tf_datasets[split] = self._create_tf_dataset_manual(split, shuffle)

    def _create_tf_dataset_manual(self, split: str, shuffle: bool):
        """Manually create TensorFlow dataset as fallback."""
        dataset = self.datasets[split]

        def gen():
            for item in dataset:
                features = {
                    'input_ids': tf.constant(item['input_ids'], dtype=tf.int32),
                    'attention_mask': tf.constant(item['attention_mask'], dtype=tf.int32),
                }
                label = tf.constant(item['label'], dtype=tf.int64)
                yield features, label

        output_signature = (
            {
                'input_ids': tf.TensorSpec(shape=(None,), dtype=tf.int32),
                'attention_mask': tf.TensorSpec(shape=(None,), dtype=tf.int32),
            },
            tf.TensorSpec(shape=(), dtype=tf.int64)
        )

        tf_dataset = tf.data.Dataset.from_generator(gen, output_signature=output_signature)

        if shuffle:
            tf_dataset = tf_dataset.shuffle(buffer_size=1000)

        tf_dataset = tf_dataset.padded_batch(
            self.config.batch_size,
            padded_shapes=(
                {
                    'input_ids': [None],
                    'attention_mask': [None],
                },
                []
            ),
            padding_values=(
                {
                    'input_ids': self.tokenizer.pad_token_id,
                    'attention_mask': 0,
                },
                -100
            )
        )

        return tf_dataset.prefetch(tf.data.AUTOTUNE)

    def calculate_class_weights(self) -> Dict[int, float]:
        """Calculate class weights for imbalanced data."""
        train_df = pd.DataFrame(self.datasets["train"])
        label_counts = train_df["label"].value_counts()
        total_count = len(train_df)

        class_weights = {}
        for i, count in label_counts.items():
            importance = self.config.class_importance[i]
            weight = importance * total_count / (len(label_counts) * count)
            class_weights[i] = weight
        for i in range(5):
            if i not in class_weights:
                class_weights[i] = 1

        self.logger.info(f"Class weights: {class_weights}")
        return class_weights

    def compile_model(self, num_train_steps: int):
        """Compile the model with optimizer and loss function."""
        try:
            optimizer, _ = create_optimizer(
                init_lr=self.config.learning_rate,
                num_train_steps=num_train_steps,
                num_warmup_steps=self.config.warmup_steps,
                adam_beta1=self.config.adam_beta1,
                adam_beta2=self.config.adam_beta2,
                adam_epsilon=self.config.adam_epsilon,
                weight_decay_rate=self.config.weight_decay,
                adam_global_clipnorm=self.config.max_grad_norm,
            )
        except Exception as e:
            self.logger.warning(f"Failed to create optimizer with transformers: {e}")
            # Fallback to standard Adam optimizer
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=self.config.learning_rate,
                beta_1=self.config.adam_beta1,
                beta_2=self.config.adam_beta2,
                epsilon=self.config.adam_epsilon,
                clipnorm=self.config.max_grad_norm
            )

        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metrics = ["accuracy"]

        self.model.compile(optimizer=optimizer, metrics=metrics, loss=loss_fn)

    def save_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                              epoch: int, suffix: str = ""):
        """Save confusion matrix plot."""
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=range(1, 6), yticklabels=range(1, 6))
        plt.xlabel('Predicted')
        plt.ylabel('True')
        title = f'Confusion Matrix{" " + suffix if suffix else ""}'
        plt.title(title)

        # Add configuration info
        config_text = (f"bs:{self.config.batch_size}, "
                       f"seed:{self.config.seed}, "
                       f"epoch:{epoch}/{self.config.num_epochs}, "
                       f"lr:{self.config.learning_rate}")
        plt.suptitle(config_text, fontsize=8)

        filename = f"confusion_matrix{('_' + suffix) if suffix else ''}.png"
        epoch_dir = self.output_dir / str(epoch)
        plt.savefig(epoch_dir / filename, dpi=600, bbox_inches='tight')
        plt.close()  # Close to prevent memory issues

    def save_results(self, epoch: int, predictions: np.ndarray,
                     true_labels: List[int], sentences: List[str],
                     report: str, suffix: str = ""):
        """Save prediction results and classification report."""
        epoch_dir = self.output_dir / str(epoch)

        # Save detailed results
        filename = f"{'ext_' if suffix else ''}test_results.txt"
        with open(epoch_dir / filename, "w", encoding='utf-8') as f:
            f.write("index;prediction;true_label;sentence\n")
            for idx, (pred, true_label, sentence) in enumerate(
                    zip(predictions, true_labels, sentences)
            ):
                pred_label = self.id2label[pred]
                # true_label_name = self.id2label[true_label]
                # Escape semicolons in sentences
                sentence_clean = sentence.replace(';', '|')
                f.write(f"{idx};{pred_label};{true_label};{sentence_clean}\n")
            f.write(f"\n{report}")

    def evaluate_external_dataset(self, epoch: int):
        """Evaluate on external test dataset."""
        try:
            ext_dataset = load_dataset('json', data_files={
                'ext_test': self.config.data_dir + self.config.ext_test_file
            })
            ext_ds=ext_dataset["ext_test"].add_column("idx",range(ext_dataset["ext_test"].num_rows))
            ext_dataset["ext_test"]=ext_ds
            # data_with_idx = []
            # for idx, item in enumerate(ext_dataset['ext_test']):
            #     item['idx'] = idx
            #     data_with_idx.append(item)
            # ext_dataset['ext_test'] = data_with_idx
            # Store original sentences before tokenization
            original_sentences = ext_dataset["ext_test"]['sentence1']
            original_labels = ext_dataset["ext_test"]['label']

            def tokenize_function(examples):
                # Handle both single and batch inputs
                if isinstance(examples['sentence1'], str):
                    sentences = [examples['sentence1']]
                    labels = [examples['label']]
                    idxx=[examples['idx']]
                else:
                    sentences = examples['sentence1']
                    labels = examples['label']
                    idxx=examples['idx']

                all_input_ids = []
                all_attention_masks = []
                all_labels = []
                all_sentences = []
                all_original_indices = []  # Track original sample index

                for (idx, sentence, label) in zip(idxx,sentences, labels):
                    tokens = self.tokenizer(
                        sentence,
                        truncation=False,
                        padding=False,
                        return_tensors=None
                    )

                    input_ids = tokens['input_ids']
                    attention_mask = tokens['attention_mask']

                    if len(input_ids) <= self.config.max_length:
                        all_input_ids.append(input_ids)
                        all_attention_masks.append(attention_mask)
                        all_labels.append(self.label2id[label] if label != -1 else -1)
                        all_sentences.append(sentence)
                        all_original_indices.append(idx)
                    else:
                        # Apply sliding window
                        self.logger.info(f"Applying sliding window to ext_datas text with {len(input_ids)} tokens")

                        stride = int(self.config.max_length * self.config.window_stride_ratio)

                        for start_idx in range(0, len(input_ids), stride):
                            end_idx = min(start_idx + self.config.max_length, len(input_ids))

                            if end_idx - start_idx < self.config.min_chunk_size:
                                continue

                            chunk_input_ids = input_ids[start_idx:end_idx]
                            chunk_attention_mask = attention_mask[start_idx:end_idx]

                            all_input_ids.append(chunk_input_ids)
                            all_attention_masks.append(chunk_attention_mask)
                            all_labels.append(self.label2id[label] if label != -1 else -1)
                            all_sentences.append(sentence[start_idx:end_idx])
                            all_original_indices.append(idx)

                            if end_idx >= len(input_ids):
                                break

                return {
                    'input_ids': all_input_ids,
                    'attention_mask': all_attention_masks,
                    'label': all_labels,
                    'sentence1': all_sentences,
                    'original_index': all_original_indices,
                    'idx': all_original_indices,
                }

            ext_dataset = ext_dataset.select_columns(['sentence1', 'label','idx'])

            print("dataset filtrelendi")
            ext_dataset = ext_dataset.map(tokenize_function,
                                          batch_size=self.config.batch_size,
                                          batched=True,
                                          remove_columns=['sentence1', 'label','idx']
                                          )
            print("dataset maplendi")
            # Prepare TensorFlow dataset
            try:
                print("tf dataset olusturuluyor")
                tf_ext_dataset = self.model.prepare_tf_dataset(
                    ext_dataset["ext_test"],
                    shuffle=False,
                    batch_size=self.config.batch_size,
                    tokenizer=self.tokenizer,
                )
            except Exception as e:
                self.logger.warning(f"Standard external dataset preparation failed: {e}")
                # Use manual method
                tf_ext_dataset = self._create_tf_dataset_manual_ext(ext_dataset["ext_test"], shuffle=False)

            # Make predictions
            predictions = self.model.predict(tf_ext_dataset, verbose=0)

            # Handle different output formats
            if isinstance(predictions, dict):
                logits = predictions["logits"]
            else:
                logits = predictions

            predicted_classes = np.argmax(logits, axis=1)

            # Aggregate predictions for samples that were split
            original_indices = ext_dataset["ext_test"]['original_index']
            # original_indices = ext_dataset["ext_test"]['idx']
            aggregated_predictions = {}

            for idx, (pred, orig_idx) in enumerate(zip(predicted_classes, original_indices)):
                if orig_idx not in aggregated_predictions:
                    aggregated_predictions[orig_idx] = []
                aggregated_predictions[orig_idx].append(pred)

            # Use majority voting for aggregation
            final_predictions = []
            for orig_idx in range(len(original_labels)):
                if orig_idx in aggregated_predictions:
                    preds = aggregated_predictions[orig_idx]
                    # Majority vote
                    final_pred = max(set(preds), key=preds.count)
                    final_predictions.append(final_pred)
                else:
                    # This shouldn't happen, but just in case
                    final_predictions.append(0)

            # Evaluate with original labels
            y_true = np.array([self.label2id[label] for label in original_labels]) + 1
            y_pred = np.array(final_predictions) + 1

            # Calculate class-specific accuracies
            class_accuracies = {}
            for class_id in [1, 2]:
                mask = (y_true == class_id)
                if np.sum(mask) > 0:
                    accuracy = np.mean(y_pred[mask] == y_true[mask])
                    class_accuracies[class_id] = accuracy
                    self.logger.info(f"External accuracy class {class_id}: {accuracy:.4f}")

            # Save results
            epoch_dir = self.output_dir / str(epoch)
            acc_filename = f"extacc_s1_{int(100 * class_accuracies.get(1, 0))}_s2_{int(100 * class_accuracies.get(2, 0))}.txt"
            with open(epoch_dir / acc_filename, "w") as f:
                f.write(f"ext_accuracy_class1: {class_accuracies.get(1, 0):.4f}\n")
                f.write(f"ext_accuracy_class2: {class_accuracies.get(2, 0):.4f}\n")

            # Generate and save confusion matrix
            self.save_confusion_matrix(y_true, y_pred, epoch, "external")

            # Generate and save classification report
            report = classification_report(y_true, y_pred,
                                           target_names=[str(i) for i in range(1, 6)])
            self.save_results(epoch, final_predictions,
                              original_labels,
                              original_sentences,
                              report, "ext")

        except Exception as e:
            self.logger.warning(f"External dataset evaluation failed: {e}")

    def _create_tf_dataset_manual_ext(self, dataset, shuffle: bool):
        """Manually create TensorFlow dataset for external evaluation."""
        def gen():
            for item in dataset:
                features = {
                    'input_ids': tf.constant(item['input_ids'], dtype=tf.int32),
                    'attention_mask': tf.constant(item['attention_mask'], dtype=tf.int32),
                }
                label = tf.constant(item['label'], dtype=tf.int64)
                yield features, label

        output_signature = (
            {
                'input_ids': tf.TensorSpec(shape=(None,), dtype=tf.int32),
                'attention_mask': tf.TensorSpec(shape=(None,), dtype=tf.int32),
            },
            tf.TensorSpec(shape=(), dtype=tf.int64)
        )

        tf_dataset = tf.data.Dataset.from_generator(gen, output_signature=output_signature)

        if shuffle:
            tf_dataset = tf_dataset.shuffle(buffer_size=1000)

        tf_dataset = tf_dataset.padded_batch(
            self.config.batch_size,
            padded_shapes=(
                {
                    'input_ids': [None],
                    'attention_mask': [None],
                },
                []
            ),
            padding_values=(
                {
                    'input_ids': self.tokenizer.pad_token_id,
                    'attention_mask': 0,
                },
                -100
            )
        )

        return tf_dataset.prefetch(tf.data.AUTOTUNE)

    def train_single_epoch(self, epoch: int, class_weights: Dict[int, float]):
        """Train for a single epoch and evaluate."""
        epoch_dir = self.output_dir / str(epoch)
        epoch_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Starting epoch {epoch + 1}/{self.config.num_epochs}")

        # Calculate steps
        train_count = len(self.datasets["train"])
        val_count = len(self.datasets["validation"])
        train_steps = train_count // self.config.batch_size
        val_steps = val_count // self.config.batch_size

        try:
            # Train for one epoch
            history = self.model.fit(
                self.tf_datasets["train"],
                validation_data=self.tf_datasets["validation"],
                epochs=1,
                steps_per_epoch=train_steps,
                validation_steps=val_steps,
                verbose=1,
                class_weight=class_weights,
            )
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            # Try without class weights if they're causing issues
            self.logger.info("Retrying without class weights...")
            history = self.model.fit(
                self.tf_datasets["train"],
                validation_data=self.tf_datasets["validation"],
                epochs=1,
                steps_per_epoch=train_steps,
                validation_steps=val_steps,
                verbose=1
            )

        # Evaluate on validation set
        val_loss, val_accuracy = self.model.evaluate(
            self.tf_datasets["validation"], verbose=0
        )

        # Test predictions
        test_predictions = self.model.predict(
            self.tf_datasets["test"], verbose=0
        )

        # Handle different output formats
        if isinstance(test_predictions, dict):
            logits = test_predictions["logits"]
        else:
            logits = test_predictions

        predicted_classes = np.argmax(logits, axis=1)

        # Calculate metrics
        y_true = np.array(self.datasets['test']['label']) + 1
        y_pred = predicted_classes + 1
        test_accuracy = accuracy_score(self.datasets['test']['label'], predicted_classes)

        # Generate reports
        report = classification_report(y_true, y_pred,
                                       target_names=[str(i) for i in range(1, 6)])

        # Save results
        self.save_confusion_matrix(y_true, y_pred, epoch)
        self.save_results(epoch, predicted_classes,
                          self.datasets['test']['label'],
                          self.datasets['test']['sentence1'], report)

        # Save model
        model_dir = epoch_dir / "model"
        self.model.save_pretrained(model_dir)
        self.tokenizer.save_pretrained(model_dir)

        # Save training logs
        with open(epoch_dir / "output.txt", "w") as f:
            f.write(f"Epoch {epoch + 1}/{self.config.num_epochs}\n")
            f.write(f"Validation loss: {val_loss:.5f}\n")
            f.write(f"Validation accuracy: {val_accuracy * 100:.4f}%\n")
            f.write(f"Test accuracy: {test_accuracy * 100:.4f}%\n")
            f.write(f"\nClassification Report:\n{report}\n")

        # Evaluate external dataset
        self.evaluate_external_dataset(epoch)

        self.logger.info(f"Epoch {epoch + 1} completed. "
                         f"Val acc: {val_accuracy:.4f}, Test acc: {test_accuracy:.4f}")

        return {
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'test_accuracy': test_accuracy
        }

    def train(self):
        """Main training loop."""
        self.logger.info("Starting training process...")

        # Load and prepare data
        self.load_data()
        self.setup_tokenizer_and_labels()
        self.tokenize_datasets()
        self.setup_model_and_datasets()

        # Calculate training parameters
        train_count = len(self.datasets["train"])
        num_train_steps = (train_count // self.config.batch_size) * self.config.num_epochs

        # Setup model
        self.compile_model(num_train_steps)
        class_weights = self.calculate_class_weights()

        # Training loop
        results = []
        for epoch in range(self.config.num_epochs):
            epoch_results = self.train_single_epoch(epoch, class_weights)
            results.append(epoch_results)

        # Save final results summary
        with open(self.output_dir / "training_summary.json", "w") as f:
            json.dump({
                'config': self.config.__dict__,
                'results': results
            }, f, indent=2)

        self.logger.info(f"Training completed! Results saved to {self.output_dir}")
