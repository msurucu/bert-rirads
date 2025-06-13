"""
Data processing utilities for RI-RADS text classification.

This module handles data loading, preprocessing, and the sliding window
approach for handling long radiology reports.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict


class DataProcessor:
    """Handles data loading and basic preprocessing."""

    def __init__(self, config):
        """
        Initialize data processor.

        Args:
            config: Training configuration object
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

    def load_datasets(self) -> DatasetDict:
        """
        Load datasets from JSON files.

        Returns:
            Dictionary of datasets (train, validation, test)
        """
        data_files = {
            'train': str(Path(self.config.data_dir) / self.config.train_file),
            'validation': str(Path(self.config.data_dir) / self.config.val_file),
            'test': str(Path(self.config.data_dir) / self.config.test_file),
        }

        # Check if files exist
        for split, filepath in data_files.items():
            if not Path(filepath).exists():
                self.logger.warning(f"{split} file not found: {filepath}")
                data_files.pop(split)

        # Load datasets
        datasets = load_dataset('json', data_files=data_files)

        # Load external test set if specified
        if self.config.external_test_file:
            ext_path = Path(self.config.data_dir) / self.config.external_test_file
            if ext_path.exists():
                ext_dataset = load_dataset('json', data_files={'test': str(ext_path)})
                datasets['external_test'] = ext_dataset['test']

        return datasets

    def tokenize_dataset(self, dataset: Dataset, tokenizer, label2id: Dict) -> Dataset:
        """
        Tokenize dataset without sliding window.

        Args:
            dataset: Dataset to tokenize
            tokenizer: Tokenizer instance
            label2id: Label to ID mapping

        Returns:
            Tokenized dataset
        """

        def tokenize_function(examples):
            # Handle both single and batch inputs
            is_batched = isinstance(examples['sentence1'], list)

            if not is_batched:
                texts = [examples['sentence1']]
                labels = [examples['label']]
            else:
                texts = examples['sentence1']
                labels = examples['label']

            # Tokenize
            tokenized = tokenizer(
                texts,
                truncation=True,
                padding=False,
                max_length=self.config.max_length,
                return_tensors=None
            )

            # Convert labels
            tokenized['label'] = [
                label2id.get(str(label), label) if label != -1 else -1
                for label in labels
            ]

            return tokenized

        # Apply tokenization
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=['sentence1', 'label']
        )

        return tokenized_dataset

    def prepare_external_dataset(self, filepath: str) -> Dataset:
        """
        Prepare external dataset for evaluation.

        Args:
            filepath: Path to external dataset

        Returns:
            Loaded dataset
        """
        data = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                data.append(item)

        return Dataset.from_pandas(pd.DataFrame(data))


class SlidingWindowProcessor:
    """Handles sliding window approach for long texts."""

    def __init__(self, config, tokenizer):
        """
        Initialize sliding window processor.

        Args:
            config: Training configuration
            tokenizer: Tokenizer instance
        """
        self.config = config
        self.tokenizer = tokenizer
        self.logger = logging.getLogger(__name__)

    def process_dataset(self, dataset: Dataset, label2id: Dict) -> Dataset:
        """
        Process dataset with sliding window for long texts.

        Args:
            dataset: Dataset to process
            label2id: Label to ID mapping

        Returns:
            Processed dataset with sliding windows applied
        """
        # Add index to track original samples
        dataset = dataset.add_column("original_idx", range(len(dataset)))

        def apply_sliding_window(examples):
            """Apply sliding window to handle long texts."""
            # Handle single vs batch
            is_batched = isinstance(examples['sentence1'], list)

            if not is_batched:
                texts = [examples['sentence1']]
                labels = [examples['label']]
                indices = [examples['original_idx']]
            else:
                texts = examples['sentence1']
                labels = examples['label']
                indices = examples['original_idx']

            # Process each text
            all_input_ids = []
            all_attention_masks = []
            all_labels = []
            all_original_indices = []
            all_window_positions = []

            for text, label, idx in zip(texts, labels, indices):
                # Tokenize without truncation first
                tokens = self.tokenizer(
                    text,
                    truncation=False,
                    padding=False,
                    return_tensors=None
                )

                input_ids = tokens['input_ids']
                attention_mask = tokens['attention_mask']

                # Convert label
                label_id = label2id.get(str(label), label) if label != -1 else -1

                # Check if sliding window is needed
                if len(input_ids) <= self.config.max_length:
                    # No sliding window needed
                    all_input_ids.append(input_ids)
                    all_attention_masks.append(attention_mask)
                    all_labels.append(label_id)
                    all_original_indices.append(idx)
                    all_window_positions.append("full")
                else:
                    # Apply sliding window
                    windows = self._create_sliding_windows(
                        input_ids,
                        attention_mask,
                        self.config.max_length,
                        self.config.window_stride_ratio
                    )

                    for i, (window_ids, window_mask) in enumerate(windows):
                        all_input_ids.append(window_ids)
                        all_attention_masks.append(window_mask)
                        all_labels.append(label_id)
                        all_original_indices.append(idx)
                        all_window_positions.append(f"window_{i}")

                    self.logger.debug(
                        f"Text {idx}: {len(input_ids)} tokens -> {len(windows)} windows"
                    )

            return {
                'input_ids': all_input_ids,
                'attention_mask': all_attention_masks,
                'label': all_labels,
                'original_idx': all_original_indices,
                'window_position': all_window_positions
            }

        # Apply sliding window processing
        processed_dataset = dataset.map(
            apply_sliding_window,
            batched=True,
            batch_size=1,  # Process one at a time for proper window handling
            remove_columns=['sentence1', 'label', 'original_idx']
        )

        # Log statistics
        original_size = len(dataset)
        processed_size = len(processed_dataset)
        expansion_factor = processed_size / original_size

        self.logger.info(
            f"Sliding window applied: {original_size} -> {processed_size} samples "
            f"(expansion factor: {expansion_factor:.2f})"
        )

        return processed_dataset

    def _create_sliding_windows(
            self,
            input_ids: List[int],
            attention_mask: List[int],
            max_length: int,
            stride_ratio: float
    ) -> List[Tuple[List[int], List[int]]]:
        """
        Create sliding windows for long sequences.

        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            max_length: Maximum window length
            stride_ratio: Overlap ratio between windows

        Returns:
            List of (input_ids, attention_mask) tuples for each window
        """
        windows = []
        stride = int(max_length * stride_ratio)

        # Ensure we have special tokens
        cls_token_id = self.tokenizer.cls_token_id
        sep_token_id = self.tokenizer.sep_token_id

        for start_idx in range(0, len(input_ids), stride):
            end_idx = min(start_idx + max_length, len(input_ids))

            # Skip if remaining chunk is too small
            if end_idx - start_idx < self.config.min_chunk_size:
                break

            # Extract window
            window_ids = input_ids[start_idx:end_idx]
            window_mask = attention_mask[start_idx:end_idx]

            # Ensure special tokens are present
            if start_idx > 0 and window_ids[0] != cls_token_id:
                # Add CLS token at beginning
                window_ids = [cls_token_id] + window_ids[1:]
                window_mask = [1] + window_mask[1:]

            if end_idx < len(input_ids) and window_ids[-1] != sep_token_id:
                # Add SEP token at end
                window_ids = window_ids[:-1] + [sep_token_id]
                window_mask = window_mask[:-1] + [1]

            windows.append((window_ids, window_mask))

            # Break if we've processed the entire sequence
            if end_idx >= len(input_ids):
                break

        return windows

    def aggregate_predictions(
            self,
            predictions: List[int],
            original_indices: List[int],
            aggregation_method: str = "majority_vote"
    ) -> Dict[int, int]:
        """
        Aggregate predictions from multiple windows per sample.

        Args:
            predictions: List of predictions
            original_indices: List of original sample indices
            aggregation_method: Method for aggregation

        Returns:
            Dictionary mapping original index to final prediction
        """
        from collections import defaultdict, Counter

        # Group predictions by original index
        predictions_by_sample = defaultdict(list)
        for pred, idx in zip(predictions, original_indices):
            predictions_by_sample[idx].append(pred)

        # Aggregate predictions
        final_predictions = {}

        for idx, preds in predictions_by_sample.items():
            if aggregation_method == "majority_vote":
                # Most common prediction
                final_predictions[idx] = Counter(preds).most_common(1)[0][0]
            elif aggregation_method == "max":
                # Highest risk category (for RI-RADS)
                final_predictions[idx] = max(preds)
            elif aggregation_method == "average":
                # Average prediction (rounded)
                final_predictions[idx] = round(sum(preds) / len(preds))
            else:
                raise ValueError(f"Unknown aggregation method: {aggregation_method}")

        return final_predictions


class DataAugmentation:
    """Optional data augmentation for radiology reports."""

    def __init__(self, config):
        """Initialize data augmentation."""
        self.config = config
        self.logger = logging.getLogger(__name__)

    def augment_dataset(self, dataset: Dataset) -> Dataset:
        """
        Apply data augmentation techniques.

        Args:
            dataset: Dataset to augment

        Returns:
            Augmented dataset
        """
        # Implement augmentation strategies if needed
        # - Synonym replacement for medical terms
        # - Back translation
        # - Paraphrasing

        self.logger.info("Data augmentation not implemented yet")
        return dataset