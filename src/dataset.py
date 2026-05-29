"""Dataset and DataLoader utilities for IMDb sentiment classification."""

from __future__ import annotations

import os
from typing import Sequence

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src import config


class IMDbDataset(Dataset):
    """PyTorch Dataset for tokenized IMDb reviews.

    Args:
        texts: Preprocessed review texts.
        labels: Integer labels where 0 is negative and 1 is positive.
        tokenizer: Hugging Face tokenizer paired with the target model.
        max_length: Maximum token sequence length after padding/truncation.
    """

    def __init__(
        self,
        texts: Sequence[str],
        labels: Sequence[int],
        tokenizer,
        max_length: int = config.MAX_LENGTH,
    ) -> None:
        if len(texts) != len(labels):
            raise ValueError("Texts and labels must have the same length.")
        self.texts = list(texts)
        self.labels = [int(label) for label in labels]
        self.tokenizer = tokenizer
        self.max_length = int(max_length)

    def __len__(self) -> int:
        """Return the number of examples in the dataset."""
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Tokenize one review and return tensors for model input."""
        text = str(self.texts[idx])
        label = self.labels[idx]

        # Fixed-length padding keeps every batch tensor rectangular for the GPU.
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }


def load_data(split: str) -> tuple[list[str], list[int]]:
    """Load one processed split from disk.

    Args:
        split: One of ``train``, ``validation``, or ``test``.

    Returns:
        Pair of review texts and integer labels.

    Raises:
        ValueError: If the split name is unsupported.
        FileNotFoundError: If the expected CSV file is missing.
    """
    valid_splits = {"train", "validation", "test"}
    if split not in valid_splits:
        raise ValueError(f"Unsupported split '{split}'. Expected one of {sorted(valid_splits)}.")

    path = os.path.join(config.DATA_DIR, f"{split}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Processed data file not found: {path}")

    df = pd.read_csv(path)
    required_columns = {"clean_text", "label_id"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing_columns)}")

    texts = df["clean_text"].astype(str).tolist()
    labels = df["label_id"].astype(int).tolist()
    return texts, labels


def create_dataloaders(
    tokenizer,
    max_length: int = config.MAX_LENGTH,
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test DataLoaders.

    Args:
        tokenizer: Hugging Face tokenizer for the selected model.
        max_length: Maximum token sequence length.
        batch_size: Number of samples per batch.
        num_workers: Parallel worker processes for data loading.

    Returns:
        Tuple containing train, validation, and test DataLoaders.
    """
    train_texts, train_labels = load_data("train")
    val_texts, val_labels = load_data("validation")
    test_texts, test_labels = load_data("test")

    train_dataset = IMDbDataset(train_texts, train_labels, tokenizer, max_length=max_length)
    val_dataset = IMDbDataset(val_texts, val_labels, tokenizer, max_length=max_length)
    test_dataset = IMDbDataset(test_texts, test_labels, tokenizer, max_length=max_length)

    pin_memory = torch.cuda.is_available()
    generator = torch.Generator()
    generator.manual_seed(config.SEED)

    common_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        common_kwargs["persistent_workers"] = True

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=True,
        generator=generator,
        **common_kwargs,
    )
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, **common_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, drop_last=False, **common_kwargs)
    return train_loader, val_loader, test_loader

