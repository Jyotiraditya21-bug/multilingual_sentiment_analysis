"""Preprocess script for Multilingual Sentiment Analysis.

Downloads mteb/amazon_reviews_multi dataset for English and Spanish, cleans text,
subsamples stratifiably, balances classes using WeightedRandomSampler, tokenizes,
and saves artifacts.
"""

import argparse
import json
import os
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import AutoTokenizer

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)


class SentimentDataset(Dataset):
    """PyTorch Dataset for Sentiment Analysis."""

    def __init__(self, encodings: Dict[str, torch.Tensor], labels: List[int]):
        """Initialize Dataset.

        Args:
            encodings: Tokenized inputs from tokenizer.
            labels: List of integer sentiment labels.
        """
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get preprocessed sample by index.

        Args:
            idx: index.

        Returns:
            Dictionary containing input_ids, attention_mask, and labels.
        """
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self) -> int:
        """Return length of dataset."""
        return len(self.labels)


def clean_text(text: str, lowercase: bool = False) -> str:
    """Clean review text.

    Strips HTML tags, removes excessive whitespaces, and optionally lowercases.

    Args:
        text: Raw review string.
        lowercase: Whether to lowercase the text.

    Returns:
        Cleaned text string.
    """
    if not isinstance(text, str):
        return ""
    # Strip HTML tags
    text = re.sub(r"<[^>]*>", "", text)
    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text).strip()
    if lowercase:
        text = text.lower()
    return text


def stratify_subsample(
    df: pd.DataFrame, target_col: str, total_samples: int
) -> pd.DataFrame:
    """Subsample a dataframe stratifiably based on the target column.

    Args:
        df: Input DataFrame.
        target_col: Column to stratify on.
        total_samples: Total number of samples desired.

    Returns:
        Stratified subsampled DataFrame.
    """
    counts = df[target_col].value_counts()
    ratios = counts / counts.sum()
    samples_per_class = (ratios * total_samples).round().astype(int)

    # Adjust for rounding issues
    diff = total_samples - samples_per_class.sum()
    if diff != 0:
        samples_per_class.iloc[0] += diff

    sampled_dfs = []
    for cls, n_samples in samples_per_class.items():
        cls_df = df[df[target_col] == cls]
        n_to_sample = min(len(cls_df), n_samples)
        sampled_dfs.append(cls_df.sample(n=n_to_sample, random_state=42))

    return (
        pd.concat(sampled_dfs)
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )


def load_and_preprocess_data(
    sample_size_per_lang: int = 50000, lowercase: bool = False
) -> pd.DataFrame:
    """Load HuggingFace mteb/amazon_reviews_multi and clean/sample.

    Args:
        sample_size_per_lang: How many samples to keep per language.
        lowercase: Whether to lowercase the text.

    Returns:
        Combined cleaned and subsampled pandas DataFrame.
    """
    languages = ["en", "es"]
    dfs = []

    for lang in languages:
        print(f"Loading and processing {lang} subset...")
        dataset = load_dataset("mteb/amazon_reviews_multi", lang, split="train")
        df = pd.DataFrame(dataset)
        df["language"] = lang

        # MTEB Labels are 0-4 representing 1-5 stars.
        # Stars rating maps:
        # label 0 (1★) & 1 (2★) -> negative (0)
        # label 2 (3★) -> neutral (1)
        # label 3 (4★) -> dropped
        # label 4 (5★) -> positive (2)
        df = df[df["label"] != 3].copy()

        def map_mteb_label(label: int) -> int:
            if label == 4:
                return 2  # positive
            elif label == 2:
                return 1  # neutral
            else:
                return 0  # negative (labels 0 and 1)

        df["target_label"] = df["label"].apply(map_mteb_label)
        df["cleaned_review"] = df["text"].apply(
            lambda x: clean_text(x, lowercase=lowercase)
        )

        # Drop rows with empty cleaned reviews
        df = df[df["cleaned_review"].str.strip() != ""]

        # Rename label for standard preprocess sampling
        df = df.rename(columns={"target_label": "class_label"})

        # Stratified subsampling
        df_sampled = stratify_subsample(df, "class_label", sample_size_per_lang)
        dfs.append(df_sampled)

    combined_df = pd.concat(dfs).sample(frac=1, random_state=42).reset_index(drop=True)
    combined_df = combined_df.drop(columns=["label"])
    combined_df = combined_df.rename(columns={"class_label": "label"})
    return combined_df


def split_data(
    df: pd.DataFrame, train_ratio: float = 0.8, val_ratio: float = 0.1
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataframe into train, validation, and test sets (stratified).

    Args:
        df: Input DataFrame.
        train_ratio: Ratio for training.
        val_ratio: Ratio for validation.

    Returns:
        Tuple of train, validation, and test DataFrames.
    """
    train_size = int(len(df) * train_ratio)
    val_size = int(len(df) * val_ratio)

    # Shuffle df
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Let's perform a simple split since it's already shuffled and stratified
    train_df = df.iloc[:train_size].reset_index(drop=True)
    val_df = df.iloc[train_size : train_size + val_size].reset_index(drop=True)
    test_df = df.iloc[train_size + val_size :].reset_index(drop=True)

    return train_df, val_df, test_df


def get_dataloaders(
    batch_size: int = 32,
    max_len: int = 128,
    sample_size_per_lang: int = 50000,
    lowercase: bool = False,
    output_dir: str = "model/artifacts/",
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Generate PyTorch DataLoaders for train, val, and test.

    Also saves tokenizer and label mappings to output_dir.

    Args:
        batch_size: DataLoader batch size.
        max_len: Max sequence length for tokenization.
        sample_size_per_lang: Subsample size per language (English/Spanish).
        lowercase: Whether to lowercase raw text.
        output_dir: Output directory to save artifacts.

    Returns:
        Tuple of Train, Val, and Test DataLoaders.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load and Clean Dataset
    df = load_and_preprocess_data(
        sample_size_per_lang=sample_size_per_lang, lowercase=lowercase
    )

    print("\nInitial combined class distribution before balancing:")
    print(df["label"].value_counts().sort_index())

    # 2. Split into Train/Val/Test (80/10/10)
    train_df, val_df, test_df = split_data(df, 0.8, 0.1)

    print(f"\nSplit Sizes: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # 3. Tokenizer initialization
    tokenizer_name = "xlm-roberta-base"
    print(f"\nInitializing tokenizer: {tokenizer_name}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.save_pretrained(output_dir)

    # Save label mappings
    label_mapping = {"negative": 0, "neutral": 1, "positive": 2}
    with open(os.path.join(output_dir, "label_mapping.json"), "w") as f:
        json.dump(label_mapping, f, indent=4)

    # 4. Tokenization of datasets
    print("Tokenizing datasets in memory...")
    train_encodings = tokenizer(
        train_df["cleaned_review"].tolist(),
        truncation=True,
        padding="max_length",
        max_length=max_len,
    )
    val_encodings = tokenizer(
        val_df["cleaned_review"].tolist(),
        truncation=True,
        padding="max_length",
        max_length=max_len,
    )
    test_encodings = tokenizer(
        test_df["cleaned_review"].tolist(),
        truncation=True,
        padding="max_length",
        max_length=max_len,
    )

    # 5. Create PyTorch datasets
    train_dataset = SentimentDataset(train_encodings, train_df["label"].tolist())
    val_dataset = SentimentDataset(val_encodings, val_df["label"].tolist())
    test_dataset = SentimentDataset(test_encodings, test_df["label"].tolist())

    # 6. Weighted Random Sampler for training class balance
    train_labels = train_df["label"].to_numpy()
    class_counts = np.bincount(train_labels)
    print(f"\nTraining set class counts before balancing sampler: {class_counts}")

    class_weights = 1.0 / class_counts
    sample_weights = torch.tensor(
        [class_weights[label] for label in train_labels], dtype=torch.double
    )

    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )

    # Simulate batch distributions after balancing
    simulated_labels = []
    indices = list(sampler)
    for idx in indices[:1000]:
        simulated_labels.append(train_labels[idx])
    simulated_counts = np.bincount(simulated_labels)
    print(
        f"Simulated sample distribution with Balanced Sampler (first 1000 samples): {simulated_counts}"
    )

    # 7. Create DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=sampler
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess Amazon Multilingual Reviews Dataset from MTEB"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="DataLoader batch size"
    )
    parser.add_argument(
        "--max_len", type=int, default=128, help="Tokenizer maximum sequence length"
    )
    parser.add_argument(
        "--sample_size_per_lang",
        type=int,
        default=50000,
        help="Subsample size per language subset",
    )
    parser.add_argument(
        "--lowercase", action="store_true", help="Whether to lowercase text"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="model/artifacts/",
        help="Directory to save preprocessed configs/tokenizer",
    )
    args = parser.parse_args()

    print("--- PREPROCESSING DRY RUN ---")
    train_dl, val_dl, test_dl = get_dataloaders(
        batch_size=args.batch_size,
        max_len=args.max_len,
        sample_size_per_lang=args.sample_size_per_lang,
        lowercase=args.lowercase,
        output_dir=args.output_dir,
    )
    print("Preprocessing completed successfully!")
    print(f"Artifacts saved to: {args.output_dir}")
