"""
Data processing utilities for medical text classification.

This module contains functions for loading, cleaning, and preprocessing
medical text data for machine learning tasks.
"""

import pandas as pd
import numpy as np
import re
from typing import List, Dict, Tuple, Optional, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import logging

logger = logging.getLogger(__name__)


def load_medical_data(file_path: str, encoding: str = "utf-8") -> pd.DataFrame:
    """
    Load medical dataset from CSV file.

    Args:
        file_path: Path to the CSV file
        encoding: File encoding

    Returns:
        Loaded DataFrame
    """
    try:
        df = pd.read_csv(file_path, encoding=encoding)
        logger.info(f"Successfully loaded {len(df)} records from {file_path}")
        return df
    except UnicodeDecodeError:
        logger.warning("UTF-8 decoding failed, trying latin-1")
        df = pd.read_csv(file_path, encoding="latin-1")
        logger.info(f"Successfully loaded {len(df)} records with latin-1 encoding")
        return df
    except Exception as e:
        logger.error(f"Error loading file {file_path}: {e}")
        raise


def clean_medical_text(
    df: pd.DataFrame,
    text_columns: List[str] = ["title", "abstract"],
    combine_columns: bool = True,
    combined_column: str = "text",
) -> pd.DataFrame:
    """
    Clean and preprocess medical text data.

    Args:
        df: Input DataFrame
        text_columns: Columns containing text to clean
        combine_columns: Whether to combine text columns
        combined_column: Name of combined text column

    Returns:
        DataFrame with cleaned text
    """
    df_clean = df.copy()

    for col in text_columns:
        if col in df_clean.columns:
            # Fill missing values
            df_clean[col] = df_clean[col].fillna("")

            # Convert to string
            df_clean[col] = df_clean[col].astype(str)

            # Clean text
            df_clean[col] = df_clean[col].apply(clean_text)

    # Combine text columns if requested
    if combine_columns and len(text_columns) > 1:
        existing_cols = [col for col in text_columns if col in df_clean.columns]
        if existing_cols:
            df_clean[combined_column] = df_clean[existing_cols].apply(
                lambda x: ". ".join(filter(None, x.astype(str))), axis=1
            )

    logger.info(f"Cleaned text in columns: {text_columns}")
    return df_clean


def clean_text(text: str) -> str:
    """
    Clean individual text string.

    Args:
        text: Input text

    Returns:
        Cleaned text
    """
    if pd.isna(text) or text == "":
        return ""

    # Convert to string
    text = str(text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove special characters but keep medical abbreviations
    text = re.sub(r"[^\w\s\-\.\,\:\;\(\)\[\]]", " ", text)

    # Fix common OCR errors
    text = text.replace("~", "-")
    text = text.replace("|", "I")

    # Normalize whitespace
    text = text.strip()

    return text


def extract_medical_features(
    df: pd.DataFrame, text_column: str = "text"
) -> pd.DataFrame:
    """
    Extract medical-specific features from text.

    Args:
        df: Input DataFrame
        text_column: Column containing text

    Returns:
        DataFrame with additional feature columns
    """
    df_features = df.copy()

    # Text length features
    df_features["text_length"] = df_features[text_column].str.len()
    df_features["word_count"] = df_features[text_column].str.split().str.len()

    # Medical term counts
    medical_patterns = {
        "medication_count": r"\b(?:drug|medication|treatment|therapy|medicine)\b",
        "symptom_count": r"\b(?:pain|symptom|discomfort|ache|fever|nausea)\b",
        "anatomy_count": r"\b(?:heart|brain|liver|kidney|lung|blood|cell)\b",
        "procedure_count": r"\b(?:surgery|operation|procedure|examination|test)\b",
        "diagnosis_count": r"\b(?:diagnosis|condition|disease|disorder|syndrome)\b",
    }

    for feature_name, pattern in medical_patterns.items():
        df_features[feature_name] = df_features[text_column].str.count(
            pattern, flags=re.IGNORECASE
        )

    # Numerical values (dosages, measurements)
    df_features["numeric_count"] = df_features[text_column].str.count(r"\b\d+\.?\d*\b")

    # Abbreviation count
    df_features["abbreviation_count"] = df_features[text_column].str.count(
        r"\b[A-Z]{2,}\b"
    )

    logger.info(f"Extracted medical features for {len(df_features)} records")
    return df_features


def analyze_label_distribution(
    df: pd.DataFrame, label_columns: List[str]
) -> Dict[str, Any]:
    """
    Analyze the distribution of labels in the dataset.

    Args:
        df: DataFrame containing labels
        label_columns: Names of label columns

    Returns:
        Dictionary with distribution statistics
    """
    stats = {}

    # Overall statistics
    total_samples = len(df)
    stats["total_samples"] = total_samples

    # Per-class statistics
    class_stats = {}
    for col in label_columns:
        if col in df.columns:
            positive_count = df[col].sum()
            negative_count = total_samples - positive_count

            class_stats[col] = {
                "positive_count": int(positive_count),
                "negative_count": int(negative_count),
                "positive_ratio": positive_count / total_samples,
                "imbalance_ratio": (
                    negative_count / positive_count
                    if positive_count > 0
                    else float("inf")
                ),
            }

    stats["per_class"] = class_stats

    # Multi-label statistics
    label_matrix = df[label_columns].values
    stats["avg_labels_per_sample"] = label_matrix.sum(axis=1).mean()
    stats["max_labels_per_sample"] = label_matrix.sum(axis=1).max()
    stats["min_labels_per_sample"] = label_matrix.sum(axis=1).min()

    # Label combinations
    label_combinations = (
        df[label_columns].apply(lambda x: tuple(x.values), axis=1).value_counts()
    )
    stats["unique_combinations"] = len(label_combinations)
    stats["most_common_combination"] = dict(label_combinations.head(5))

    logger.info(f"Analyzed label distribution for {len(label_columns)} classes")
    return stats


def split_medical_data(
    df: pd.DataFrame,
    text_column: str,
    label_columns: List[str],
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    stratify_on: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split medical data into train, validation, and test sets.

    Args:
        df: Input DataFrame
        text_column: Column containing text
        label_columns: Columns containing labels
        test_size: Proportion of test set
        val_size: Proportion of validation set
        random_state: Random seed
        stratify_on: Column to stratify on (optional)

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    # First split: train+val vs test
    if stratify_on and stratify_on in df.columns:
        stratify_col = df[stratify_on]
    else:
        # For multilabel, use label sum for stratification
        stratify_col = df[label_columns].sum(axis=1)

    train_val_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=stratify_col
    )

    # Second split: train vs val
    if val_size > 0:
        if stratify_on and stratify_on in train_val_df.columns:
            stratify_col_val = train_val_df[stratify_on]
        else:
            stratify_col_val = train_val_df[label_columns].sum(axis=1)

        # Adjust val_size relative to train_val_df
        adjusted_val_size = val_size / (1 - test_size)

        train_df, val_df = train_test_split(
            train_val_df,
            test_size=adjusted_val_size,
            random_state=random_state,
            stratify=stratify_col_val,
        )
    else:
        train_df = train_val_df
        val_df = pd.DataFrame()

    logger.info(
        f"Data split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}"
    )

    return train_df, val_df, test_df


def prepare_multilabel_data(
    df: pd.DataFrame, text_column: str, label_columns: List[str]
) -> Tuple[List[str], np.ndarray, MultiLabelBinarizer]:
    """
    Prepare data for multilabel classification.

    Args:
        df: Input DataFrame
        text_column: Column containing text
        label_columns: Columns containing labels

    Returns:
        Tuple of (texts, labels, label_binarizer)
    """
    # Extract texts
    texts = df[text_column].tolist()

    # Extract and binarize labels
    labels = df[label_columns].values.astype(int)

    # Create MultiLabelBinarizer for consistency
    mlb = MultiLabelBinarizer()
    mlb.fit([label_columns])
    mlb.classes_ = np.array(label_columns)

    logger.info(f"Prepared {len(texts)} samples with {len(label_columns)} labels")
    return texts, labels, mlb


def handle_missing_data(
    df: pd.DataFrame,
    text_columns: List[str],
    label_columns: List[str],
    strategy: str = "drop",
) -> pd.DataFrame:
    """
    Handle missing data in the dataset.

    Args:
        df: Input DataFrame
        text_columns: Text columns to check
        label_columns: Label columns to check
        strategy: Strategy for handling missing data ('drop', 'fill')

    Returns:
        DataFrame with missing data handled
    """
    df_clean = df.copy()

    # Check for missing text data
    text_missing = df_clean[text_columns].isnull().any(axis=1)

    # Check for missing label data
    label_missing = df_clean[label_columns].isnull().any(axis=1)

    if strategy == "drop":
        # Drop rows with missing critical data
        before_count = len(df_clean)
        df_clean = df_clean[~(text_missing | label_missing)]
        after_count = len(df_clean)

        logger.info(f"Dropped {before_count - after_count} rows with missing data")

    elif strategy == "fill":
        # Fill missing text with empty strings
        for col in text_columns:
            df_clean[col] = df_clean[col].fillna("")

        # Fill missing labels with 0
        for col in label_columns:
            df_clean[col] = df_clean[col].fillna(0)

        logger.info("Filled missing data with default values")

    return df_clean


def balance_dataset(
    df: pd.DataFrame,
    label_columns: List[str],
    strategy: str = "undersample",
    target_ratio: float = 0.5,
) -> pd.DataFrame:
    """
    Balance the dataset for multilabel classification.

    Args:
        df: Input DataFrame
        label_columns: Label columns to balance
        strategy: Balancing strategy ('undersample', 'oversample')
        target_ratio: Target ratio for balancing

    Returns:
        Balanced DataFrame
    """
    if strategy == "undersample":
        # Simple undersampling for majority class
        balanced_dfs = []

        for col in label_columns:
            positive_samples = df[df[col] == 1]
            negative_samples = df[df[col] == 0]

            # Calculate target sizes
            pos_count = len(positive_samples)
            target_neg_count = int(pos_count / target_ratio - pos_count)

            if target_neg_count < len(negative_samples):
                negative_samples = negative_samples.sample(
                    n=target_neg_count, random_state=42
                )

            balanced_dfs.append(pd.concat([positive_samples, negative_samples]))

        # Combine and remove duplicates
        df_balanced = pd.concat(balanced_dfs).drop_duplicates()

    else:
        # For now, return original dataset
        df_balanced = df.copy()
        logger.warning(
            f"Strategy '{strategy}' not implemented, returning original data"
        )

    logger.info(f"Balanced dataset: {len(df)} -> {len(df_balanced)} samples")
    return df_balanced


def create_text_statistics(
    df: pd.DataFrame, text_column: str = "text"
) -> Dict[str, Any]:
    """
    Create comprehensive text statistics.

    Args:
        df: DataFrame containing text
        text_column: Column with text data

    Returns:
        Dictionary with text statistics
    """
    texts = df[text_column].dropna()

    # Length statistics
    text_lengths = texts.str.len()
    word_counts = texts.str.split().str.len()

    stats = {
        "total_documents": len(texts),
        "text_length": {
            "mean": text_lengths.mean(),
            "median": text_lengths.median(),
            "std": text_lengths.std(),
            "min": text_lengths.min(),
            "max": text_lengths.max(),
            "percentiles": {
                "25%": text_lengths.quantile(0.25),
                "75%": text_lengths.quantile(0.75),
                "95%": text_lengths.quantile(0.95),
                "99%": text_lengths.quantile(0.99),
            },
        },
        "word_count": {
            "mean": word_counts.mean(),
            "median": word_counts.median(),
            "std": word_counts.std(),
            "min": word_counts.min(),
            "max": word_counts.max(),
        },
    }

    # Vocabulary statistics
    all_words = " ".join(texts).lower().split()
    unique_words = set(all_words)

    stats["vocabulary"] = {
        "total_words": len(all_words),
        "unique_words": len(unique_words),
        "vocabulary_richness": len(unique_words) / len(all_words) if all_words else 0,
    }

    logger.info(f"Generated text statistics for {len(texts)} documents")
    return stats


def validate_data_quality(
    df: pd.DataFrame, text_column: str, label_columns: List[str]
) -> Dict[str, Any]:
    """
    Validate data quality and identify potential issues.

    Args:
        df: DataFrame to validate
        text_column: Text column to check
        label_columns: Label columns to check

    Returns:
        Dictionary with validation results
    """
    issues = {
        "empty_texts": 0,
        "very_short_texts": 0,
        "very_long_texts": 0,
        "duplicate_texts": 0,
        "missing_labels": 0,
        "all_negative_labels": 0,
    }

    # Check text issues
    if text_column in df.columns:
        texts = df[text_column].fillna("")

        issues["empty_texts"] = (texts == "").sum()
        issues["very_short_texts"] = (texts.str.len() < 10).sum()
        issues["very_long_texts"] = (texts.str.len() > 5000).sum()
        issues["duplicate_texts"] = df.duplicated(subset=[text_column]).sum()

    # Check label issues
    valid_label_cols = [col for col in label_columns if col in df.columns]
    if valid_label_cols:
        label_matrix = df[valid_label_cols]

        issues["missing_labels"] = label_matrix.isnull().any(axis=1).sum()
        issues["all_negative_labels"] = (label_matrix.sum(axis=1) == 0).sum()

    # Calculate quality score (higher is better)
    total_samples = len(df)
    if total_samples > 0:
        quality_score = 1.0 - sum(issues.values()) / (total_samples * len(issues))
        issues["quality_score"] = max(0.0, quality_score)
    else:
        issues["quality_score"] = 0.0

    logger.info(
        f"Data quality validation completed. Score: {issues['quality_score']:.3f}"
    )
    return issues
