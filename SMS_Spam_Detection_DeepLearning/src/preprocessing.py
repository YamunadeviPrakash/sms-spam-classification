"""
preprocessing.py

Reusable functions for loading, cleaning, and preparing the
UCI SMS Spam Collection dataset for the LSTM spam classifier.
"""

import os
import re
import pandas as pd


def load_dataset(dataset_path):
    """
    Load the UCI SMS Spam Collection dataset.

    The original UCI file is tab-separated with two columns and NO header:
        label \t message

    Returns
    -------
    pd.DataFrame with columns ["label", "message"]
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Dataset not found at '{dataset_path}'. "
            f"Please make sure the SMSSpamCollection file exists in the dataset/ folder."
        )

    df = pd.read_csv(
        dataset_path,
        sep="\t",
        header=None,
        names=["label", "message"],
        encoding="utf-8",
    )

    if df.empty:
        raise ValueError("The dataset file is empty.")

    if "label" not in df.columns or "message" not in df.columns:
        raise ValueError("Dataset does not have the expected 'label' and 'message' columns.")

    return df


def clean_dataframe(df):
    """
    Handle missing values and duplicate rows.

    Returns
    -------
    cleaned pd.DataFrame, stats dict (missing_values, duplicate_count)
    """
    stats = {}

    stats["missing_values"] = int(df.isnull().sum().sum())
    df = df.dropna(subset=["label", "message"]).reset_index(drop=True)

    stats["duplicate_count"] = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)

    # Keep only valid labels
    df = df[df["label"].isin(["ham", "spam"])].reset_index(drop=True)

    return df, stats


def clean_text(text):
    """
    Basic, gentle text cleaning for SMS spam detection.

    IMPORTANT: We deliberately do NOT strip out numbers, URLs, currency
    symbols, or promotional words, since these are strong spam indicators
    that the LSTM model should be allowed to learn from.
    """
    text = str(text)

    # Lowercase
    text = text.lower()

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove characters that are not letters, numbers, common punctuation,
    # currency symbols, or whitespace. This keeps URLs, £/$ signs, numbers.
    text = re.sub(r"[^a-z0-9\s£$%!?.,:/@#&*+-]", " ", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text


def encode_labels(df):
    """
    Convert labels: ham -> 0, spam -> 1
    """
    df = df.copy()
    df["label_encoded"] = df["label"].map({"ham": 0, "spam": 1})
    return df


def preprocess_pipeline(dataset_path):
    """
    Full preprocessing pipeline: load -> clean -> encode.

    Returns
    -------
    df : pd.DataFrame with columns [label, message, clean_message, label_encoded]
    stats : dict of dataset statistics
    """
    raw_df = load_dataset(dataset_path)
    total_raw = len(raw_df)

    df, clean_stats = clean_dataframe(raw_df)

    df["clean_message"] = df["message"].apply(clean_text)
    df = encode_labels(df)

    # Remove any rows that became empty after cleaning
    df = df[df["clean_message"].str.len() > 0].reset_index(drop=True)

    total_messages = len(df)
    spam_count = int((df["label_encoded"] == 1).sum())
    ham_count = int((df["label_encoded"] == 0).sum())

    stats = {
        "total_raw_messages": total_raw,
        "total_messages": total_messages,
        "spam_count": spam_count,
        "ham_count": ham_count,
        "spam_percentage": round(100 * spam_count / total_messages, 2) if total_messages else 0,
        "ham_percentage": round(100 * ham_count / total_messages, 2) if total_messages else 0,
        "missing_values": clean_stats["missing_values"],
        "duplicate_count": clean_stats["duplicate_count"],
    }

    return df, stats
