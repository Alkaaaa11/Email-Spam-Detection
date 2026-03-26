from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd

# Project paths (resolved relative to this file, so it works from anywhere)
BASE_DIR = Path(__file__).resolve().parent
DATASET_CSV_PATH = BASE_DIR / "dataset" / "spam_combined.csv"
SMALL_DATASET_CSV_PATH = BASE_DIR / "dataset" / "spam.csv"
SMS_SPAM_COLLECTION_CSV_PATH = BASE_DIR / "dataset" / "sms_spam_collection.csv"
MODEL_PKL_PATH = BASE_DIR / "models" / "spam_logreg.pkl"


def normalize_label(label: object) -> int:
    """
    Convert dataset labels to binary values:
      - spam -> 1
      - ham  -> 0
    """
    if pd.isna(label):
        raise ValueError("Label is missing/NaN.")

    s = str(label).strip().lower()
    if s == "spam":
        return 1
    if s == "ham":
        return 0
    raise ValueError(f"Unexpected label value: {label!r}. Expected 'spam' or 'ham'.")


def load_dataset(csv_path: Union[str, Path] = DATASET_CSV_PATH) -> pd.DataFrame:
    """
    Load and clean the raw dataset.

    Expected CSV columns:
      - label: 'spam' or 'ham'
      - text: message content
    """
    df = pd.read_csv(csv_path)
    if missing_cols := [c for c in ["label", "text"] if c not in df.columns]:
        raise ValueError(f"Dataset is missing columns: {missing_cols}. Found: {list(df.columns)}")

    # Keep only required columns (helps if CSV contains extra fields)
    df = df[["label", "text"]].copy()

    # Handle missing values early
    df = df.dropna(subset=["label", "text"])
    df["label"] = df["label"].apply(normalize_label)
    df["text"] = df["text"].astype(str).str.strip()

    # Remove empty messages (after trimming)
    df = df[df["text"] != ""].reset_index(drop=True)
    return df


def get_label_distribution(df: pd.DataFrame) -> dict[str, int]:
    """
    Return a readable class distribution for quick reporting.
    (1=spam, 0=ham)
    """
    counts = df["label"].value_counts().to_dict()
    return {
        "ham(0)": counts.get(0, 0),
        "spam(1)": counts.get(1, 0),
        "total": len(df),
    }

