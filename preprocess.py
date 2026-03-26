from __future__ import annotations

import re
from typing import Iterable, Set

import pandas as pd

# NLTK is used for stopword removal + (optional) tokenization.
import nltk

from nltk.corpus import stopwords as nltk_stopwords
from nltk.tokenize import word_tokenize


# Small fallback list in case NLTK data (stopwords/punkt) isn't available.
# This keeps the project runnable even when NLTK downloads fail.
FALLBACK_STOPWORDS: Set[str] = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "if",
    "while",
    "with",
    "to",
    "of",
    "in",
    "on",
    "for",
    "at",
    "by",
    "from",
    "is",
    "it",
    "this",
    "that",
    "these",
    "those",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "not",
    "no",
    "yes",
    "you",
    "i",
    "we",
    "they",
    "he",
    "she",
    "them",
    "my",
    "your",
    "our",
    "their",
    "me",
    "him",
    "her",
}


def get_stop_words() -> Set[str]:
    """Return English stopwords (with safe fallback)."""
    try:
        return set(nltk_stopwords.words("english"))
    except LookupError:
        # If NLTK resources are missing, try downloading once.
        try:
            nltk.download("stopwords", quiet=True)
            return set(nltk_stopwords.words("english"))
        except Exception:
            return FALLBACK_STOPWORDS


def tokenize(text: str) -> list[str]:
    """
    Tokenize a cleaned string.

    Uses `nltk.word_tokenize` when available; otherwise falls back to splitting by whitespace.
    """
    try:
        return word_tokenize(text)
    except LookupError:
        try:
            nltk.download("punkt", quiet=True)
            return word_tokenize(text)
        except Exception:
            return text.split()


def clean_text(text: str) -> str:
    """
    Basic normalization:
      1) lowercase
      2) remove numbers
      3) remove punctuation/symbols (keep only letters + whitespace)
      4) collapse repeated spaces
    """
    text = text.lower()
    text = re.sub(r"\d+", " ", text)  # remove digits
    text = re.sub(r"[^a-zA-Z\s]", " ", text)  # remove punctuation/symbols
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_text(text: object, *, stop_words: Set[str] | None = None) -> str:
    """
    Convert a raw email message to a cleaned/tokenized string suitable for TF-IDF.
    """
    if pd.isna(text):
        return ""

    stop_words = stop_words or get_stop_words()
    text_str = str(text).strip()
    if not text_str:
        return ""

    normalized = clean_text(text_str)
    if not normalized:
        return ""

    tokens = tokenize(normalized)

    # Remove stopwords + very short tokens.
    tokens = [t for t in tokens if t.lower() not in stop_words and len(t) > 1]
    return " ".join(tokens)


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a `clean_text` column that is used for vectorization.
    Removes rows that become empty after preprocessing.
    """
    df = df.copy()
    stop_words = get_stop_words()
    df["clean_text"] = df["text"].apply(lambda x: preprocess_text(x, stop_words=stop_words))
    df = df[df["clean_text"].str.strip() != ""].reset_index(drop=True)
    return df

