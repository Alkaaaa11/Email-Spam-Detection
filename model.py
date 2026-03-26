from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import pickle

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from preprocess import preprocess_text


@dataclass(frozen=True)
class ModelArtifacts:
    """What we need to run predictions later."""

    vectorizer: TfidfVectorizer
    model: LogisticRegression
    threshold: float = 0.5


def build_tfidf_vectorizer(
    *,
    max_features: int = 20000,
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 2,
) -> TfidfVectorizer:
    """
    TF-IDF turns tokenized text into numeric features.

    - TfidfVectorizer computes TF-IDF weights so the model focuses on words
      that are frequent in a particular class (spam/ham) rather than just frequent overall.
    """
    return TfidfVectorizer(
        # Increasing max_features + adding bigrams often improves performance for spam detection.
        max_features=max_features,
        ngram_range=ngram_range,
        # We already pass a cleaned/tokenized string; default tokenization works fine.
        token_pattern=r"(?u)\b\w+\b",
        min_df=min_df,
        sublinear_tf=True,
    )


def train_logistic_regression(
    df: pd.DataFrame,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    tfidf_max_features: int = 20000,
    tfidf_ngram_range: tuple[int, int] = (1, 2),
    tfidf_min_df: int = 2,
    C: float = 4.0,
) -> tuple[ModelArtifacts, dict[str, Any]]:
    """
    Train + evaluate a Logistic Regression classifier.

    Uses an 80-20 train-test split (configurable via test_size).
    """
    if "clean_text" not in df.columns:
        raise ValueError("Expected a 'clean_text' column. Run preprocess_dataframe() first.")
    if "label" not in df.columns:
        raise ValueError("Expected a 'label' column.")

    X = df["clean_text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    vectorizer = build_tfidf_vectorizer(
        max_features=tfidf_max_features,
        ngram_range=tfidf_ngram_range,
        min_df=tfidf_min_df,
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Logistic Regression is a good baseline for text classification with TF-IDF features.
    # class_weight='balanced' helps when the dataset has more ham than spam.
    model = LogisticRegression(max_iter=4000, C=C, class_weight="balanced", solver="liblinear")
    model.fit(X_train_vec, y_train)

    preds = model.predict(X_test_vec)
    probs = model.predict_proba(X_test_vec)[:, 1]

    # Probability threshold is 0.5 by default.
    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1_score": f1_score(y_test, preds, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
    }

    artifacts = ModelArtifacts(vectorizer=vectorizer, model=model, threshold=0.5)
    return artifacts, metrics


def predict_label(
    artifacts: ModelArtifacts,
    input_text: str,
) -> tuple[str, float]:
    """
    Predict a human-friendly label from a custom input message.

    Returns:
      - "Spam" or "Not Spam"
      - probability of being spam (class 1)
    """
    cleaned = preprocess_text(input_text)
    if cleaned.strip() == "":
        # If the input is empty after preprocessing, we can't extract features reliably.
        # Default to "Not Spam" with probability 0.0 (safe/explicit behavior).
        return "Not Spam", 0.0

    vec = artifacts.vectorizer.transform([cleaned])
    prob_spam = float(artifacts.model.predict_proba(vec)[:, 1][0])
    label = "Spam" if prob_spam >= artifacts.threshold else "Not Spam"
    return label, prob_spam


def save_artifacts(artifacts: ModelArtifacts, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(
            {
                "vectorizer": artifacts.vectorizer,
                "model": artifacts.model,
                "threshold": artifacts.threshold,
            },
            f,
        )


def load_artifacts(path: str | Path) -> ModelArtifacts:
    path = Path(path)
    with path.open("rb") as f:
        data = pickle.load(f)
    return ModelArtifacts(
        vectorizer=data["vectorizer"],
        model=data["model"],
        threshold=float(data.get("threshold", 0.5)),
    )

