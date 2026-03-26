from __future__ import annotations

import argparse
from pathlib import Path

from sklearn.model_selection import train_test_split

from model import load_artifacts, predict_label, save_artifacts, train_logistic_regression
from preprocess import preprocess_dataframe
from utils import DATASET_CSV_PATH, MODEL_PKL_PATH, get_label_distribution, load_dataset


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Logistic Regression for Email Spam Detection.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DATASET_CSV_PATH,
        help="Path to dataset CSV with columns: label, text",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=MODEL_PKL_PATH,
        help="Where to save the trained model artifacts (pickle file).",
    )
    parser.add_argument(
        "--predict",
        type=str,
        default=None,
        help="Optional: run a prediction on the provided text after training.",
    )
    parser.add_argument(
        "--skip-testing",
        action="store_true",
        help="Skip the dataset/custom-email demo predictions printed in the terminal.",
    )
    return parser


def format_spam_label(value: int) -> str:
    """Convert numeric label (1/0) into a human-friendly label string."""
    return "Spam" if value == 1 else "Not Spam"


def structured_testing_and_output(artifacts, df, *, test_size: float = 0.2, random_state: int = 42) -> None:
    """
    Beginner-friendly testing output:
    1) Predict on a handful of real dataset messages from the test split and compare with the true label.
    2) Predict on a curated list of custom emails (with an 'expected' label for easy verification).
    """
    print("\n" + "=" * 80)
    print("Testing / Demo Predictions (terminal output)")
    print("=" * 80)

    # 1) Dataset samples from the same 80-20 split used for metrics
    df_train, df_test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["label"],
    )

    X_test_clean = df_test["clean_text"]
    y_test = df_test["label"]

    X_test_vec = artifacts.vectorizer.transform(X_test_clean)
    preds = artifacts.model.predict(X_test_vec)
    prob_spam = artifacts.model.predict_proba(X_test_vec)[:, 1]

    # Choose up to 8 samples while ensuring we show both classes (Spam and Not Spam)
    max_samples = 8
    spam_indices = [i for i, v in enumerate(y_test) if v == 1]
    ham_indices = [i for i, v in enumerate(y_test) if v == 0]

    selected_indices: list[int] = []
    for idx_list in (spam_indices, ham_indices):
        for idx in idx_list:
            if len(selected_indices) >= max_samples:
                break
            selected_indices.append(idx)
        if len(selected_indices) >= max_samples:
            break

    # If we couldn't get both classes due to a very small split, fill the rest
    if len(selected_indices) < max_samples:
        remaining = [i for i in range(len(y_test)) if i not in set(selected_indices)]
        selected_indices.extend(remaining[: max_samples - len(selected_indices)])

    correct = 0
    print("\n1) Dataset test samples (true label vs predicted)")
    for i, idx in enumerate(selected_indices, start=1):
        true_label = y_test.iloc[idx]
        predicted_label = preds[idx]
        is_correct = true_label == predicted_label
        correct += 1 if is_correct else 0

        text = str(df_test.iloc[idx]["text"]).strip()
        text_preview = text if len(text) <= 280 else f"{text[:277]}..."

        print("-" * 80)
        print(f"Sample {i}")
        print(f"  True:      {format_spam_label(true_label)}")
        print(f"  Predicted: {format_spam_label(predicted_label)}")
        print(f"  Spam probability: {prob_spam[idx]:.4f}")
        print(f"  Match:     {'YES' if is_correct else 'NO'}")
        print(f"  Text: {text_preview}")

    print("\nDataset sample verification:")
    print(f"  Correct predictions: {correct}/{len(selected_indices)}")

    # 2) Custom input emails (easy verification for demos)
    print("\n2) Custom email examples (expected vs predicted)")
    custom_emails = [
        # Spam-like examples
        {"text": "Congratulations! You have been selected for a free reward. Click to claim now.", "expected": "Spam"},
        {"text": "Urgent: your account will be closed unless you verify your identity immediately.", "expected": "Spam"},
        {"text": "Win money now!!! Limited time offer. Act today and get your prize.", "expected": "Spam"},
        {"text": "You've received a special promotion. Buy now and save big on your next purchase.", "expected": "Spam"},
        {"text": "Final notice: claim your discount code today before it expires.", "expected": "Spam"},
        # Ham-like examples
        {"text": "Hi, are we still meeting for lunch tomorrow? Let me know what time works for you.", "expected": "Not Spam"},
        {"text": "Please find the report attached for tomorrow. Thanks for reviewing it.", "expected": "Not Spam"},
        {"text": "Could you call me when you have a moment about the project timeline?", "expected": "Not Spam"},
        {"text": "Thanks for your email. I will get back to you after the meeting.", "expected": "Not Spam"},
        {"text": "Reminder: complete the timesheet by Friday and send any updates in the thread.", "expected": "Not Spam"},
    ]

    passed = 0
    for i, item in enumerate(custom_emails, start=1):
        predicted, prob = predict_label(artifacts, item["text"])
        expected = item["expected"]
        ok = predicted == expected
        passed += 1 if ok else 0

        print("-" * 80)
        print(f"Email {i}")
        print(f"  Expected:  {expected}")
        print(f"  Predicted: {predicted}")
        print(f"  Spam probability: {prob:.4f}")
        print(f"  Match:     {'YES' if ok else 'NO'}")
        print(f"  Text: {item['text']}")

    print("\nCustom email verification:")
    print(f"  Correct predictions: {passed}/{len(custom_emails)}")
    print("=" * 80 + "\n")


def main() -> None:
    args = build_arg_parser().parse_args()

    print("Loading dataset...")
    df = load_dataset(args.dataset)
    print(f"Rows loaded: {len(df)}")
    print(f"Dataset columns found: {list(df.columns)}")
    print(f"Class distribution (raw): {get_label_distribution(df)}")

    print("Preprocessing text (lowercase, regex cleaning, stopwords, tokenization)...")
    df = preprocess_dataframe(df)
    print(f"Rows after preprocessing: {len(df)}")
    print(f"Class distribution (after preprocessing): {get_label_distribution(df)}")

    print("Training TF-IDF + Logistic Regression (80-20 split, optimized TF-IDF + balanced classes)...")
    artifacts, metrics = train_logistic_regression(
        df,
        test_size=0.2,
        random_state=42,
        tfidf_max_features=20000,
        tfidf_ngram_range=(1, 2),
        tfidf_min_df=2,
        C=4.0,
    )

    print("\nEvaluation metrics (test split):")
    for k in ["accuracy", "precision", "recall", "f1_score"]:
        print(f"  {k}: {metrics[k]:.4f}")
    cm = metrics["confusion_matrix"]  # [[TN, FP], [FN, TP]]
    print("  confusion matrix:")
    print(f"    TN: {cm[0][0]}  FP: {cm[0][1]}")
    print(f"    FN: {cm[1][0]}  TP: {cm[1][1]}")

    print(f"\nSaving model to: {args.model_path}")
    save_artifacts(artifacts, args.model_path)

    # Beginner-friendly output: run automatic testing/demos after training.
    # You can skip it with `--skip-testing` if you only want raw metrics + the pickle file.
    if not args.skip_testing:
        structured_testing_and_output(artifacts, df, test_size=0.2, random_state=42)

    if args.predict is not None:
        # Demonstrate the prediction function directly.
        label, prob = predict_label(artifacts, args.predict)
        print("\nPrediction for custom input:")
        print(f"  label: {label}")
        print(f"  spam_probability: {prob:.4f}")


if __name__ == "__main__":
    main()

