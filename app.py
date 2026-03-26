from __future__ import annotations

import streamlit as st
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from model import load_artifacts, predict_label
from preprocess import preprocess_dataframe
from utils import DATASET_CSV_PATH, MODEL_PKL_PATH, load_dataset


st.set_page_config(page_title="Email Spam Detection", layout="centered")

# ---------------- Styling (simple, modern, beginner-friendly) ----------------
st.markdown(
    """
<style>
.app-subtitle { color: #6b7280; margin-top: -10px; }
.section-title { margin-top: 10px; }
.result-box { padding: 14px 16px; border-radius: 12px; border: 1px solid rgba(0,0,0,0.08); }
.result-spam { background: rgba(239, 68, 68, 0.10); }
.result-ham { background: rgba(34, 197, 94, 0.10); }
.muted { color: #6b7280; }
.small { font-size: 0.95rem; }
</style>
""",
    unsafe_allow_html=True,
)


# ---------------- Header Section ----------------
st.markdown("## 📧 Email Spam Detection")
st.markdown(
    "<div class='app-subtitle small'>TF‑IDF + Logistic Regression • Enter a message and get a prediction instantly.</div>",
    unsafe_allow_html=True,
)


def _format_label(label: str) -> str:
    """Ensure a consistent label display in the UI."""
    return label.strip()


@st.cache_resource(show_spinner=False)
def get_artifacts():
    """
    Load the trained model artifacts (TF-IDF vectorizer + Logistic Regression).

    This app intentionally does NOT retrain every time.
    You should run `python3 main.py` once to create `models/spam_logreg.pkl`.
    """
    if not MODEL_PKL_PATH.exists():
        st.error(
            "Trained model not found.\n\n"
            "Step 1 (train once):\n"
            "`python3 main.py`\n\n"
            "Step 2 (run UI):\n"
            "`streamlit run app.py`"
        )
        return None
    return load_artifacts(MODEL_PKL_PATH)


def compute_evaluation_metrics(artifacts) -> dict[str, object]:
    """
    Compute evaluation metrics for display in the UI.

    Note: this does not retrain the model.
    It uses the saved vectorizer + model to predict on an 80-20 test split.
    """
    df = load_dataset(DATASET_CSV_PATH)
    df = preprocess_dataframe(df)

    _df_train, df_test = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    X_test_clean = df_test["clean_text"]
    y_test = df_test["label"]

    X_test_vec = artifacts.vectorizer.transform(X_test_clean)
    preds = artifacts.model.predict(X_test_vec)

    cm = confusion_matrix(y_test, preds, labels=[0, 1])  # [[TN, FP], [FN, TP]]

    return {
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "f1_score": float(f1_score(y_test, preds, zero_division=0)),
        "confusion_matrix": cm.tolist(),
    }


@st.cache_data(show_spinner=False)
def get_metrics_for_display():
    """
    Cached metrics for display.

    We avoid passing the (unhashable) model object into Streamlit's cache;
    instead we reload from the saved pickle path.
    """
    if not MODEL_PKL_PATH.exists():
        return None
    artifacts_local = load_artifacts(MODEL_PKL_PATH)
    return compute_evaluation_metrics(artifacts_local)


# Load artifacts once (cached) so prediction is fast.
artifacts = get_artifacts()


# ---------------- Input Section ----------------
st.markdown("### ✍️ Input")
st.markdown("<div class='muted small'>Tip: Try the examples below, or paste any email/SMS text.</div>", unsafe_allow_html=True)

if "email_text" not in st.session_state:
    st.session_state.email_text = ""

input_text = st.text_area(
    "Email text",
    height=150,
    placeholder="Example: Congratulations! You have won a prize. Click the link to claim now...",
    disabled=artifacts is None,
    key="email_text",
)

cols = st.columns([1, 1, 2])
with cols[0]:
    predict_clicked = st.button("🔍 Predict", use_container_width=True, disabled=artifacts is None)
with cols[1]:
    clear_clicked = st.button("🧹 Clear", use_container_width=True, disabled=artifacts is None)
if clear_clicked:
    st.session_state.email_text = ""

# ---------------- Prediction Section ----------------
st.markdown("### ✅ Prediction")

if predict_clicked:
    if not input_text.strip():
        st.warning("Please enter some email text first.")
    else:
        with st.spinner("Running prediction..."):
            label, prob_spam = predict_label(artifacts, input_text)

        label = _format_label(label)
        prob_pct = prob_spam * 100.0

        if label == "Spam":
            st.markdown(
                f"""
<div class='result-box result-spam'>
  <div><b>🚨 Result:</b> <span style='font-size: 1.05rem;'>Spam</span></div>
  <div class='muted small'>Spam probability: <b>{prob_pct:.2f}%</b></div>
</div>
""",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
<div class='result-box result-ham'>
  <div><b>✅ Result:</b> <span style='font-size: 1.05rem;'>Not Spam</span></div>
  <div class='muted small'>Spam probability: <b>{prob_pct:.2f}%</b></div>
</div>
""",
                unsafe_allow_html=True,
            )

        st.progress(min(max(prob_spam, 0.0), 1.0))


# ---------------- Metrics Section ----------------
st.divider()
st.markdown("### 📊 Metrics")
st.markdown("<div class='muted small'>Computed on an 80/20 test split using the saved model (no retraining).</div>", unsafe_allow_html=True)

metrics = get_metrics_for_display()
if artifacts is None:
    st.info("Train the model once using `python3 main.py`, then reload this app.")
elif metrics is None:
    st.warning("Could not compute metrics yet.")
else:
    # Use Streamlit "cards" for a clean, beginner-friendly summary
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("✅ Accuracy", f"{metrics['accuracy']:.4f}")
    c2.metric("🎯 Precision", f"{metrics['precision']:.4f}")
    c3.metric("📥 Recall", f"{metrics['recall']:.4f}")
    c4.metric("⭐ F1-score", f"{metrics['f1_score']:.4f}")

    # ---------------- Visualization Section ----------------
    st.markdown("### 🧩 Confusion Matrix")
    cm = metrics["confusion_matrix"]  # [[TN, FP], [FN, TP]]

    # Prefer seaborn heatmap; fall back to matplotlib/table.
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(figsize=(5.2, 4.2))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Not Spam (0)", "Spam (1)"],
            yticklabels=["Not Spam (0)", "Spam (1)"],
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        st.pyplot(fig, clear_figure=True)
    except Exception:
        # Beginner-friendly fallback (always works)
        st.table(
            {
                "": ["Pred Not Spam (0)", "Pred Spam (1)"],
                "True Not Spam (0)": [cm[0][0], cm[0][1]],
                "True Spam (1)": [cm[1][0], cm[1][1]],
            }
        )


# ---------------- Example Section ----------------
st.divider()
st.markdown("### 🧪 Examples (click to try)")
st.markdown("<div class='muted small'>Click an example to fill the input box, then press <b>Predict</b>.</div>", unsafe_allow_html=True)

example_spam = [
    "🚨 Urgent! Your account will be suspended. Verify now to avoid closure.",
    "🎁 Congratulations! You've won a FREE gift card. Click to claim your prize.",
    "💰 Earn money fast from home. Limited time offer, sign up today!",
]
example_ham = [
    "✅ Hi, are we still meeting tomorrow at 11am? Let me know.",
    "📎 Please review the attached report and share your feedback.",
    "Thanks! I’ll send the updated document after the meeting.",
]

col_a, col_b = st.columns(2)
with col_a:
    st.markdown("**Spam examples**")
    for i, text in enumerate(example_spam, start=1):
        if st.button(f"Try spam #{i}", key=f"spam_{i}", use_container_width=True, disabled=artifacts is None):
            st.session_state.email_text = text
with col_b:
    st.markdown("**Not spam examples**")
    for i, text in enumerate(example_ham, start=1):
        if st.button(f"Try not spam #{i}", key=f"ham_{i}", use_container_width=True, disabled=artifacts is None):
            st.session_state.email_text = text

