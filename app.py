from __future__ import annotations

import streamlit as st
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from model import load_artifacts, predict_label
from preprocess import preprocess_dataframe
from utils import DATASET_CSV_PATH, MODEL_PKL_PATH, load_dataset


st.set_page_config(page_title="Email Spam Detection", layout="wide")

# ---------------- Styling (simple, modern, beginner-friendly) ----------------
st.markdown(
    """
<style>
.stApp { background: #ffffff; }
.block-container { padding-top: 1.6rem; padding-bottom: 2rem; }
.app-title { color: #111827; font-size: 2.2rem; font-weight: 800; margin-bottom: 0.25rem; }
.app-subtitle { color: #6b7280; font-size: 1.08rem; margin-top: 0; margin-bottom: 0.2rem; }
.section-heading { color: #111827; font-size: 1.45rem; font-weight: 800; margin-bottom: 0.25rem; }
.helper-text { color: #6b7280; font-size: 1.02rem; margin-top: 0; }
.card {
  background: #ffffff;
  border: 1px solid #e2e8f0;
  border-radius: 14px;
  padding: 16px 18px;
  box-shadow: 0 2px 10px rgba(15, 23, 42, 0.05);
}
.result-box {
  padding: 16px 16px;
  border-radius: 12px;
  border: 1px solid #e2e8f0;
  box-shadow: 0 2px 8px rgba(15, 23, 42, 0.05);
  font-size: 1.08rem;
}
.result-spam { background: #ffe9e7; border-color: #ff8a80; }
.result-ham { background: #eaf9f1; border-color: #a8e6cf; }
.metric-card {
  background: #f9fbff;
  border: 1px solid #e2e8f0;
  border-radius: 12px;
  padding: 14px 10px;
  text-align: center;
  box-shadow: 0 1px 6px rgba(15, 23, 42, 0.04);
}
.metric-label { color: #6b7280; font-size: 1.0rem; }
.metric-value { color: #111827; font-size: 1.5rem; font-weight: 800; }
.soft-divider { height: 1px; background: #e2e8f0; margin: 14px 0 16px 0; }
.muted { color: #6b7280; font-size: 1.02rem; }
.stTextArea label, .stTextInput label, .stSelectbox label { font-size: 1.08rem !important; font-weight: 600; }
.stTextArea textarea { font-size: 1.05rem !important; line-height: 1.6 !important; }
.stButton>button {
  border-radius: 10px;
  border: 1px solid #d1d5db;
  background: #ffffff;
  color: #111827;
  font-size: 1rem;
  font-weight: 600;
  transition: all 0.2s ease;
}
.stButton>button:hover {
  border-color: #4F8EF7;
  box-shadow: 0 4px 10px rgba(79, 142, 247, 0.18);
  transform: translateY(-1px);
}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------- Header Section ----------------
st.markdown("<h1 class='app-title'>📧 Email Spam Detection Dashboard</h1>", unsafe_allow_html=True)
st.markdown(
    "<p class='app-subtitle'>TF‑IDF + Logistic Regression • Fast spam classification with a clean, interactive UI.</p>",
    unsafe_allow_html=True,
)
st.markdown("<div class='soft-divider'></div>", unsafe_allow_html=True)


def _format_label(label: str) -> str:
    """Ensure a consistent label display in the UI."""
    return label.strip()


def set_email_text(value: str) -> None:
    """Callback-safe way to update text_area state."""
    st.session_state.email_text = value


def clear_email_text() -> None:
    """Callback-safe way to clear text_area state."""
    st.session_state.email_text = ""


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
    class_counts = df["label"].value_counts().to_dict()

    return {
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "f1_score": float(f1_score(y_test, preds, zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "ham_count": int(class_counts.get(0, 0)),
        "spam_count": int(class_counts.get(1, 0)),
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


# Session state defaults (must exist before widgets are created)
if "email_text" not in st.session_state:
    st.session_state.email_text = ""
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None


# ---------------- Input + Prediction (side-by-side) ----------------
left_col, right_col = st.columns([1.25, 1], gap="large")

with left_col:
    st.markdown("<div class='section-heading'>📩 Input</div>", unsafe_allow_html=True)
    st.markdown(
        "<p class='helper-text'>Paste an email/SMS message, then click <b>Predict</b>. "
        "Use examples below if you are testing quickly.</p>",
        unsafe_allow_html=True,
    )
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    input_text = st.text_area(
        "Message",
        height=190,
        placeholder="Example: Congratulations! You have won a prize. Click the link to claim now...",
        disabled=artifacts is None,
        key="email_text",
        help="Enter the text you want to classify.",
        label_visibility="collapsed",
    )
    btn_cols = st.columns([1, 1, 2], gap="small")
    with btn_cols[0]:
        predict_clicked = st.button(
            "🔍 Predict",
            use_container_width=True,
            disabled=artifacts is None,
            help="Run spam detection on current input.",
        )
    with btn_cols[1]:
        st.button(
            "🧹 Clear",
            use_container_width=True,
            disabled=artifacts is None,
            on_click=clear_email_text,
            help="Clear the input box.",
        )
    st.markdown("</div>", unsafe_allow_html=True)

with right_col:
    st.markdown("<div class='section-heading'>⚠️ Prediction Result</div>", unsafe_allow_html=True)
    st.markdown(
        "<p class='helper-text'>This panel shows the latest prediction and spam probability.</p>",
        unsafe_allow_html=True,
    )

    if predict_clicked:
        if not input_text.strip():
            st.warning("Please enter some email text first.")
        else:
            with st.spinner("Running prediction..."):
                label, prob_spam = predict_label(artifacts, input_text)
            st.session_state.last_prediction = {"label": _format_label(label), "prob_spam": prob_spam}

    if st.session_state.last_prediction is None:
        st.markdown(
            """
<div class='card'>
  <p class='muted'>No prediction yet. Enter text on the left and click <b>Predict</b>.</p>
</div>
""",
            unsafe_allow_html=True,
        )
    else:
        pred = st.session_state.last_prediction
        label = pred["label"]
        prob_spam = pred["prob_spam"]
        prob_pct = prob_spam * 100.0

        if label == "Spam":
            st.markdown(
                f"""
<div class='result-box result-spam'>
  <div><b>⚠️ Result:</b> <span style='font-size: 1.08rem;'>Spam</span></div>
  <div class='muted'>Spam probability: <b>{prob_pct:.2f}%</b></div>
</div>
""",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
<div class='result-box result-ham'>
  <div><b>✅ Result:</b> <span style='font-size: 1.08rem;'>Not Spam</span></div>
  <div class='muted'>Spam probability: <b>{prob_pct:.2f}%</b></div>
</div>
""",
                unsafe_allow_html=True,
            )
        st.progress(min(max(prob_spam, 0.0), 1.0))


# ---------------- Metrics Section ----------------
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<div class='soft-divider'></div>", unsafe_allow_html=True)
st.markdown("<div class='section-heading'>📊 Metrics</div>", unsafe_allow_html=True)
st.markdown("<p class='helper-text'>Computed on an 80/20 test split using the saved model (no retraining).</p>", unsafe_allow_html=True)

metrics = get_metrics_for_display()
if artifacts is None:
    st.info("Train the model once using `python3 main.py`, then reload this app.")
elif metrics is None:
    st.warning("Could not compute metrics yet.")
else:
    # Use HTML/CSS cards for a clean, modern metrics display.
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"<div class='metric-card'><div class='metric-label'>✅ Accuracy</div><div class='metric-value'>{metrics['accuracy']:.4f}</div></div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"<div class='metric-card'><div class='metric-label'>🎯 Precision</div><div class='metric-value'>{metrics['precision']:.4f}</div></div>",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"<div class='metric-card'><div class='metric-label'>📥 Recall</div><div class='metric-value'>{metrics['recall']:.4f}</div></div>",
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"<div class='metric-card'><div class='metric-label'>⭐ F1-score</div><div class='metric-value'>{metrics['f1_score']:.4f}</div></div>",
            unsafe_allow_html=True,
        )

    # ---------------- Visualization Section ----------------
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-heading'>📈 Visualizations</div>", unsafe_allow_html=True)
    st.markdown("<p class='helper-text'>Compact visuals for model performance and data distribution.</p>", unsafe_allow_html=True)
    cm = metrics["confusion_matrix"]  # [[TN, FP], [FN, TP]]

    # Show confusion matrix + performance bar chart + class distribution pie chart side-by-side.
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        viz_col1, viz_col2, viz_col3 = st.columns(3, gap="medium")

        with viz_col1:
            fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap=sns.light_palette("#4F8EF7", as_cmap=True),
                cbar=False,
                xticklabels=["Not Spam", "Spam"],
                yticklabels=["Not Spam", "Spam"],
                ax=ax_cm,
            )
            ax_cm.set_xlabel("Predicted", fontsize=10)
            ax_cm.set_ylabel("Actual", fontsize=10)
            ax_cm.set_title("Confusion Matrix", fontsize=12, fontweight="bold")
            st.pyplot(fig_cm, clear_figure=True)

        with viz_col2:
            metric_names = ["Accuracy", "Precision", "Recall", "F1-score"]
            metric_vals = [
                metrics["accuracy"],
                metrics["precision"],
                metrics["recall"],
                metrics["f1_score"],
            ]
            bar_colors = ["#4F8EF7", "#7FB3FF", "#A8E6CF", "#FFB3AB"]
            fig_bar, ax_bar = plt.subplots(figsize=(5, 4))
            ax_bar.bar(metric_names, metric_vals, color=bar_colors, edgecolor="#d1d5db")
            ax_bar.set_ylim(0, 1.0)
            ax_bar.set_title("Metric Scores", fontsize=12, fontweight="bold")
            ax_bar.set_ylabel("Score")
            ax_bar.tick_params(axis="x", labelrotation=15, labelsize=9)
            ax_bar.grid(axis="y", linestyle="--", alpha=0.25)
            st.pyplot(fig_bar, clear_figure=True)

        with viz_col3:
            sizes = [metrics["ham_count"], metrics["spam_count"]]
            labels = ["Not Spam", "Spam"]
            colors = ["#A8E6CF", "#FF8A80"]
            fig_pie, ax_pie = plt.subplots(figsize=(5, 4))
            ax_pie.pie(
                sizes,
                labels=labels,
                autopct="%1.1f%%",
                startangle=90,
                colors=colors,
                textprops={"fontsize": 10},
            )
            ax_pie.set_title("Dataset Distribution", fontsize=12, fontweight="bold")
            ax_pie.axis("equal")
            st.pyplot(fig_pie, clear_figure=True)
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
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<div class='soft-divider'></div>", unsafe_allow_html=True)
st.markdown("<div class='section-heading'>🧪 Examples (click to try)</div>", unsafe_allow_html=True)
st.markdown("<p class='helper-text'>Choose an example to auto-fill the input, then click <b>Predict</b>.</p>", unsafe_allow_html=True)

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
        st.button(
            f"Try spam #{i}",
            key=f"spam_{i}",
            use_container_width=True,
            disabled=artifacts is None,
            on_click=set_email_text,
            args=(text,),
        )
with col_b:
    st.markdown("**Not spam examples**")
    for i, text in enumerate(example_ham, start=1):
        st.button(
            f"Try not spam #{i}",
            key=f"ham_{i}",
            use_container_width=True,
            disabled=artifacts is None,
            on_click=set_email_text,
            args=(text,),
        )

