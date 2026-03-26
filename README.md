# Email Spam Detection (Logistic Regression) - Beginner Project (macOS + Python 3)

This project trains a **Logistic Regression** model using **TF-IDF** features to classify emails as:
- `Spam` (label `1`)
- `Not Spam` (label `0`)

It includes:
- CLI training + evaluation: `python3 main.py`
- Streamlit UI: `streamlit run app.py`
- NLTK-based stopword removal (with safe fallbacks if NLTK data isn’t installed)
- Model persistence with `pickle` (save/load)

## 1) Setup on macOS (create/activate a virtual environment)

From the project folder (`Email-Spam-Detect`):

```bash
python3 -m venv spam_env
source spam_env/bin/activate
pip3 install -r requirements.txt
```

After this, always run the project commands **inside** the activated environment to avoid `ModuleNotFoundError`.

## 2) Check Python version (optional)

```bash
python3 --version
pip3 --version
```

## 3) Train + evaluate from the command line

```bash
python3 main.py
```

When you run `python3 main.py`, the program will:
- load `dataset/spam.csv`
- preprocess the text (lowercase, regex cleaning, stopwords, tokenization)
- train TF-IDF + Logistic Regression (80-20 split)
- print evaluation metrics (accuracy, precision, recall, F1-score, confusion matrix)
- automatically run a beginner-friendly demo/testing output in the terminal:
  - predictions for several real dataset samples (true vs predicted)
  - predictions for 10 custom example emails (expected vs predicted)

To skip the demo/testing output (only show training metrics + save the model), run:

```bash
python3 main.py --skip-testing
```

You can also run a single custom prediction from the command line:

```bash
python3 main.py --predict "Congratulations, you have won a prize!"
```

Trained model artifacts are saved to:
- `models/spam_logreg.pkl`

## 4) Run the Streamlit interface

```bash
streamlit run app.py
```

The app will:
- load the trained model artifacts from `models/spam_logreg.pkl` (it does not retrain every time)
- show a text box + **Predict** button (prediction + spam probability)
- display model evaluation metrics (accuracy, precision, recall, F1-score) and a confusion matrix (test split)

If you see a message that the model is missing, run `python3 main.py` once, then restart:
`streamlit run app.py`

## 5) Dataset

The project uses a small starter dataset shipped with the repo:
- `dataset/spam.csv`

It has exactly two columns:
- `label` (values: `spam` or `ham`)
- `text` (email message content)

If you replace `dataset/spam.csv`, keep the same column names (`label`, `text`).

## 6) Notes about NLTK downloads

The code tries to use NLTK stopwords and tokenization. If NLTK data files are missing, it will:
- attempt to download them, and if that fails,
- fall back to a small built-in stopword list + whitespace tokenization.

