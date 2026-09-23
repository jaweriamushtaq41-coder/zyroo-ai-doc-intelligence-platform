"""
Week 3 - Train, compare and evaluate document classifiers.
Run:  python train_model.py
Creates: doc_classifier.joblib, evaluation_report.txt
"""
import os
import re
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "document_dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "doc_classifier.joblib")
REPORT_PATH = os.path.join(BASE_DIR, "evaluation_report.txt")

RESUME_KEYWORDS = [
    "resume", "curriculum vitae", "objective", "career objective",
    "work experience", "professional experience", "education", "skills",
    "certifications", "references", "linkedin.com/in",
]
INVOICE_KEYWORDS = [
    "invoice", "bill to", "invoice number", "invoice no", "total amount",
    "amount due", "subtotal", "payment terms", "tax invoice", "receipt",
    "quantity", "unit price", "gst", "bill from",
]


def clean_text(text: str) -> str:
    """Remove extra spaces/blank lines; return '' for empty/very short text."""
    if not text:
        return ""
    text = text.replace("\r", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    text = "\n".join(line.strip() for line in text.split("\n")).strip()
    return text if len(text) >= 5 else ""


def keyword_score(text: str, keywords: list) -> int:
    t = text.lower()
    return sum(1 for kw in keywords if kw in t)


def rule_based_label(text: str) -> str:
    """Week 2 rule-based approach (kept as baseline)."""
    r, i = keyword_score(text, RESUME_KEYWORDS), keyword_score(text, INVOICE_KEYWORDS)
    if r >= 2 and r > i:
        return "Resume"
    if i >= 2 and i > r:
        return "Invoice"
    return "Other"


def make_pipe(model):
    return make_pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True), model)


def build_models() -> dict:
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Linear SVM": LinearSVC(),
        "Naive Bayes": MultinomialNB(),
    }


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATASET_PATH)
    df["clean_text"] = df["text"].apply(clean_text)
    return df[df["clean_text"] != ""].reset_index(drop=True)


def _report_block(name, y, pred, labels):
    rep = classification_report(y, pred, labels=labels, output_dict=True, zero_division=0)
    cm = pd.DataFrame(confusion_matrix(y, pred, labels=labels),
                      index=[f"true_{l}" for l in labels],
                      columns=[f"pred_{l}" for l in labels])
    m = rep["macro avg"]
    text = (f"=== {name} ===\n"
            f"Accuracy : {accuracy_score(y, pred):.3f}\n"
            f"Precision: {m['precision']:.3f} (macro)\n"
            f"Recall   : {m['recall']:.3f} (macro)\n"
            f"F1-score : {m['f1-score']:.3f} (macro)\n"
            f"Confusion matrix:\n{cm.to_string()}\n\n")
    return m["f1-score"], text


def evaluate(df: pd.DataFrame):
    """Same data + same stratified folds for every model = fair comparison."""
    y, X = df["label"], df["clean_text"]
    labels = sorted(y.unique())
    n_splits = max(2, min(5, int(y.value_counts().min())))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    scores, report = {}, [f"Dataset: {len(df)} docs | classes: {dict(y.value_counts())} | "
                          f"{n_splits}-fold cross-validation\n\n"]
    f1, txt = _report_block("Rule-based baseline", y, X.apply(rule_based_label), labels)
    report.append(txt)
    for name, model in build_models().items():
        pred = cross_val_predict(make_pipe(model), X, y, cv=cv)
        scores[name], txt = _report_block(name, y, pred, labels)
        report.append(txt)

    best = max(scores, key=scores.get)  # ties -> first (Logistic Regression)
    report.append(f"Selected model: {best} (highest macro F1 = {scores[best]:.3f})\n")
    return best, "".join(report)


def train_and_save():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"document_dataset.csv not found in {BASE_DIR}")
    df = load_dataset()
    best, report = evaluate(df)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    pipe = make_pipe(build_models()[best]).fit(df["clean_text"], df["label"])
    joblib.dump(pipe, MODEL_PATH)
    return pipe, report


if __name__ == "__main__":
    _, rep = train_and_save()
    print(rep)
    print(f"Saved model -> {MODEL_PATH}\nSaved report -> {REPORT_PATH}")