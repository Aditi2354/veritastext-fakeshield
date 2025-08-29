#!/usr/bin/env python3
"""
train.py — Train Fake News detector, save artifacts, report metrics,
plot confusion matrix, and run k-fold cross-validation.
"""

import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import joblib

from utils import clean_text

LABEL_MAP = {0: "Real", 1: "Fake"}

def load_data(data_dir: Path) -> pd.DataFrame:
    true_path = data_dir / "True.csv"
    fake_path = data_dir / "Fake.csv"
    if not true_path.exists() or not fake_path.exists():
        raise FileNotFoundError(
            f"Dataset not found. Expected:\n  {true_path}\n  {fake_path}\n"
            "Download the Kaggle 'Fake and real news dataset' and place CSVs there."
        )

    true_df = pd.read_csv(true_path)
    fake_df = pd.read_csv(fake_path)

    def normalize(df):
        if "text" in df.columns:
            return df["text"].fillna("")
        if "content" in df.columns:
            return df["content"].fillna("")
        if "title" in df.columns and "text" in df.columns:
            return (df["title"].fillna("") + " " + df["text"].fillna("")).str.strip()
        raise ValueError("Could not find text content columns.")

    X_true = normalize(true_df)
    X_fake = normalize(fake_df)
    y_true = np.zeros(len(X_true), dtype=int)
    y_fake = np.ones(len(X_fake), dtype=int)

    X = pd.concat([X_true, X_fake], ignore_index=True)
    y = np.concatenate([y_true, y_fake], axis=0)

    df = pd.DataFrame({"text": X, "label": y})
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    return df

def plot_confusion_matrix(cm: np.ndarray, labels, out_path: Path):
    fig, ax = plt.subplots(figsize=(5, 4), dpi=150)
    im = ax.imshow(cm, interpolation='nearest')
    ax.set_title("Confusion Matrix")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data", help="Directory with True.csv and Fake.csv")
    parser.add_argument("--out_dir", type=str, default="artifacts", help="Where to save model files & reports")
    parser.add_argument("--max_features", type=int, default=50000)
    parser.add_argument("--ngram_max", type=int, default=2)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=0)
    parser.add_argument("--cv_splits", type=int, default=5, help="StratifiedKFold splits")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(data_dir)
    X = df["text"].astype(str).values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    vec = TfidfVectorizer(
        max_features=args.max_features,
        ngram_range=(1, args.ngram_max),
        lowercase=False,
        preprocessor=clean_text,
    )
    nb = MultinomialNB()
    pipe = Pipeline([("tfidf", vec), ("nb", nb)])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4, target_names=[LABEL_MAP[0], LABEL_MAP[1]])
    cm = confusion_matrix(y_test, y_pred)

    (out_dir / "label_map.json").write_text(json.dumps(LABEL_MAP, indent=2), encoding="utf-8")
    (out_dir / "metrics.txt").write_text(
        f"Accuracy: {acc:.4f}\n\nClassification Report:\n{report}\nConfusion Matrix:\n{cm}\n", encoding="utf-8"
    )
    plot_confusion_matrix(cm, [LABEL_MAP[0], LABEL_MAP[1]], out_dir / "confusion_matrix.png")

    import joblib
    joblib.dump(pipe, out_dir / "pipeline.joblib")
    joblib.dump(pipe.named_steps["tfidf"], out_dir / "tfidf.joblib")
    joblib.dump(pipe.named_steps["nb"], out_dir / "model_nb.joblib")

    from sklearn.model_selection import StratifiedKFold, cross_val_score
    skf = StratifiedKFold(n_splits=args.cv_splits, shuffle=True, random_state=args.random_state)
    cv_acc = cross_val_score(pipe, X, y, cv=skf, scoring="accuracy")
    cv_f1 = cross_val_score(pipe, X, y, cv=skf, scoring="f1_macro")

    (out_dir / "crossval.txt").write_text(
        f"StratifiedKFold={args.cv_splits}\n"
        f"Accuracy: mean={cv_acc.mean():.4f} std={cv_acc.std():.4f}\n"
        f"F1-macro: mean={cv_f1.mean():.4f} std={cv_f1.std():.4f}\n",
        encoding="utf-8"
    )

    print(f"Saved artifacts to: {out_dir.resolve()}")
    print(f"Test Accuracy: {acc:.4f}")
    print("CV Accuracy (mean ± std): {:.4f} ± {:.4f}".format(cv_acc.mean(), cv_acc.std()))
    print("CV F1 (mean ± std): {:.4f} ± {:.4f}".format(cv_f1.mean(), cv_f1.std()))

if __name__ == "__main__":
    main()
