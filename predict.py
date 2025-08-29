#!/usr/bin/env python3
"""
predict.py — CLI for single-text or file-based predictions using saved artifacts.
Usage:
  python predict.py --text "some news text"
  python predict.py --file path/to/texts.txt   # one article per line
"""
import argparse
from pathlib import Path
import joblib
import numpy as np

LABEL_MAP = {0: "Real", 1: "Fake"}

def load_artifacts(art_dir: Path):
    pipe_path = art_dir / "pipeline.joblib"
    if pipe_path.exists():
        pipe = joblib.load(pipe_path)
        return pipe, None, None
    vec = joblib.load(art_dir / "tfidf.joblib")
    model = joblib.load(art_dir / "model_nb.joblib")
    return None, vec, model

def predict_texts(texts, art_dir: Path):
    pipe, vec, model = load_artifacts(art_dir)
    texts = [t if isinstance(t, str) else str(t) for t in texts]
    if pipe is not None:
        probs = pipe.predict_proba(texts)
        preds = pipe.predict(texts)
    else:
        X = vec.transform(texts)  # assumes same preprocessor baked into vectorizer
        probs = model.predict_proba(X)
        preds = model.predict(X)
    return preds, probs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, help="Single input text to classify")
    parser.add_argument("--file", type=str, help="Path to a text file (one item per line)")
    parser.add_argument("--artifacts", type=str, default="artifacts", help="Artifacts directory")
    args = parser.parse_args()

    art_dir = Path(args.artifacts)
    if args.text:
        texts = [args.text]
    elif args.file:
        lines = Path(args.file).read_text(encoding="utf-8").splitlines()
        texts = [ln.strip() for ln in lines if ln.strip()]
    else:
        raise SystemExit("Provide --text or --file")

    preds, probs = predict_texts(texts, art_dir)
    for i, (p, pr) in enumerate(zip(preds, probs)):
        label = LABEL_MAP.get(int(p), str(p))
        conf = float(np.max(pr))
        print(f"[{i}] -> {label} (confidence={conf:.4f})")

if __name__ == "__main__":
    main()
