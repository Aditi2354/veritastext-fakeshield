# VeritasText — FakeShield (Fake News Detector)

A polished, portfolio-ready **NLP project** that trains a TF‑IDF + Multinomial Naive Bayes model to classify news as **Real** or **Fake**, with saved artifacts, a CLI **predictor**, a **Streamlit** demo, **confusion matrix** plot, and **k‑fold cross‑validation**.

## Project Structure
```
VeritasText-FakeShield/
├─ app.py
├─ predict.py
├─ train.py
├─ utils.py
├─ requirements.txt
├─ README.md
├─ data/
│   ├─ True.csv    # place here
│   └─ Fake.csv    # place here
└─ artifacts/      # created by train.py
    ├─ pipeline.joblib
    ├─ tfidf.joblib
    ├─ model_nb.joblib
    ├─ label_map.json
    ├─ metrics.txt
    ├─ crossval.txt
    └─ confusion_matrix.png
```

## Quickstart

```bash
# 1) Create & activate a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Put dataset files
#   ./data/True.csv
#   ./data/Fake.csv
# (Kaggle: "Fake and real news dataset")

# 4) Train & evaluate
python train.py --data_dir data --out_dir artifacts

# Optional hyperparams
python train.py --max_features 50000 --ngram_max 2 --test_size 0.2 --random_state 0 --cv_splits 5

# 5) Predict from CLI
python predict.py --text "This is an example news text."
# or a file, one item per line:
python predict.py --file samples.txt

# 6) Streamlit demo
streamlit run app.py
```

## What gets saved
- `pipeline.joblib`: Full sklearn pipeline (TF-IDF + NB) — recommended loader
- `tfidf.joblib`, `model_nb.joblib`: Vectorizer and classifier, saved separately
- `metrics.txt`: Hold-out accuracy & classification report
- `crossval.txt`: StratifiedKFold accuracy and F1 (mean ± std)
- `confusion_matrix.png`: Confusion matrix image

## Notes
- Text cleaning is baked into the TF‑IDF vectorizer via a custom preprocessor, so inference uses the **same pipeline** as training.
- For different models (e.g., Linear SVM, Logistic Regression), swap the classifier in `train.py`.
