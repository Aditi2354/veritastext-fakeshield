# VeritasText — FakeShield 🛡️📰
*A compact, portfolio-ready NLP project that detects **Fake vs Real** news using TF-IDF + Naive Bayes, with saved artifacts, CLI prediction, a Streamlit demo, confusion matrix, and k-fold cross-validation.*

![Demo Confusion Matrix (mini dataset)](docs/cm_mini.png)
![Confusion Matrix (imbalanced run)](docs/cm_imbalanced.png)

---

## ✨ Highlights
- **End-to-end pipeline:** cleaning → TF-IDF (1–3 grams, up to 100k feats) → Multinomial Naive Bayes  
- **Reproducible artifacts:** whole pipeline saved with `joblib` (`pipeline.joblib`)  
- **Evaluation:** hold-out metrics + **k-fold CV** (+ confusion matrix image)  
- **Easy inference:** `predict.py` (CLI) and `Streamlit` UI (`app.py`)  
- **Clean code:** small, readable scripts + `requirements.txt` + `README`  

---

## 📦 Project Structure
VeritasText-FakeShield/
├─ app.py # Streamlit demo
├─ predict.py # CLI predictor
├─ train.py # Training + CV + confusion matrix + artifacts
├─ utils.py # Text cleaning (stopwords + lemmatization)
├─ requirements.txt
├─ README.md
├─ data/
│ ├─ True.csv # place dataset here
│ └─ Fake.csv
├─ artifacts/ # generated after training
│ ├─ pipeline.joblib
│ ├─ tfidf.joblib
│ ├─ model_nb.joblib
│ ├─ label_map.json
│ ├─ metrics.txt
│ ├─ crossval.txt
│ └─ confusion_matrix.png
└─ docs/
├─ cm_mini.png # ← your screenshot 1 (optional)
└─ cm_imbalanced.png # ← your screenshot 2 (optional)

yaml
Copy code

---

## 🚀 Quickstart

> Windows PowerShell commands shown; macOS/Linux me `source .venv/bin/activate` use karein.

```powershell
# 1) Create and activate a virtualenv
python -m venv .venv
.venv\Scripts\activate

# 2) Install deps
pip install -r requirements.txt

# 3) Put dataset
# data\True.csv  and  data\Fake.csv
# (Kaggle: "Fake and real news dataset")

# 4) Train & evaluate (saves artifacts + metrics + confusion matrix)
python train.py --data_dir data --out_dir artifacts --cv_splits 5 --test_size 0.2 --max_features 100000 --ngram_max 3

# 5) Predict from CLI
python predict.py --text "NASA releases new images from the lunar mission."
# or batch (one article per line)
python predict.py --file samples.txt

# 6) Streamlit demo
streamlit run app.py
🧠 How it works (short)
Cleaning: lowercase, non-letters remove, NLTK stopwords, WordNet lemmatization

Vectorization: TF-IDF with 1–3-grams, up to 100k features

Model: Multinomial Naive Bayes (sklearn)

Artifacts & Reports:

pipeline.joblib (full pipeline to load in one line)

metrics.txt (accuracy + classification report)

crossval.txt (StratifiedKFold mean±std for Accuracy & F1)

confusion_matrix.png

📊 Results (sample)
Screenshots above show two runs:

Mini sample (few lines each): for quick pipeline sanity (not indicative of real performance).

Imbalanced run: intentionally skewed data shows high accuracy but poor macro-F1 (predicting one class); great talking point about class imbalance and evaluation beyond accuracy.

For portfolio, train on the full Kaggle dataset (tens of thousands of rows) for realistic metrics.

🔧 Useful Commands
Change n-gram/feature budget

powershell
Copy code
python train.py --ngram_max 3 --max_features 100000
Adjust CV/test split (small data)

powershell
Copy code
python train.py --cv_splits 3 --test_size 0.2
Explicit artifacts path in predict

powershell
Copy code
python predict.py --artifacts artifacts --text "Some news text"
📝 Notes & Tips
If NLTK resources error:

powershell
Copy code
python -c "import nltk; [nltk.download(x) for x in ['stopwords','wordnet','omw-1.4']]"
.gitignore should exclude .venv/, artifacts/, and data/*.csv (keep data/.gitkeep).

Class imbalance can make accuracy misleading; consider undersampling/oversampling, or try ComplementNB/Linear SVM as an upgrade.

🛣️ Roadmap (nice to have)
Model comparison (NB vs Logistic Regression vs Linear SVM)

Hyperparameter search (GridSearchCV / RandomizedSearchCV)

Explainability (LIME/SHAP for sample texts)

Dockerfile + small CI (lint + smoke tests)

Model card & dataset card

📤 Publish to GitHub (short)
powershell
Copy code
git init
@"
__pycache__/
*.pyc
.venv/
artifacts/
data/*.csv
!data/.gitkeep
.DS_Store
Thumbs.db
.vscode/
"@ | Out-File .gitignore -Encoding UTF8

git add .
git commit -m "VeritasText-FakeShield: TF-IDF + NB with Streamlit demo"
git branch -M main
git remote add origin https://github.com/<username>/veritastext-fakeshield.git
git push -u origin main
