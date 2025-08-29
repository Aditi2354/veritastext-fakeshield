#!/usr/bin/env python3
# app.py — Streamlit demo for Fake News detection
import streamlit as st
from pathlib import Path
import joblib
import numpy as np

LABEL_MAP = {0: "Real", 1: "Fake"}

@st.cache_resource
def load_pipeline(art_dir: str = "artifacts"):
    path = Path(art_dir) / "pipeline.joblib"
    return joblib.load(path)

def main():
    st.set_page_config(page_title="VeritasText — Fake News Detector", page_icon="📰")
    st.title("📰 VeritasText — Fake News Detector")
    st.write("Type/paste a news paragraph below and click **Predict**.")

    text = st.text_area("Input text", height=200, placeholder="Paste news content here...")
    if st.button("Predict"):
        if not text.strip():
            st.warning("Please enter some text.")
        else:
            pipe = load_pipeline()
            probs = pipe.predict_proba([text])[0]
            pred = int(pipe.predict([text])[0])
            label = LABEL_MAP.get(pred, str(pred))
            st.subheader(f"Prediction: {label}")
            st.write(f"Confidence: {float(np.max(probs)):.4f}")
            st.progress(float(np.max(probs)))

    with st.expander("How this works"):
        st.markdown("""
- TF-IDF (1–2 grams, 50k features) + Multinomial Naive Bayes
- Custom text cleaner (lowercase, remove non-letters, stopwords, lemmatize)
- Trained on Kaggle 'Fake vs True' dataset
""")

if __name__ == "__main__":
    main()
