#!/usr/bin/env python3
# utils.py - shared text cleaning utilities

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Lazy globals
_STOP_WORDS = None
_LEMMATIZER = None
_NON_ALPHA = re.compile(r"[^a-zA-Z]+")

def ensure_nltk():
    try:
        stopwords.words("english")
    except LookupError:
        nltk.download("stopwords")
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet")
        nltk.download("omw-1.4")

def _get_stopwords():
    global _STOP_WORDS
    if _STOP_WORDS is None:
        _STOP_WORDS = set(stopwords.words("english"))
    return _STOP_WORDS

def _get_lemmatizer():
    global _LEMMATIZER
    if _LEMMATIZER is None:
        _LEMMATIZER = WordNetLemmatizer()
    return _LEMMATIZER

def clean_text(text: str) -> str:
    if text is None:
        return ""
    ensure_nltk()
    sw = _get_stopwords()
    lem = _get_lemmatizer()

    text = text.lower()
    text = _NON_ALPHA.sub(" ", text)
    tokens = text.split()
    tokens = [lem.lemmatize(t) for t in tokens if t not in sw and len(t) > 2]
    return " ".join(tokens)
