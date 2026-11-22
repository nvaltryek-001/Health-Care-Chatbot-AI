"""Small text-normalisation helpers shared by the NLP components."""

from __future__ import annotations

import re
import unicodedata

_APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "`": "'"})
_NON_WORD = re.compile(r"[^a-z0-9'\s]+")
_SPACES = re.compile(r"\s+")

# Expand common contractions so "can't breathe" and "cannot breathe" look alike.
_CONTRACTIONS = {
    "can't": "cannot",
    "cant": "cannot",
    "won't": "will not",
    "don't": "do not",
    "dont": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "isn't": "is not",
    "i'm": "i am",
    "im": "i am",
    "i've": "i have",
    "ive": "i have",
    "it's": "it is",
    "what's": "what is",
    "haven't": "have not",
    "hasn't": "has not",
    "aren't": "are not",
    "wasn't": "was not",
}


def normalize(text: str) -> str:
    """Lower-case, strip accents/punctuation and expand contractions.

    Hyphens become spaces so "low-grade fever" == "low grade fever".
    """
    text = unicodedata.normalize("NFKD", text or "")
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower().translate(_APOSTROPHES).replace("-", " ")
    text = _NON_WORD.sub(" ", text)
    words = [_CONTRACTIONS.get(w, w) for w in text.split()]
    text = " ".join(words).replace("'", "")
    return _SPACES.sub(" ", text).strip()


def tokens(text: str) -> list[str]:
    return normalize(text).split()
