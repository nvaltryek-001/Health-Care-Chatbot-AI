"""Intent classifier trained on ``data/intents.json``.

A TF-IDF (word + character n-gram) representation feeds a logistic-regression
classifier. Character n-grams make it tolerant to typos such as "helo" or
"thnaks". The model is tiny and trains in well under a second.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from .knowledge import DATA_DIR
from .text import normalize


@dataclass
class IntentPrediction:
    tag: str
    confidence: float


class IntentModel:
    def __init__(self, pipeline: Pipeline, responses: dict[str, list[str]]):
        self.pipeline = pipeline
        self.responses = responses

    @staticmethod
    def build_pipeline() -> Pipeline:
        features = FeatureUnion(
            [
                ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True)),
                ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True)),
            ]
        )
        clf = LogisticRegression(C=20, max_iter=2000)
        return Pipeline([("features", features), ("clf", clf)])

    @classmethod
    def load_intents(cls, path: Path | None = None) -> dict:
        path = path or DATA_DIR / "intents.json"
        return json.loads(path.read_text(encoding="utf-8"))

    @classmethod
    def train(cls, intents: dict | None = None) -> "IntentModel":
        intents = intents or cls.load_intents()
        x, y = [], []
        for intent in intents["intents"]:
            for pattern in intent["patterns"]:
                x.append(normalize(pattern))
                y.append(intent["tag"])
        pipeline = cls.build_pipeline()
        pipeline.fit(x, y)
        responses = {i["tag"]: i["responses"] for i in intents["intents"]}
        return cls(pipeline, responses)

    def predict(self, text: str) -> IntentPrediction:
        norm = normalize(text)
        if not norm:
            return IntentPrediction("unknown", 0.0)
        probs = self.pipeline.predict_proba([norm])[0]
        best = probs.argmax()
        return IntentPrediction(self.pipeline.classes_[best], float(probs[best]))

    def respond(self, tag: str, rng: random.Random | None = None) -> str:
        options = self.responses.get(tag) or ["Sorry, I didn't get that."]
        return (rng or random).choice(options)

    # ------------------------------------------------------------ persistence
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": self.pipeline, "responses": self.responses}, path)

    @classmethod
    def load(cls, path: Path) -> "IntentModel":
        blob = joblib.load(path)
        return cls(blob["pipeline"], blob["responses"])
