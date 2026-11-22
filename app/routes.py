"""HTML pages and JSON API."""

from __future__ import annotations

import json
from collections import Counter

from flask import Blueprint, jsonify, render_template, request

from . import __version__
from .heart_model import FEATURES, dataset_insights
from .knowledge import load_knowledge_base
from .services import get_bot, get_heart_model

pages = Blueprint("pages", __name__)
api = Blueprint("api", __name__)


# --------------------------------------------------------------------- pages
@pages.get("/")
def chat_page():
    return render_template("chat.html", active="chat")


@pages.get("/heart")
def heart_page():
    return render_template("heart.html", active="heart", features=FEATURES)


@pages.get("/insights")
def insights_page():
    return render_template("insights.html", active="insights")


@pages.get("/conditions")
def conditions_page():
    kb = load_knowledge_base()
    conditions = sorted(kb.conditions, key=lambda c: c.name)
    categories = sorted({c.category for c in conditions})
    labels = {c.id: [kb.label(s) for s in c.symptoms] for c in conditions}
    return render_template("conditions.html", active="conditions", conditions=conditions,
                           categories=categories, labels=labels)


@pages.get("/about")
def about_page():
    return render_template("about.html", active="about")


# ----------------------------------------------------------------------- api
def _error(message: str, status: int = 400, **extra):
    return jsonify({"error": message, **extra}), status


@api.get("/health")
def health():
    return jsonify({"status": "ok", "version": __version__})


@api.post("/chat")
def chat():
    data = request.get_json(silent=True) or {}
    message = data.get("message")
    if not isinstance(message, str):
        return _error("'message' must be a string")
    symptoms = data.get("symptoms") or []
    if not isinstance(symptoms, list) or not all(isinstance(s, str) for s in symptoms):
        return _error("'symptoms' must be a list of symptom ids")
    if data.get("action") == "rerank":
        return jsonify(get_bot().rerank(symptoms).to_dict())
    return jsonify(get_bot().reply(message, symptoms).to_dict())


@api.post("/heart/predict")
def heart_predict():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return _error("Request body must be a JSON object")
    try:
        result = get_heart_model().predict(data)
    except ValueError as exc:
        return _error("Invalid input", fields=json.loads(str(exc)))
    return jsonify(result.to_dict())


@api.get("/heart/features")
def heart_features():
    return jsonify(FEATURES)


@api.get("/insights")
def insights():
    kb = load_knowledge_base()
    model = get_heart_model()
    by_category = Counter(c.category for c in kb.conditions)
    return jsonify({
        "heart": dataset_insights(),
        "model": model.metrics,
        "knowledge_base": {
            "conditions": len(kb.conditions),
            "symptoms": len(kb.symptoms),
            "phrases": sum(len(s.phrases) for s in kb.symptoms.values()),
            "by_category": [{"group": k, "count": v} for k, v in by_category.most_common()],
        },
    })


@api.get("/conditions")
def conditions():
    kb = load_knowledge_base()
    return jsonify([c.to_dict() for c in kb.conditions])


@api.get("/symptoms")
def symptoms():
    kb = load_knowledge_base()
    return jsonify([{"id": s.id, "label": s.label} for s in kb.symptoms.values()])
