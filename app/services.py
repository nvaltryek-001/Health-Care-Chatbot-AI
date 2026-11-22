"""Model registry: loads trained artefacts from ``models/`` or trains them on first use."""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

from .chatbot import MedBot
from .heart_model import HeartModel
from .intent_model import IntentModel
from .knowledge import load_knowledge_base

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
INTENT_PATH = MODEL_DIR / "intent_model.joblib"
HEART_PATH = MODEL_DIR / "heart_model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"

_lock = threading.Lock()
_cache: dict[str, object] = {}


def train_all(model_dir: Path = MODEL_DIR) -> dict:
    """Train every model, save it, and return the evaluation metrics."""
    intents = IntentModel.train()
    intents.save(model_dir / INTENT_PATH.name)
    heart = HeartModel.train()
    heart.save(model_dir / HEART_PATH.name)
    metrics = {"heart": heart.metrics}
    (model_dir / METRICS_PATH.name).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def _load_or_train(key: str, path: Path, loader, trainer):
    with _lock:
        if key in _cache:
            return _cache[key]
        model = None
        if path.exists():
            try:
                model = loader(path)
            except Exception:  # incompatible pickle (e.g. different sklearn version)
                log.warning("Could not load %s, retraining", path, exc_info=True)
        if model is None:
            model = trainer()
            try:
                model.save(path)
            except OSError:
                log.warning("Could not save %s (read-only filesystem?)", path)
        _cache[key] = model
        return model


def get_heart_model() -> HeartModel:
    return _load_or_train("heart", HEART_PATH, HeartModel.load, HeartModel.train)


def get_bot() -> MedBot:
    intents = _load_or_train("intents", INTENT_PATH, IntentModel.load, IntentModel.train)
    with _lock:
        if "bot" not in _cache:
            _cache["bot"] = MedBot(load_knowledge_base(), intents)
        return _cache["bot"]
