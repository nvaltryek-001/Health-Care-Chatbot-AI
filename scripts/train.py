"""Train all MedBot models and print their evaluation metrics.

Usage:  python scripts/train.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services import MODEL_DIR, train_all  # noqa: E402


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    metrics = train_all()
    heart = metrics["heart"]
    test = heart["test"]
    print("Intent model ........ trained on data/intents.json")
    print(f"Heart model ......... {heart['model']} on {heart['n_samples']} patients")
    print(f"  hold-out accuracy   {test['accuracy']:.3f}")
    print(f"  hold-out ROC-AUC    {test['roc_auc']:.3f}")
    print(f"  precision / recall  {test['precision']:.3f} / {test['recall']:.3f}")
    for name, cv in heart["comparison"].items():
        print(f"  5-fold CV {name:<20} AUC {cv['cv_roc_auc']:.3f}  acc {cv['cv_accuracy']:.3f}")
    print(f"Saved to {MODEL_DIR}")


if __name__ == "__main__":
    main()
