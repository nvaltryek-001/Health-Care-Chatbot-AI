"""Heart-disease risk model trained on ``data/heart-disease.csv``.

About the dataset
-----------------
The CSV is the popular Kaggle re-upload of the UCI *Cleveland* heart-disease
data (303 patients, 13 features). In that re-upload several columns were
re-coded and the label is inverted relative to the original study: patients
with ``target == 1`` are younger, reach a higher max heart rate, rarely have
exercise-induced angina and have fewer blocked vessels — the profile of the
*healthy* group. The class counts (165 / 138) also match the original UCI
counts for "no disease" / "disease" (164 / 139). We therefore model
``has_disease = 1 - target``.

Categorical codes were decoded the same way, by matching category frequencies
with the original UCI documentation (see ``FEATURES`` below).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .knowledge import DATA_DIR

DATASET = DATA_DIR / "heart-disease.csv"
RANDOM_STATE = 42

NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
BINARY = ["sex", "fbs", "exang"]
CATEGORICAL = ["cp", "restecg", "slope", "thal"]

# Field metadata drives both validation and the web form.
FEATURES: dict[str, dict] = {
    "age": {"label": "Age", "unit": "years", "min": 18, "max": 100, "default": 50},
    "sex": {"label": "Sex", "options": {1: "Male", 0: "Female"}, "default": 1},
    "cp": {
        "label": "Chest pain type",
        "options": {0: "No chest pain (asymptomatic)", 1: "Atypical angina", 2: "Non-anginal pain", 3: "Typical angina"},
        "default": 0,
        "help": "Typical angina = pressure-like chest pain on exertion, relieved by rest.",
    },
    "trestbps": {"label": "Resting blood pressure", "unit": "mmHg", "min": 80, "max": 220, "default": 130},
    "chol": {"label": "Serum cholesterol", "unit": "mg/dl", "min": 100, "max": 600, "default": 240},
    "fbs": {"label": "Fasting blood sugar > 120 mg/dl", "options": {0: "No", 1: "Yes"}, "default": 0},
    "restecg": {
        "label": "Resting ECG",
        "options": {1: "Normal", 2: "ST-T wave abnormality", 0: "Left ventricular hypertrophy"},
        "default": 1,
    },
    "thalach": {"label": "Max heart rate achieved", "unit": "bpm", "min": 60, "max": 220, "default": 150,
                "help": "Highest heart rate reached during an exercise (treadmill) test."},
    "exang": {"label": "Chest pain during exercise", "options": {0: "No", 1: "Yes"}, "default": 0},
    "oldpeak": {"label": "ST depression (oldpeak)", "unit": "mm", "min": 0, "max": 7, "step": 0.1, "default": 1.0,
                "help": "ST-segment depression on ECG during exercise relative to rest."},
    "slope": {
        "label": "Slope of peak exercise ST segment",
        "options": {2: "Upsloping", 1: "Flat", 0: "Downsloping"},
        "default": 2,
    },
    "ca": {"label": "Major vessels coloured by fluoroscopy", "min": 0, "max": 3, "default": 0,
           "help": "Number of major vessels (0–3) seen on angiography. Use 0 if unknown."},
    "thal": {
        "label": "Thalassemia stress test",
        "options": {2: "Normal", 1: "Fixed defect", 3: "Reversible defect"},
        "default": 2,
    },
}

FRIENDLY = {
    "age": "Age", "sex": "Male sex", "trestbps": "Resting blood pressure", "chol": "Cholesterol",
    "fbs": "High fasting blood sugar", "thalach": "Max heart rate", "exang": "Exercise-induced chest pain",
    "oldpeak": "ST depression", "ca": "Blocked major vessels",
    "cp_0": "No chest-pain symptoms*", "cp_1": "Atypical angina", "cp_2": "Non-anginal chest pain",
    "cp_3": "Typical angina", "restecg_0": "LV hypertrophy on ECG", "restecg_1": "Normal ECG",
    "restecg_2": "ST-T abnormality on ECG", "slope_0": "Downsloping ST", "slope_1": "Flat ST slope",
    "slope_2": "Upsloping ST", "thal_1": "Fixed thal defect", "thal_2": "Normal thal result",
    "thal_3": "Reversible thal defect",
}


def load_dataset(path: Path = DATASET) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig").drop_duplicates().reset_index(drop=True)
    # Missing values were stored as out-of-range codes in this version.
    df.loc[df["thal"] == 0, "thal"] = 2
    df.loc[df["ca"] == 4, "ca"] = 0
    df["has_disease"] = 1 - df["target"]
    return df


def _pipeline(model) -> Pipeline:
    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC + BINARY),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )
    return Pipeline([("pre", pre), ("model", model)])


@dataclass
class RiskResult:
    probability: float
    level: str
    factors: list[dict]

    def to_dict(self) -> dict:
        return {"probability": self.probability, "percent": min(99, max(1, round(self.probability * 100))),
                "level": self.level, "factors": self.factors}


class HeartModel:
    def __init__(self, pipeline: Pipeline, metrics: dict):
        self.pipeline = pipeline
        self.metrics = metrics

    # ---------------------------------------------------------------- training
    @classmethod
    def train(cls, df: pd.DataFrame | None = None) -> "HeartModel":
        df = load_dataset() if df is None else df
        x = df[NUMERIC + BINARY + CATEGORICAL]
        y = df["has_disease"]
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

        candidates = {
            "Logistic Regression": LogisticRegression(C=0.3, max_iter=2000),
            "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=5, random_state=RANDOM_STATE),
        }
        comparison = {}
        for name, est in candidates.items():
            auc = cross_val_score(_pipeline(est), x, y, cv=cv, scoring="roc_auc")
            acc = cross_val_score(_pipeline(est), x, y, cv=cv, scoring="accuracy")
            comparison[name] = {"cv_roc_auc": round(float(auc.mean()), 3), "cv_accuracy": round(float(acc.mean()), 3)}

        # Logistic regression is deployed: it's as accurate as the forest on this
        # small dataset and its coefficients give an honest per-feature explanation.
        x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
        holdout = _pipeline(LogisticRegression(C=0.3, max_iter=2000)).fit(x_tr, y_tr)
        proba = holdout.predict_proba(x_te)[:, 1]
        pred = (proba >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()

        final = _pipeline(LogisticRegression(C=0.3, max_iter=2000)).fit(x, y)
        metrics = {
            "model": "Logistic Regression",
            "n_samples": int(len(df)),
            "n_test": int(len(y_te)),
            "test": {
                "accuracy": round(accuracy_score(y_te, pred), 3),
                "precision": round(precision_score(y_te, pred), 3),
                "recall": round(recall_score(y_te, pred), 3),
                "f1": round(f1_score(y_te, pred), 3),
                "roc_auc": round(roc_auc_score(y_te, proba), 3),
                "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
            },
            "comparison": comparison,
        }
        model = cls(final, metrics)
        metrics["coefficients"] = model.coefficients()
        return model

    # -------------------------------------------------------------- inference
    def _feature_names(self) -> list[str]:
        names = self.pipeline.named_steps["pre"].get_feature_names_out()
        return [n.split("__", 1)[1] for n in names]

    def coefficients(self) -> list[dict]:
        coefs = self.pipeline.named_steps["model"].coef_[0]
        rows = [{"feature": f, "label": FRIENDLY.get(f, f), "weight": round(float(c), 3)}
                for f, c in zip(self._feature_names(), coefs)]
        return sorted(rows, key=lambda r: -abs(r["weight"]))

    @staticmethod
    def validate(payload: dict) -> dict:
        clean, errors = {}, {}
        for name, meta in FEATURES.items():
            raw = payload.get(name, meta["default"])
            try:
                value = float(raw)
            except (TypeError, ValueError):
                errors[name] = "must be a number"
                continue
            if "options" in meta:
                if int(value) not in meta["options"]:
                    errors[name] = f"must be one of {sorted(meta['options'])}"
                    continue
                value = int(value)
            elif not meta["min"] <= value <= meta["max"]:
                errors[name] = f"must be between {meta['min']} and {meta['max']}"
                continue
            clean[name] = value
        if errors:
            raise ValueError(json.dumps(errors))
        return clean

    def predict(self, payload: dict) -> RiskResult:
        values = self.validate(payload)
        row = pd.DataFrame([values])[NUMERIC + BINARY + CATEGORICAL]
        prob = float(self.pipeline.predict_proba(row)[0, 1])
        level = "Low" if prob < 0.3 else "Moderate" if prob < 0.6 else "High"

        # Contribution of each feature to the log-odds (relative to an average patient).
        transformed = self.pipeline.named_steps["pre"].transform(row)
        transformed = transformed.toarray() if hasattr(transformed, "toarray") else transformed
        coefs = self.pipeline.named_steps["model"].coef_[0]
        contrib = transformed[0] * coefs
        names = self._feature_names()
        factors = []
        for idx in np.argsort(-np.abs(contrib))[:5]:
            if abs(contrib[idx]) < 0.05:
                continue
            factors.append({
                "feature": names[idx], "label": self._describe(names[idx], values),
                "impact": round(float(contrib[idx]), 3),
                "direction": "increases" if contrib[idx] > 0 else "decreases",
            })
        return RiskResult(round(prob, 4), level, factors)

    @staticmethod
    def _describe(feature: str, values: dict) -> str:
        """Human-readable 'Field: value' for an (encoded) feature name."""
        base = feature.split("_")[0] if feature.split("_")[0] in CATEGORICAL else feature
        meta = FEATURES[base]
        value = values[base]
        if "options" in meta:
            shown = meta["options"][int(value)]
        else:
            shown = f"{value:g} {meta.get('unit', '')}".strip()
        return f"{meta['label']}: {shown}"

    # ------------------------------------------------------------ persistence
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": self.pipeline, "metrics": self.metrics}, path)

    @classmethod
    def load(cls, path: Path) -> "HeartModel":
        blob = joblib.load(path)
        return cls(blob["pipeline"], blob["metrics"])


def dataset_insights(df: pd.DataFrame | None = None) -> dict:
    """Aggregates for the Insights dashboard."""
    df = load_dataset() if df is None else df
    bins = [0, 40, 50, 60, 70, 120]
    labels = ["<40", "40–49", "50–59", "60–69", "70+"]
    df = df.assign(age_group=pd.cut(df["age"], bins=bins, labels=labels, right=False))

    def rate_by(col: str, mapping: dict | None = None) -> list[dict]:
        out = []
        for key, grp in df.groupby(col, observed=True):
            out.append({
                "group": mapping.get(int(key), str(key)) if mapping else str(key),
                "count": int(len(grp)),
                "rate": round(float(grp["has_disease"].mean()) * 100, 1),
            })
        return out

    healthy = df[df.has_disease == 0]
    sick = df[df.has_disease == 1]
    compare = []
    for col, label in [("age", "Age (yrs)"), ("trestbps", "Resting BP"), ("chol", "Cholesterol"),
                       ("thalach", "Max heart rate"), ("oldpeak", "ST depression")]:
        compare.append({"metric": label, "healthy": round(float(healthy[col].mean()), 1),
                        "disease": round(float(sick[col].mean()), 1)})

    return {
        "total": int(len(df)),
        "with_disease": int(df.has_disease.sum()),
        "without_disease": int((df.has_disease == 0).sum()),
        "by_age": rate_by("age_group"),
        "by_sex": rate_by("sex", {0: "Female", 1: "Male"}),
        "by_cp": rate_by("cp", FEATURES["cp"]["options"]),
        "by_exang": rate_by("exang", {0: "No exercise angina", 1: "Exercise angina"}),
        "compare": compare,
    }
