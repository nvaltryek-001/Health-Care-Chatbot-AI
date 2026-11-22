"""Loads the curated medical knowledge base (symptoms + conditions) from ``data/``."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# A specific symptom implies its general form ("high fever" => "fever").
SYMPTOM_PARENTS = {
    "high_fever": "fever",
    "low_grade_fever": "fever",
    "throbbing_headache": "headache",
    "pressing_headache": "headache",
    "morning_headache": "headache",
    "dry_cough": "cough",
    "wet_cough": "cough",
    "painful_swallowing": "sore_throat",
}


@dataclass(frozen=True)
class Symptom:
    id: str
    label: str
    phrases: tuple[str, ...]


@dataclass
class Condition:
    id: str
    name: str
    aliases: list[str]
    category: str
    urgency: str
    specialist: str
    summary: str
    symptoms: list[str]
    self_care: list[str]
    medicines: list[str]
    see_doctor: list[str]
    prevention: list[str]
    prevalence: str = "common"

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class KnowledgeBase:
    symptoms: dict[str, Symptom]
    conditions: list[Condition]
    idf: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.conditions)
        df: dict[str, int] = {}
        for cond in self.conditions:
            for sid in set(cond.symptoms):
                df[sid] = df.get(sid, 0) + 1
        # Rare symptoms (e.g. "pain behind the eyes") are more informative
        # than common ones (e.g. "fatigue").
        self.idf = {sid: math.log(1 + n / df.get(sid, 1)) for sid in self.symptoms}

    def condition(self, cid: str) -> Condition | None:
        return next((c for c in self.conditions if c.id == cid), None)

    def label(self, sid: str) -> str:
        sym = self.symptoms.get(sid)
        return sym.label if sym else sid.replace("_", " ").title()


@lru_cache(maxsize=1)
def load_knowledge_base(data_dir: str | None = None) -> KnowledgeBase:
    base = Path(data_dir) if data_dir else DATA_DIR
    raw_symptoms = json.loads((base / "symptoms.json").read_text(encoding="utf-8"))
    raw_conditions = json.loads((base / "conditions.json").read_text(encoding="utf-8"))

    symptoms = {
        sid: Symptom(id=sid, label=v["label"], phrases=tuple(v["phrases"]))
        for sid, v in raw_symptoms.items()
    }
    conditions = [Condition(**c) for c in raw_conditions]

    unknown = {s for c in conditions for s in c.symptoms if s not in symptoms}
    if unknown:
        raise ValueError(f"conditions.json references unknown symptoms: {sorted(unknown)}")
    return KnowledgeBase(symptoms=symptoms, conditions=conditions)
