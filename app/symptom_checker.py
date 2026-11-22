"""Symptom extraction and condition ranking.

Extraction is phrase-based (longest phrase wins, with negation handling and
light typo tolerance). Ranking uses an IDF-weighted overlap score between the
user's symptoms and each condition's symptom profile, so rare, specific
symptoms count for more than generic ones like "fatigue".
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass

from .knowledge import SYMPTOM_PARENTS, KnowledgeBase
from .text import normalize as _normalize

# Determiners are dropped on both sides so "pain behind my eyes" matches
# the phrase "pain behind eyes".
_FILLERS = {"my", "the", "a", "an", "some", "of", "both"}


def normalize(text: str) -> str:
    return " ".join(w for w in _normalize(text).split() if w not in _FILLERS)

# Everyday conditions are slightly favoured over less common ones on a tie.
PREVALENCE_PRIOR = {"common": 1.0, "moderate": 0.93, "uncommon": 0.86}

NEGATORS = {"no", "not", "without", "never", "nor", "denies", "none", "zero"}
NEGATION_WINDOW = 4  # tokens to look back for a negator
# Words that end a negation's scope: "no fever but headache".
SCOPE_BREAKERS = {"but", "however", "although", "though", "yet", "except"}


@dataclass
class Extraction:
    present: list[str]
    negated: list[str]


@dataclass
class RankedCondition:
    id: str
    name: str
    score: float
    matched: list[str]
    missing: list[str]


class SymptomChecker:
    def __init__(self, kb: KnowledgeBase):
        self.kb = kb
        phrase_map: dict[str, str] = {}
        for sym in kb.symptoms.values():
            for phrase in sym.phrases:
                phrase_map.setdefault(normalize(phrase), sym.id)
        # Longest phrases first so "high fever" beats "fever".
        self._phrases = sorted(phrase_map.items(), key=lambda kv: -len(kv[0]))
        self._patterns = [
            (re.compile(rf"\b{re.escape(p)}\b"), sid) for p, sid in self._phrases
        ]
        self._single_words = {p: sid for p, sid in self._phrases if " " not in p and len(p) >= 5}

    # ------------------------------------------------------------------ extract
    def extract(self, text: str) -> Extraction:
        norm = normalize(text)
        if not norm:
            return Extraction([], [])
        consumed = [False] * len(norm)
        hits: list[tuple[int, str]] = []

        for pattern, sid in self._patterns:
            for m in pattern.finditer(norm):
                if any(consumed[m.start(): m.end()]):
                    continue
                for i in range(m.start(), m.end()):
                    consumed[i] = True
                hits.append((m.start(), sid))

        # Typo tolerance for single words that weren't matched ("diarhea").
        pos = 0
        for word in norm.split(" "):
            start = norm.index(word, pos)
            pos = start + len(word)
            if len(word) < 5 or any(consumed[start:pos]):
                continue
            close = difflib.get_close_matches(word, self._single_words.keys(), n=1, cutoff=0.86)
            if close:
                hits.append((start, self._single_words[close[0]]))

        present: list[str] = []
        negated: list[str] = []
        for start, sid in sorted(hits):
            target = negated if self._is_negated(norm, start) else present
            if sid not in target:
                target.append(sid)
        present = [s for s in present if s not in negated]
        return Extraction(present=present, negated=negated)

    @staticmethod
    def _is_negated(norm: str, start: int) -> bool:
        before = norm[:start].split()
        window = before[-NEGATION_WINDOW:]
        for word in reversed(window):
            if word in SCOPE_BREAKERS:
                return False
            if word in NEGATORS:
                return True
        return False

    # --------------------------------------------------------------------- rank
    def expand(self, symptoms: list[str]) -> set[str]:
        """Add implied parent symptoms (high fever -> fever)."""
        expanded = set(symptoms)
        for sid in symptoms:
            parent = SYMPTOM_PARENTS.get(sid)
            if parent:
                expanded.add(parent)
        return expanded

    def rank(self, symptoms: list[str], top_k: int = 3) -> list[RankedCondition]:
        if not symptoms:
            return []
        user = self.expand(symptoms)
        idf = self.kb.idf
        user_weight = sum(idf[s] for s in user) or 1.0
        results: list[RankedCondition] = []

        for cond in self.kb.conditions:
            n = len(cond.symptoms)
            cond_set = set(cond.symptoms)
            cond_weight = 0.0
            matched_weight = 0.0
            explained: set[str] = set()
            matched: list[str] = []
            for i, sid in enumerate(cond.symptoms):
                # Symptoms listed first are the most characteristic.
                w = idf[sid] * (1.0 - 0.5 * i / max(n - 1, 1))
                cond_weight += w
                credit = 0.0
                if sid in user:
                    credit = 1.0
                    explained.add(sid)
                elif SYMPTOM_PARENTS.get(sid) in user and SYMPTOM_PARENTS[sid] not in cond_set:
                    credit = 0.6  # user said "fever", condition expects "high fever"
                    explained.add(SYMPTOM_PARENTS[sid])
                if credit:
                    matched_weight += w * credit
                    matched.append(sid)
            if not matched:
                continue
            precision = matched_weight / cond_weight  # how much of the profile is present
            recall = sum(idf[s] for s in explained) / user_weight  # how much of the user is explained
            score = (2 * precision * recall) / (precision + recall)
            score *= PREVALENCE_PRIOR.get(cond.prevalence, 1.0)
            missing = [s for s in cond.symptoms if s not in user and s not in matched]
            results.append(RankedCondition(cond.id, cond.name, round(score, 4), matched, missing))

        results.sort(key=lambda r: (-r.score, r.name))
        return results[:top_k]

    def follow_up_questions(self, ranked: list[RankedCondition], known: list[str], limit: int = 4) -> list[str]:
        """Pick unreported symptoms that best separate the top candidates."""
        known_set = self.expand(known)
        candidates: dict[str, float] = {}
        for rank, rc in enumerate(ranked):
            for i, sid in enumerate(rc.missing):
                if sid in known_set:
                    continue
                gain = self.kb.idf[sid] * (1.0 / (rank + 1)) * (1.0 - 0.08 * i)
                candidates[sid] = candidates.get(sid, 0.0) + gain
        ordered: list[str] = []
        for sid in sorted(candidates, key=lambda s: -candidates[s]):
            family = {sid, SYMPTOM_PARENTS.get(sid)}
            if any(family & {o, SYMPTOM_PARENTS.get(o)} - {None} for o in ordered):
                continue  # don't ask about "fever" and "mild fever" together
            ordered.append(sid)
        return ordered[:limit]
