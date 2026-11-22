"""MedBot conversation engine.

Every message goes through the same pipeline::

    safety check -> symptom / condition extraction -> intent model -> fallback

The server is stateless: the client sends back the list of symptoms collected
so far, and every reply contains the updated list.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

from . import safety
from .intent_model import IntentModel
from .knowledge import SYMPTOM_PARENTS, KnowledgeBase
from .symptom_checker import SymptomChecker
from .text import normalize

INTENT_THRESHOLD = 0.30
MAX_SYMPTOMS = 25

# Question focus -> trigger phrases (matched on normalised text).
FOCUS_KEYWORDS: dict[str, list[str]] = {
    "prevention": ["prevent", "prevention", "avoid", "precaution", "precautions", "protect", "stop getting", "reduce risk"],
    "treatment": [
        "treat", "treatment", "cure", "medicine", "medicines", "medication", "tablet", "tablets", "drug",
        "remedy", "remedies", "what to do", "what should i do", "get rid", "manage", "relief", "relieve", "heal",
    ],
    "symptoms": ["symptom", "symptoms", "sign", "signs", "how do i know", "how to know", "how to tell"],
    "doctor": ["which doctor", "specialist", "when to see", "see a doctor", "consult", "when should i worry"],
    "general": ["what is", "what are", "tell me about", "about", "explain", "information", "info", "define", "meaning", "know more"],
}

STARTER_SUGGESTIONS = [
    {"label": "I have fever and headache", "message": "I have fever and headache"},
    {"label": "How to prevent dengue?", "message": "How to prevent dengue?"},
    {"label": "Check my heart risk", "message": "Check my heart risk"},
    {"label": "What can you do?", "message": "What can you do?"},
]

FALLBACK_REPLIES = [
    "Sorry, I didn't quite get that. Could you describe your symptoms? For example: \"I have a sore throat and a mild fever\".",
    "I'm not sure I understood. You can tell me how you feel (\"stomach pain and vomiting\") or ask about a condition (\"what is asthma?\").",
]


@dataclass
class BotReply:
    type: str
    reply: str
    symptoms: list[dict] = field(default_factory=list)
    conditions: list[dict] = field(default_factory=list)
    info: dict | None = None
    suggestions: list[dict] = field(default_factory=list)
    alert: dict | None = None
    action: dict | None = None
    intent: dict | None = None

    def to_dict(self) -> dict:
        return dict(self.__dict__)


class MedBot:
    def __init__(self, kb: KnowledgeBase, intents: IntentModel, seed: int | None = None):
        self.kb = kb
        self.intents = intents
        self.checker = SymptomChecker(kb)
        self.rng = random.Random(seed)
        aliases: list[tuple[str, str]] = []
        for cond in kb.conditions:
            for alias in cond.aliases:
                aliases.append((normalize(alias), cond.id))
        aliases.sort(key=lambda a: -len(a[0]))
        self._aliases = [(re.compile(rf"\b{re.escape(a)}\b"), cid) for a, cid in aliases]

    # ----------------------------------------------------------------- helpers
    def _symptom_list(self, ids: list[str]) -> list[dict]:
        return [{"id": s, "label": self.kb.label(s)} for s in ids]

    def _focus(self, norm: str) -> str | None:
        padded = f" {norm} "
        for focus, words in FOCUS_KEYWORDS.items():
            if any(f" {w} " in padded for w in words):
                return focus
        return None

    def _mentioned_condition(self, norm: str) -> str | None:
        for pattern, cid in self._aliases:
            if pattern.search(norm):
                return cid
        return None

    def _condition_card(self, cid: str, score: float | None = None, matched: list[str] | None = None) -> dict:
        cond = self.kb.condition(cid)
        card = cond.to_dict()
        card.pop("aliases", None)
        card["symptom_labels"] = [self.kb.label(s) for s in cond.symptoms]
        if score is not None:
            card["match"] = round(min(score, 1.0) * 100)
            matched = matched or []
            implied = {SYMPTOM_PARENTS.get(s) for s in matched}
            card["matched"] = [self.kb.label(s) for s in matched if s not in implied]
        return card

    def _symptom_chips(self, ids: list[str]) -> list[dict]:
        return [
            {"label": f"+ {self.kb.label(s)}", "message": f"I also have {self.kb.label(s).lower()}"}
            for s in ids
        ]

    # -------------------------------------------------------------------- main
    def reply(self, message: str, known_symptoms: list[str] | None = None) -> BotReply:
        known = [s for s in (known_symptoms or []) if s in self.kb.symptoms][:MAX_SYMPTOMS]
        message = (message or "").strip()[:500]
        norm = normalize(message)

        if not norm:
            return BotReply("prompt", "Please type a message — for example, describe your symptoms.",
                            symptoms=self._symptom_list(known), suggestions=STARTER_SUGGESTIONS)

        focus = self._focus(norm)
        alert = safety.check_text(message, informational=focus is not None)
        if alert:
            return BotReply(alert.kind, alert.message, symptoms=self._symptom_list(known),
                            alert={"kind": alert.kind, "message": alert.message},
                            suggestions=[{"label": "Emergency numbers", "message": "emergency number"}])

        extraction = self.checker.extract(message)
        condition_id = self._mentioned_condition(norm)
        prediction = self.intents.predict(message)
        intent = {"tag": prediction.tag, "confidence": round(prediction.confidence, 3)}

        # Explicit reset always wins.
        if prediction.tag == "reset" and prediction.confidence >= 0.5 and not extraction.present:
            return BotReply("reset", self.intents.respond("reset", self.rng), symptoms=[],
                            suggestions=STARTER_SUGGESTIONS, intent=intent)

        # 1) Questions about a named condition: "how to prevent dengue?"
        if condition_id and (focus or not (extraction.present or extraction.negated)):
            return self._condition_info(condition_id, focus or "general", known, intent)

        # 2) New symptoms (or symptoms the user says they DON'T have).
        if extraction.present or (extraction.negated and set(extraction.negated) & set(known)):
            merged = [s for s in known if s not in extraction.negated]
            for s in extraction.present:
                if s not in merged:
                    merged.append(s)
            return self._analysis(merged[:MAX_SYMPTOMS], new=extraction.present,
                                  removed=[s for s in extraction.negated if s in known], intent=intent)

        # 3) Small talk / app features.
        if prediction.confidence >= INTENT_THRESHOLD:
            return self._intent_reply(prediction.tag, known, intent)

        # 4) Nothing matched.
        return BotReply("fallback", self.rng.choice(FALLBACK_REPLIES), symptoms=self._symptom_list(known),
                        suggestions=STARTER_SUGGESTIONS, intent=intent)

    def rerank(self, known_symptoms: list[str]) -> BotReply:
        """Re-run the analysis after the user edits their symptom list."""
        known = [s for s in known_symptoms if s in self.kb.symptoms][:MAX_SYMPTOMS]
        if not known:
            return BotReply("reset", "I've cleared your symptoms. How are you feeling?", suggestions=STARTER_SUGGESTIONS)
        reply = self._analysis(known, new=[], removed=[], intent=None)
        reply.reply = "Updated. " + reply.reply
        return reply

    # ---------------------------------------------------------------- branches
    def _condition_info(self, cid: str, focus: str, known: list[str], intent: dict) -> BotReply:
        cond = self.kb.condition(cid)
        lead = {
            "prevention": f"Here's how to lower your risk of {cond.name}:",
            "treatment": f"Here's how {cond.name} is usually managed:",
            "symptoms": f"Common symptoms of {cond.name}:",
            "doctor": f"When to get medical help for {cond.name} (specialist: {cond.specialist}):",
            "general": f"{cond.name} — {cond.summary}",
        }[focus]
        info = self._condition_card(cid)
        info["focus"] = focus
        others = [f for f in ("symptoms", "treatment", "prevention", "doctor") if f != focus]
        label = {"symptoms": "Symptoms", "treatment": "Treatment", "prevention": "Prevention", "doctor": "When to see a doctor"}
        query = {"symptoms": "symptoms of", "treatment": "treatment for", "prevention": "how to prevent", "doctor": "when to see a doctor for"}
        suggestions = [
            {"label": f"{label[f]}", "message": f"{query[f]} {cond.aliases[0]}"} for f in others
        ]
        alert = None
        if cond.urgency == "urgent":
            alert = {"kind": "warning", "message": cond.see_doctor[0]}
        return BotReply("info", lead, symptoms=self._symptom_list(known), info=info,
                        suggestions=suggestions, alert=alert, intent=intent)

    def _analysis(self, symptoms: list[str], new: list[str], removed: list[str], intent: dict) -> BotReply:
        if not symptoms:
            return BotReply("analysis", "Okay, noted. I don't have any symptoms recorded right now — tell me how you're feeling.",
                            symptoms=[], suggestions=STARTER_SUGGESTIONS, intent=intent)

        ranked = self.checker.rank(symptoms, top_k=3)
        cards = [self._condition_card(r.id, r.score, r.matched) for r in ranked]
        follow_ups = self.checker.follow_up_questions(ranked, symptoms)

        labels = ", ".join(self.kb.label(s).lower() for s in symptoms)
        parts = []
        if removed:
            parts.append("Got it, I've removed " + ", ".join(self.kb.label(s).lower() for s in removed) + ".")
        if len(symptoms) == 1:
            parts.append(f"I've noted: {labels}. One symptom alone can mean many things — "
                         "do you have anything else? Here's what it could be related to:")
        else:
            parts.append(f"Based on your symptoms ({labels}), these conditions match best:")

        alert = None
        cardiac = safety.check_symptoms(self.checker.expand(symptoms))
        if cardiac:
            alert = {"kind": "emergency", "message": cardiac.message}
        elif cards and cards[0]["urgency"] == "urgent":
            alert = {"kind": "warning", "message": cards[0]["see_doctor"][0]}

        suggestions = self._symptom_chips(follow_ups)
        suggestions.append({"label": "No other symptoms", "message": "no other symptoms"})
        suggestions.append({"label": "Start over", "message": "start over"})
        return BotReply("analysis", " ".join(parts), symptoms=self._symptom_list(symptoms), conditions=cards,
                        suggestions=suggestions, alert=alert, intent=intent)

    def _intent_reply(self, tag: str, known: list[str], intent: dict) -> BotReply:
        if tag == "deny" and known:
            ranked = self.checker.rank(known, top_k=1)
            if ranked:
                top = self.kb.condition(ranked[0].id)
                text = (f"Thanks. Your symptoms most closely match {top.name}. "
                        f"I'd suggest seeing a {top.specialist} if they persist or get worse. "
                        "Remember, this is not a diagnosis.")
                return BotReply("summary", text, symptoms=self._symptom_list(known),
                                conditions=[self._condition_card(top.id, ranked[0].score, ranked[0].matched)],
                                suggestions=[{"label": "Find a doctor", "message": "find a doctor"},
                                             {"label": "Start over", "message": "start over"}],
                                intent=intent)

        reply = BotReply(tag, self.intents.respond(tag, self.rng), symptoms=self._symptom_list(known), intent=intent)
        if tag == "heart_check":
            reply.action = {"label": "Open Heart Risk check", "href": "/heart"}
        if tag in {"greeting", "capabilities", "affirm", "feeling_unwell", "how_are_you", "name"}:
            reply.suggestions = STARTER_SUGGESTIONS
        return reply
