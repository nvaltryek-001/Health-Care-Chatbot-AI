"""Red-flag detection. Runs before anything else so emergencies are never missed."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .text import normalize

EMERGENCY_NUMBER = "112"
MENTAL_HEALTH_LINE = "Tele-MANAS 14416"


@dataclass
class SafetyAlert:
    kind: str  # "crisis" | "emergency"
    message: str


_CRISIS = [
    r"\bsuicid\w*", r"\bkill (myself|me)\b", r"\bend (my|it) (life|all)\b", r"\bwant to die\b",
    r"\bwanna die\b", r"\bself harm\w*", r"\bhurt(ing)? myself\b", r"\bcut(ting)? myself\b",
    r"\bno reason to live\b", r"\bbetter off dead\b", r"\btake my (own )?life\b",
]

_EMERGENCY = [
    r"\bheart attack\b",
    r"(?<!heat )(?<!sun )\bstroke\b", r"\bunconscious\b", r"\bnot breathing\b",
    r"\bcannot breathe\b", r"\bcant breathe\b", r"\bunable to breathe\b", r"\bchoking\b",
    r"\bseizure\w*\b", r"\bfits\b", r"\bconvulsion\w*", r"\bsevere bleeding\b",
    r"\bbleeding (heavily|a lot|wont stop|will not stop)\b", r"\boverdose\w*\b",
    r"\bpoison\w*\b", r"\bswallowed (bleach|poison|pills)\b", r"\bface (is )?drooping\b",
    r"\bslurred speech\b", r"\bcoughing (up )?blood\b", r"\bvomiting blood\b",
    r"\bsnake bite\b", r"\bsnakebite\b", r"\banaphyla\w*", r"\b(lips|face) (are |is )?(turning )?blue\b",
    r"\bsevere burn\w*\b",
]

_CRISIS_RE = re.compile("|".join(_CRISIS))
_EMERGENCY_RE = re.compile("|".join(_EMERGENCY))

CRISIS_MESSAGE = (
    "I'm really sorry you're feeling this way. You don't have to go through it alone, "
    "and talking to someone can help right now.\n\n"
    f"• Call {MENTAL_HEALTH_LINE} (India, free, 24×7) or KIRAN 1800-599-0019\n"
    f"• If you are in immediate danger, call {EMERGENCY_NUMBER}\n"
    "• Outside India, find a local helpline at findahelpline.com\n\n"
    "If you can, reach out to someone you trust and let them know how you're feeling. "
    "I'm here to keep talking too."
)

EMERGENCY_MESSAGE = (
    "This sounds like a medical emergency. Please call "
    f"{EMERGENCY_NUMBER} (or 108 for an ambulance in India) or go to the nearest emergency "
    "department right now. Don't wait for symptoms to get better, and don't drive yourself if "
    "you feel faint."
)

CARDIAC_MESSAGE = (
    "Chest pain together with sweating, breathlessness or pain spreading to the arm or jaw can be "
    f"a sign of a heart attack. Call {EMERGENCY_NUMBER} / 108 now, stop all activity and sit down. "
    "Don't drive yourself to hospital."
)


def check_text(text: str, informational: bool = False) -> SafetyAlert | None:
    """``informational`` questions ("how to prevent a stroke") skip emergency
    keywords but never skip the crisis check."""
    norm = normalize(text)
    if _CRISIS_RE.search(norm):
        return SafetyAlert("crisis", CRISIS_MESSAGE)
    if not informational and _EMERGENCY_RE.search(norm):
        return SafetyAlert("emergency", EMERGENCY_MESSAGE)
    return None


def check_symptoms(symptoms: set[str]) -> SafetyAlert | None:
    """Dangerous combinations of otherwise ordinary symptoms."""
    if "chest_pain" in symptoms and symptoms & {"arm_jaw_pain", "sweating", "shortness_of_breath", "fainting"}:
        return SafetyAlert("emergency", CARDIAC_MESSAGE)
    return None
