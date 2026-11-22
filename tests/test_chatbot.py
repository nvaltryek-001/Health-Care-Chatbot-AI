import pytest

from app.knowledge import load_knowledge_base
from app.symptom_checker import SymptomChecker


@pytest.fixture(scope="module")
def checker():
    return SymptomChecker(load_knowledge_base())


@pytest.mark.parametrize(
    "text, expected",
    [
        ("I have a headache and a runny nose", {"headache", "runny_nose"}),
        ("high fever with chills", {"high_fever", "chills"}),
        ("pain behind my eyes", {"pain_behind_eyes"}),
        ("diarhea since morning", {"diarrhea"}),  # typo tolerance
        ("I can't sleep and feel anxious", {"sleep_problems", "worry"}),
    ],
)
def test_extracts_symptoms(checker, text, expected):
    assert set(checker.extract(text).present) == expected


def test_negation(checker):
    result = checker.extract("sneezing and runny nose but no fever")
    assert "fever" in result.negated
    assert "fever" not in result.present
    assert {"sneezing", "runny_nose"} <= set(result.present)


def test_negation_scope_ends_at_but(checker):
    result = checker.extract("no cough but headache")
    assert result.negated == ["cough"]
    assert result.present == ["headache"]


@pytest.mark.parametrize(
    "symptoms, top",
    [
        (["high_fever", "joint_pain", "pain_behind_eyes", "rash"], "dengue"),
        (["painful_urination", "frequent_urination"], "uti"),
        (["increased_thirst", "frequent_urination", "blurred_vision", "fatigue"], "diabetes"),
        (["throbbing_headache", "light_sensitivity", "nausea"], "migraine"),
        (["diarrhea", "vomiting", "abdominal_pain"], "gastroenteritis"),
        (["sadness", "loss_of_interest", "sleep_problems"], "depression"),
        (["wheezing", "shortness_of_breath", "chest_tightness"], "asthma"),
        (["sneezing", "itchy_eyes", "runny_nose"], "allergic_rhinitis"),
    ],
)
def test_ranking(checker, symptoms, top):
    assert checker.rank(symptoms)[0].id == top


def test_greeting(bot):
    r = bot.reply("hello")
    assert r.type == "greeting"
    assert r.suggestions


def test_symptom_conversation_accumulates(bot):
    r1 = bot.reply("I have a fever and body ache")
    assert r1.type == "analysis"
    ids = [s["id"] for s in r1.symptoms]
    r2 = bot.reply("I also have chills", ids)
    assert {"fever", "body_aches", "chills"} <= {s["id"] for s in r2.symptoms}
    r3 = bot.reply("actually no fever", [s["id"] for s in r2.symptoms])
    assert "fever" not in {s["id"] for s in r3.symptoms}


def test_condition_question(bot):
    r = bot.reply("how to prevent dengue?")
    assert r.type == "info"
    assert r.info["id"] == "dengue" and r.info["focus"] == "prevention"


def test_reset(bot):
    r = bot.reply("start over", ["fever", "cough"])
    assert r.type == "reset" and r.symptoms == []


def test_crisis_is_detected(bot):
    r = bot.reply("I want to kill myself")
    assert r.type == "crisis"
    assert "14416" in r.reply


def test_emergency_is_detected(bot):
    assert bot.reply("my father is unconscious").type == "emergency"


def test_cardiac_combination_raises_alert(bot):
    r = bot.reply("chest pain and sweating with pain in my left arm")
    assert r.alert and r.alert["kind"] == "emergency"
    assert r.conditions[0]["id"] == "heart_disease"


def test_information_question_is_not_an_emergency(bot):
    r = bot.reply("how to prevent heart attack")
    assert r.type == "info"


def test_gibberish_falls_back(bot):
    assert bot.reply("qwrtzxv plmok").type == "fallback"


def test_unknown_symptom_ids_are_ignored(bot):
    r = bot.reply("hello", ["not_a_symptom"])
    assert r.symptoms == []
