import json

from app.knowledge import DATA_DIR, load_knowledge_base


def test_every_condition_symptom_exists():
    kb = load_knowledge_base()
    for cond in kb.conditions:
        for sid in cond.symptoms:
            assert sid in kb.symptoms, f"{cond.id} references unknown symptom {sid}"


def test_conditions_are_complete():
    kb = load_knowledge_base()
    ids = [c.id for c in kb.conditions]
    assert len(ids) == len(set(ids)), "duplicate condition ids"
    for cond in kb.conditions:
        assert cond.urgency in {"self-care", "see-doctor", "urgent"}
        for section in ("self_care", "medicines", "see_doctor", "prevention", "aliases"):
            assert getattr(cond, section), f"{cond.id}.{section} is empty"


def test_intents_file_is_valid():
    data = json.loads((DATA_DIR / "intents.json").read_text(encoding="utf-8"))
    tags = [i["tag"] for i in data["intents"]]
    assert len(tags) == len(set(tags))
    for intent in data["intents"]:
        assert intent["patterns"] and intent["responses"]
