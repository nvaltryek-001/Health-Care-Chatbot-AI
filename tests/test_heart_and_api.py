from app.heart_model import load_dataset

HIGH = {"age": 64, "sex": 1, "cp": 0, "trestbps": 158, "chol": 286, "fbs": 1, "restecg": 0,
        "thalach": 112, "exang": 1, "oldpeak": 2.4, "slope": 1, "ca": 2, "thal": 3}
LOW = {"age": 38, "sex": 0, "cp": 2, "trestbps": 118, "chol": 195, "fbs": 0, "restecg": 1,
       "thalach": 176, "exang": 0, "oldpeak": 0, "slope": 2, "ca": 0, "thal": 2}


def test_dataset_label_orientation():
    df = load_dataset()
    sick, healthy = df[df.has_disease == 1], df[df.has_disease == 0]
    # Medically, patients with disease reach lower max heart rates and have more exercise angina.
    assert sick.thalach.mean() < healthy.thalach.mean()
    assert sick.exang.mean() > healthy.exang.mean()


def test_model_quality(heart):
    assert heart.metrics["test"]["roc_auc"] > 0.85
    assert heart.metrics["test"]["accuracy"] > 0.78


def test_model_orders_risk(heart):
    assert heart.predict(HIGH).probability > 0.7
    assert heart.predict(LOW).probability < 0.3


def test_pages_render(client):
    for url in ["/", "/heart", "/insights", "/conditions", "/about"]:
        assert client.get(url).status_code == 200, url


def test_health(client):
    assert client.get("/api/health").get_json()["status"] == "ok"


def test_chat_api(client):
    res = client.post("/api/chat", json={"message": "sore throat and fever", "symptoms": []})
    data = res.get_json()
    assert res.status_code == 200
    assert data["type"] == "analysis" and data["conditions"]


def test_chat_api_rerank(client):
    data = client.post("/api/chat", json={"action": "rerank", "message": "", "symptoms": ["fever", "cough"]}).get_json()
    assert {s["id"] for s in data["symptoms"]} == {"fever", "cough"}


def test_chat_api_validation(client):
    assert client.post("/api/chat", json={"message": 5}).status_code == 400
    assert client.post("/api/chat", json={"message": "hi", "symptoms": "fever"}).status_code == 400


def test_heart_api(client):
    data = client.post("/api/heart/predict", json=HIGH).get_json()
    assert data["level"] == "High" and data["factors"]


def test_heart_api_validation(client):
    res = client.post("/api/heart/predict", json={**LOW, "age": 300, "cp": 9})
    assert res.status_code == 400
    assert set(res.get_json()["fields"]) == {"age", "cp"}


def test_insights_api(client):
    data = client.get("/api/insights").get_json()
    assert data["heart"]["total"] == 302
    assert data["knowledge_base"]["conditions"] >= 25
