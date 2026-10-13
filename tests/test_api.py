from fastapi.testclient import TestClient
from app.main import app

c = TestClient(app)


def test_health():
    assert c.get("/health").json()["status"] == "ok"


def test_predict_is_antisymmetric():
    names = [p["name"] for p in c.get("/players").json()[:2]]
    a, b = names
    pab = c.get("/predict", params={"a": a, "b": b}).json()["p_a"]
    pba = c.get("/predict", params={"a": b, "b": a}).json()["p_a"]
    assert abs(pab + pba - 1) < 1e-3


def test_errors():
    assert c.get("/predict", params={"a": "Nobody X.", "b": "Y Z."}).status_code == 404
    name = c.get("/players").json()[0]["name"]
    assert c.get("/predict", params={"a": name, "b": name}).status_code == 400


def test_predict_accepts_query_params_as_strings():
    a, b = [p["name"] for p in c.get("/players").json()[:2]]
    r = c.get("/predict", params={"a": a, "b": b, "surface": "Clay", "best_of": "3"})
    assert r.status_code == 200
    assert c.get("/predict", params={"a": a, "b": b, "best_of": "4"}).status_code == 400
