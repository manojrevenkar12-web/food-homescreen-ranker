from fastapi.testclient import TestClient

from fhr.serve.app import Replica, create_app


def client(small, **kw):
    city, world, _, bundle = small
    return TestClient(create_app(Replica(city, world.fork(), bundle), **kw))


def test_sagemaker_contract(small):
    c = client(small)
    assert c.get("/ping").status_code == 200
    r = c.post("/invocations", json={"customer": 3, "minute": 9 * 1440 + 19 * 60})
    assert r.status_code == 200
    body = r.json()
    assert body["policy"] == "page" and body["rows"]
    assert all(len(row["merchants"]) <= 8 for row in body["rows"])


def test_unknown_customer(small):
    assert client(small).post("/v1/home", json={"customer": 10**7, "minute": 0}).status_code == 404


def test_falls_back_to_todays_page_when_the_model_fails(small, monkeypatch):
    from fhr.policies import learned

    def broken(*a, **k):
        raise RuntimeError("feature store timeout")

    monkeypatch.setattr(learned.PageAssembler, "__call__", broken)
    c = client(small)
    body = c.post("/v1/home", json={"customer": 3, "minute": 9 * 1440 + 12 * 60}).json()
    assert body["policy"] == "current" and body["rows"]
    assert 'home_fallbacks_total{reason="error"} 1.0' in c.get("/metrics").text
