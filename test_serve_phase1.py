"""
Tests for Model Serving Phase 1 — the prediction API.
Drives the real FastAPI app in-process via httpx ASGITransport (no server).
Run: python test_serve_phase1.py
"""
import asyncio
import httpx
import serve_phase1 as svc


async def with_app(body):
    async with svc.app.router.lifespan_context(svc.app):
        transport = httpx.ASGITransport(app=svc.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            await body(c)


def test_health_reports_model():
    async def body(c):
        r = await c.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert data["model"] == "stub-text-clf-v1"
        assert data["labels"] == ["real", "fake"]
    asyncio.run(with_app(body))
    print("  health_reports_model: PASS")


def test_predict_returns_label_and_confidence():
    async def body(c):
        r = await c.post("/predict", json={"text": "ordinary news about the city budget"})
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["label"] in ("real", "fake")
        assert 0.0 <= data["confidence"] <= 1.0
        assert "model" in data
    asyncio.run(with_app(body))
    print("  predict_returns_label_and_confidence: PASS")


def test_clickbait_classified_fake():
    async def body(c):
        r = await c.post("/predict",
                         json={"text": "SHOCKING miracle cure exposed!!! you won't believe"})
        data = r.json()
        assert data["label"] == "fake", data
        assert data["confidence"] > 0.7
    asyncio.run(with_app(body))
    print("  clickbait_classified_fake: PASS")


def test_empty_text_rejected():
    async def body(c):
        r = await c.post("/predict", json={"text": ""})
        assert r.status_code == 422, r.status_code   # pydantic min_length
    asyncio.run(with_app(body))
    print("  empty_text_rejected: PASS")


def test_missing_field_rejected():
    async def body(c):
        r = await c.post("/predict", json={})
        assert r.status_code == 422, r.status_code
    asyncio.run(with_app(body))
    print("  missing_field_rejected: PASS")


def test_deterministic():
    # same input -> same output (important for a serving stub)
    async def body(c):
        a = await c.post("/predict", json={"text": "the same sentence twice"})
        b = await c.post("/predict", json={"text": "the same sentence twice"})
        assert a.json() == b.json()
    asyncio.run(with_app(body))
    print("  deterministic: PASS")


def test_model_is_pluggable():
    # a custom model with the same interface should drop straight in
    class AlwaysReal(svc.Model):
        name = "always-real"; labels = ("real", "fake")
        def predict(self, text): return ("real", 0.99)
    saved = svc.load_model
    svc.load_model = lambda: AlwaysReal()
    async def body(c):
        r = await c.post("/predict", json={"text": "anything at all"})
        assert r.json()["label"] == "real" and r.json()["model"] == "always-real"
    try:
        asyncio.run(with_app(body))
    finally:
        svc.load_model = saved
    print("  model_is_pluggable: PASS")


if __name__ == "__main__":
    print("Running Model Serving Phase 1 tests:")
    test_health_reports_model()
    test_predict_returns_label_and_confidence()
    test_clickbait_classified_fake()
    test_empty_text_rejected()
    test_missing_field_rejected()
    test_deterministic()
    test_model_is_pluggable()
    print("All tests passed.")
