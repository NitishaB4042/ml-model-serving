"""
Tests for Model Serving Phase 2 — validation & confidence-based abstain.
Run: python test_serve_phase2.py
"""
import asyncio
import httpx
import serve_phase2 as svc


async def with_app(body):
    async with svc.app.router.lifespan_context(svc.app):
        transport = httpx.ASGITransport(app=svc.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            await body(c)


def test_high_confidence_returns_label():
    async def body(c):
        # clickbait scores very high -> well above threshold -> real label returned
        r = await c.post("/predict",
                         json={"text": "SHOCKING miracle cure exposed!!! you won't believe"})
        data = r.json()
        assert data["abstained"] is False, data
        assert data["label"] == "fake"
        assert data["confidence"] >= data["threshold"]
    asyncio.run(with_app(body))
    print("  high_confidence_returns_label: PASS")


def test_low_confidence_abstains():
    async def body(c):
        # an ordinary, signal-free sentence scores near 0.5 -> below 0.60 -> abstain
        r = await c.post("/predict", json={"text": "the meeting is on tuesday afternoon"})
        data = r.json()
        assert data["abstained"] is True, data
        assert data["label"] == "low_confidence"
        assert "predicted_label" in data        # still tells you what it leaned toward
    asyncio.run(with_app(body))
    print("  low_confidence_abstains: PASS")


def test_threshold_is_configurable():
    async def body(c):
        text = "the meeting is on tuesday afternoon"
        # with a very low threshold, the same borderline input is NOT abstained
        r = await c.post("/predict", json={"text": text, "threshold": 0.05})
        assert r.json()["abstained"] is False, r.json()
        # with a very high threshold, even confident inputs abstain
        r2 = await c.post("/predict",
                          json={"text": "SHOCKING miracle cure!!!", "threshold": 0.999999})
        assert r2.json()["abstained"] is True, r2.json()
    asyncio.run(with_app(body))
    print("  threshold_is_configurable: PASS")


def test_blank_text_rejected():
    async def body(c):
        r = await c.post("/predict", json={"text": "    "})
        assert r.status_code == 422, r.status_code     # whitespace-only blocked
    asyncio.run(with_app(body))
    print("  blank_text_rejected: PASS")


def test_too_long_rejected():
    async def body(c):
        r = await c.post("/predict", json={"text": "x" * (svc.MAX_TEXT_LEN + 1)})
        assert r.status_code == 422, r.status_code
    asyncio.run(with_app(body))
    print("  too_long_rejected: PASS")


def test_bad_threshold_rejected():
    async def body(c):
        r = await c.post("/predict", json={"text": "hello there", "threshold": 1.5})
        assert r.status_code == 422, r.status_code     # must be in [0,1]
    asyncio.run(with_app(body))
    print("  bad_threshold_rejected: PASS")


def test_missing_text_rejected():
    async def body(c):
        r = await c.post("/predict", json={"threshold": 0.5})
        assert r.status_code == 422, r.status_code
    asyncio.run(with_app(body))
    print("  missing_text_rejected: PASS")


if __name__ == "__main__":
    print("Running Model Serving Phase 2 tests:")
    test_high_confidence_returns_label()
    test_low_confidence_abstains()
    test_threshold_is_configurable()
    test_blank_text_rejected()
    test_too_long_rejected()
    test_bad_threshold_rejected()
    test_missing_text_rejected()
    print("All tests passed.")
