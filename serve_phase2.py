"""
ML Model Serving & Monitoring — Phase 2: validation & honest confidence.

Real inference services fail ugly on bad input and overstate shaky predictions.
Phase 2 hardens the door:

  - strict input validation (type, length, content) with clear error responses
  - a confidence THRESHOLD: when the model isn't sure, the service ABSTAINS
    ("low_confidence") instead of returning a coin-flip label
  - configurable threshold per request

Builds on Phase 1's pluggable Model. Run:
    uvicorn serve_phase2:app --port 8000
Tests:
    python test_phase2.py
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

# reuse the model interface + stub from Phase 1
from serve_phase1 import Model, StubTextClassifier


_state: dict = {}

# default: if the winning label's confidence is below this, abstain.
DEFAULT_THRESHOLD = 0.60
MAX_TEXT_LEN = 5000


def load_model() -> Model:
    return StubTextClassifier()


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["model"] = load_model()
    yield
    _state.clear()


app = FastAPI(title="ML Model Serving (Phase 2)", lifespan=lifespan)


class PredictRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_TEXT_LEN)
    threshold: float = Field(default=DEFAULT_THRESHOLD, ge=0.0, le=1.0)

    @field_validator("text")
    @classmethod
    def not_blank(cls, v: str) -> str:
        # reject strings that are only whitespace (min_length alone allows " ")
        if not v.strip():
            raise ValueError("text must not be blank")
        return v


@app.get("/health")
async def health():
    model = _state.get("model")
    if model is None:
        return JSONResponse(status_code=503, content={"status": "starting"})
    return {"status": "ok", "model": model.name, "labels": list(model.labels)}


@app.post("/predict")
async def predict(req: PredictRequest):
    model = _state["model"]
    label, confidence = model.predict(req.text)
    confidence = round(float(confidence), 4)

    # honest confidence: abstain rather than return a shaky guess
    if confidence < req.threshold:
        return {
            "label": "low_confidence",
            "predicted_label": label,        # what it leaned toward, for transparency
            "confidence": confidence,
            "threshold": req.threshold,
            "abstained": True,
            "model": model.name,
        }
    return {
        "label": label,
        "confidence": confidence,
        "threshold": req.threshold,
        "abstained": False,
        "model": model.name,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
