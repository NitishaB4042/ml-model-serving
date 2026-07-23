"""
ML Model Serving & Monitoring — Phase 1: a clean prediction API.

Wraps a trained model behind a FastAPI service. The model is loaded ONCE at
startup and called per request, behind a small pluggable Model interface so a
stub and your real model are interchangeable.

Design choice (same pattern as the RAG embedder/LLM): the Model is PLUGGABLE.
Two ship here:

  - StubTextClassifier : dependency-free, deterministic. Mimics a binary text
                         classifier (e.g. a fake-news detector) so the whole
                         serving stack can be built and tested anywhere.
  - TransformerClassifier : the real model for Colab (commented out). Wraps your
                         fine-tuned DistilBERT. Same interface, nothing else
                         changes.

Run the server:  uvicorn serve_phase1:app --port 8000
Run the tests:   python test_phase1.py
"""

import re
import math
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI
from pydantic import BaseModel, Field


# ===========================================================================
# Model interface — anything with .name, .labels, and .predict(text) -> (label, conf)
# ===========================================================================
class Model:
    name = "base"
    labels = ("negative", "positive")
    def predict(self, text: str) -> tuple[str, float]:
        raise NotImplementedError
    def predict_batch(self, texts: list[str]) -> list[tuple[str, float]]:
        # default: loop; a real model overrides this to run a true batch
        return [self.predict(t) for t in texts]


class StubTextClassifier(Model):
    """A deterministic stand-in for a binary text classifier.

    Mimics a 'fake vs real news' style detector: it scores text by simple
    signal words and length, then squashes to a probability. NOT a real model
    \u2014 it exists so the serving infrastructure (API, validation, batching,
    metrics, drift) can be built and tested with zero heavy dependencies.
    Swap in TransformerClassifier for the real predictions.
    """
    name = "stub-text-clf-v1"
    labels = ("real", "fake")

    # toy signal: sensational words push toward 'fake'
    FAKE_SIGNALS = {"shocking", "miracle", "exposed", "secret", "unbelievable",
                    "cure", "hoax", "you", "won't", "believe", "click", "!!!"}

    def _score(self, text: str) -> float:
        words = re.findall(r"[a-z']+|!{2,}", text.lower())
        if not words:
            return 0.5
        hits = sum(1 for w in words if w in self.FAKE_SIGNALS)
        exclam = text.count("!")
        raw = hits * 1.2 + exclam * 0.4 - len(words) * 0.01
        return 1.0 / (1.0 + math.exp(-raw))     # sigmoid -> probability of 'fake'

    def predict(self, text: str) -> tuple[str, float]:
        p_fake = self._score(text)
        if p_fake >= 0.5:
            return "fake", p_fake
        return "real", 1.0 - p_fake


# Real model for Colab. Uncomment and `pip install transformers torch`:
#
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import torch
# class TransformerClassifier(Model):
#     name = "distilbert-fakenews"
#     labels = ("real", "fake")
#     def __init__(self, model_dir):
#         self.tok = AutoTokenizer.from_pretrained(model_dir)
#         self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
#         self.model.eval()
#     def predict(self, text):
#         return self.predict_batch([text])[0]
#     def predict_batch(self, texts):
#         enc = self.tok(texts, return_tensors="pt", padding=True, truncation=True)
#         with torch.no_grad():
#             probs = self.model(**enc).logits.softmax(-1)
#         out = []
#         for row in probs:
#             idx = int(row.argmax())
#             out.append((self.labels[idx], float(row[idx])))
#         return out


# ===========================================================================
# The service
# ===========================================================================
_state: dict = {}


def load_model() -> Model:
    """Load the model once. Swap to TransformerClassifier(...) on Colab."""
    return StubTextClassifier()


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["model"] = load_model()        # loaded ONCE at startup
    yield
    _state.clear()


app = FastAPI(title="ML Model Serving", lifespan=lifespan)


class PredictRequest(BaseModel):
    text: str = Field(min_length=1)


@app.get("/health")
async def health():
    model = _state.get("model")
    if model is None:
        return {"status": "starting"}
    return {"status": "ok", "model": model.name, "labels": list(model.labels)}


@app.post("/predict")
async def predict(req: PredictRequest):
    model = _state["model"]
    label, confidence = model.predict(req.text)
    return {"label": label, "confidence": round(confidence, 4), "model": model.name}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
