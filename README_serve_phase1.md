# ML Model Serving & Monitoring — Phase 1

A clean prediction API. Wraps a trained model behind FastAPI, loaded once at
startup and called per request, with the model behind a pluggable interface.

## Files
- `serve_phase1.py` — the pluggable Model interface, a stub classifier, the API
- `test_serve_phase1.py` — tests (drive the real app in-process, no server)

## What it does
```
POST /predict   { "text": "..." }
   -> { "label": "fake", "confidence": 0.9992, "model": "stub-text-clf-v1" }

GET  /health
   -> { "status": "ok", "model": "...", "labels": ["real", "fake"] }
```

## The pluggable model (the key design choice)
A model is anything with `.predict(text) -> (label, confidence)` (and an
optional `.predict_batch`). Two ship here:

- **StubTextClassifier** — dependency-free, deterministic. Mimics a fake-news
  detector by scoring sensational signal words. It exists so the *serving
  infrastructure* (API, validation, batching, metrics, drift in later phases)
  can be built and tested anywhere, with no heavy ML libraries.
- **TransformerClassifier** — the real model (commented out at the top of
  `serve_phase1.py`). On Colab: `pip install transformers torch`, uncomment it,
  point it at your fine-tuned DistilBERT, and return it from `load_model()`.
  Nothing else in the service changes.

> Why a stub here? Your real models (DistilBERT, EfficientNet) are large
> PyTorch models that need a GPU/Colab. The serving, batching, metrics, and
> drift logic are **model-agnostic** — so the right move is to build and test
> them against a lightweight stand-in, then plug in the real model. The
> infrastructure *is* the contribution of this project. (Same decoupling as the
> RAG embedder and LLM.)

## Run
```bash
pip install fastapi "uvicorn[standard]" httpx
uvicorn serve_phase1:app --port 8000
```
```bash
curl -X POST localhost:8000/predict -H 'content-type: application/json' \
  -d '{"text":"SHOCKING secret cure they do not want you to see!!!"}'
# {"label":"fake","confidence":0.9992,"model":"stub-text-clf-v1"}
```

## Discipline introduced in Phase 1
- **Load once at startup** via FastAPI `lifespan` — model loading is slow, so
  you never reload per request.
- **Model behind an interface** — a stub and the real model are interchangeable.
- **Validated input** — empty or missing text is rejected (422) by Pydantic.

## Tests cover
- health reports the model name and labels
- predict returns a valid label + confidence in [0, 1]
- clickbait is classified 'fake' with high confidence (sanity of the stub)
- empty text and missing field are rejected (422)
- predictions are deterministic (same input -> same output)
- the model is genuinely pluggable (a custom model drops straight in)

## Still deferred
- Strict validation + low-confidence "abstain" → **Phase 2**
- Request batching for throughput → **Phase 3**
- Metrics + monitoring dashboard (p50/p95/p99, prediction mix) → **Phase 4**
- Drift detection + chart → **Phase 5**
