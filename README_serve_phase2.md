# ML Model Serving & Monitoring — Phase 2

Hardens the prediction API: strict input validation, graceful errors, and a
confidence threshold so the service **abstains** instead of returning a shaky
guess.

## Files
- `serve_phase2.py` — validation + the abstain logic (reuses Phase 1's model)
- `test_serve_phase2.py` — tests

## What's new vs Phase 1
| Phase 1 | Phase 2 |
|---|---|
| accepts any non-empty text | rejects blank, whitespace-only, and over-long text |
| always returns a label | **abstains** ("low_confidence") when unsure |
| fixed behaviour | per-request confidence `threshold` |

## The abstain behaviour (the ML-specific part)
A mature ML system knows when *not* to answer. Each prediction has a confidence;
if it falls below a threshold (default 0.60), the service returns:

```json
{
  "label": "low_confidence",
  "predicted_label": "real",   // what it leaned toward, for transparency
  "confidence": 0.5125,
  "threshold": 0.6,
  "abstained": true
}
```
A confident prediction returns normally:
```json
{ "label": "fake", "confidence": 0.9998, "abstained": false }
```
The caller can raise or lower `threshold` per request to trade coverage for
precision — strict (high threshold) abstains more but is surer when it answers.

## Validation
- empty / whitespace-only text → 422
- text longer than `MAX_TEXT_LEN` (5000) → 422
- `threshold` outside [0, 1] → 422
- missing `text` → 422

All enforced by Pydantic before the model runs — bad input never reaches the
model or produces a garbage prediction.

## Run
```bash
uvicorn serve_phase2:app --port 8000
curl -X POST localhost:8000/predict -H 'content-type: application/json' \
  -d '{"text":"the quarterly report is attached"}'
# -> low_confidence (abstained), leaned "real" at 0.51
```

## Test
```bash
python test_serve_phase2.py
```
Covers: high-confidence returns a label, low-confidence abstains, the threshold
is configurable, and the four validation rejections.

## Interview notes
- **Abstaining is a feature, not a failure.** In real systems, a wrong confident
  answer is often worse than "I'm not sure" — e.g. routing uncertain cases to a
  human. The threshold is the precision/coverage dial.
- **Validate before the model.** Cheap, clear rejection of bad input protects
  both the model and the caller, and keeps your metrics clean.

## Still deferred
- Request batching for throughput → **Phase 3**
- Metrics + dashboard (p50/p95/p99, prediction mix) → **Phase 4**
- Drift detection + chart → **Phase 5**
