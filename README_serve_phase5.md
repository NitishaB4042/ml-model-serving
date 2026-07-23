# ML Model Serving & Monitoring — Phase 5 (Final)

Drift detection — the MLOps headline. A model silently degrades when live inputs
stop looking like its training data. This phase detects that shift and alerts,
before anyone notices the model has quietly gone wrong.

## Files
- `drift_phase5.py` — PSI drift measure + `DriftDetector` + demo
- `test_serve_phase5.py` — tests
- `serving_drift.png` — PSI over time, crossing the alert line as inputs shift
- `serving_drift_results.json` — the raw drift scores

![Drift detection](serving_drift.png)

## What drift is
A model trained on yesterday's data gets worse when today's inputs differ —
new slang, a changed user base, a broken upstream feed. The model doesn't throw
an error; it just becomes quietly less accurate. Detecting this is a core
ML-engineering responsibility.

## How we detect it: PSI
We compare the **live** distribution of a tracked quantity (here, model
confidence) against a fixed **baseline** captured from reference data, using the
**Population Stability Index (PSI)** — a standard, defensible drift measure:

```
PSI = sum over bins of (live% - base%) * ln(live% / base%)
```
Industry rule-of-thumb bands:
- PSI < 0.10 → no significant drift
- 0.10–0.25 → moderate drift (watch)
- ≥ 0.25 → significant drift (**alert**)

`DriftDetector` holds the baseline and scores each live window:
```python
det = DriftDetector(baseline_confidences, alert_threshold=0.25)
det.score(live_window)   # -> {psi, level, alert, n}
```

## The result
In the demo, windows 0–9 match the baseline (PSI ~0.05–0.13, no alert). From
window 10 the input distribution drifts; PSI immediately crosses 0.25 and the
alert fires, climbing as drift worsens. The chart shows PSI hugging zero, then
shooting past the alert line exactly when inputs start shifting.

## The crucial framing (say this in interviews)
Detecting drift in the **inputs** is **not** the same as proving the model's
**accuracy** dropped — in production you usually have no live ground-truth
labels to measure accuracy directly. Input/confidence drift is an **early
warning** that accuracy *may* be degrading, which prompts investigation,
relabelling, or retraining. Conflating the two is a common mistake; being
precise about it shows real understanding.

## Run
```bash
pip install matplotlib
python drift_phase5.py        # streams stable-then-drifting data, charts PSI
python test_serve_phase5.py   # tests
```

## Tests cover
- PSI ≈ 0 for identical and same-distribution data (**no false alarms**)
- PSI large for a clearly shifted distribution (**catches real drift**)
- the classification bands (none / moderate / significant)
- the detector stays quiet when stable and alerts on drift
- empty live window → no drift; empty baseline → rejected
- the histogram fractions sum to 1

The sensitive-vs-specific pair (catches real drift, ignores noise) is what makes
a drift detector trustworthy rather than a nuisance.

## The complete project, recapped
| Phase | Adds | Key idea |
|-------|------|----------|
| 1 | prediction API | model loaded once, behind a pluggable interface |
| 2 | validation & abstain | reject bad input; refuse when unsure |
| 3 | batching | amortize per-call cost — measured ~12x throughput |
| 4 | metrics | throughput, p95/p99 latency, prediction mix |
| 5 | drift detection | PSI baseline comparison with alerting |
