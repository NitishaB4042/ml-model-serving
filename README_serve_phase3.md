# ML Model Serving & Monitoring — Phase 3

Dynamic request **batching** — the signature ML-serving optimization. Collect
requests arriving within a few milliseconds, run the model once on the whole
batch, and route each result back to its caller.

## Files
- `batch_phase3.py` — the `BatchedModel` wrapper + a throughput demo
- `test_serve_phase3.py` — tests

## Why batch?
A model processes many inputs almost as cheaply as one — there's a fixed cost
to *invoking* the model (especially a GPU) that you pay whether you send 1 input
or 32. Running one request at a time pays that fixed cost every time; batching
pays it once per batch. The tradeoff: each request waits a few extra
milliseconds for the batch to form.

> Ferry, not a rowboat per passenger: waiting a moment to fill the ferry and
> crossing once moves far more people per hour than rowing each across alone.

## The measured result
```
  200 requests, model call-cost 20ms + 1ms/item

  unbatched: 4.23s  (   47 req/s)
  batched:   0.35s  (  565 req/s)  in 7 batches

  speedup: 11.9x
```
(The model here is a stub with a realistic 20ms fixed per-call cost. With a real
GPU model the absolute numbers differ, but the *shape* — batching amortizes the
fixed cost — is exactly the same.)

## How it works
`BatchedModel` wraps any Phase 1 `Model`. Callers `await infer(text)`:
1. the request joins an internal queue with its own `future`
2. a background worker pulls the first item, then keeps pulling until either
   `max_batch_size` is reached or `max_delay_ms` elapses
3. the model's `predict_batch` runs **once** on the whole batch
4. each caller's `future` is resolved with its own result

Two knobs trade latency for throughput:
- `max_batch_size` — the most requests to run together
- `max_delay_ms` — the longest a request waits to let a batch fill

This is the same independent-callers / shared-worker concurrency pattern as the
crawler and rate limiter, applied to inference.

## Run
```bash
python batch_phase3.py        # the throughput demo above
python test_serve_phase3.py   # tests
```

## Tests cover
- **each caller gets its own result** (no cross-batch mix-ups — verified with an
  echo model that returns each input's text)
- requests are genuinely grouped (30 requests run in far fewer than 30 batches)
- the `max_batch_size` cap is respected
- a single request still works, and a lone request returns within ~`max_delay`
  (doesn't hang waiting for a batch that never fills)
- a model exception propagates back to the waiting callers (the worker never
  dies silently)

## Interview notes
- **Latency vs throughput dial.** Bigger batches and longer delays raise
  throughput but add per-request latency. You tune to your SLA.
- **The hard part is correctness, not speed.** Many independent callers share
  one batched worker; the result-routing (each future gets *its* result) and
  not-hanging-a-lone-request are what the tests pin down.
- **Where it'd go next.** Under real load you might cap queue depth (shed load
  when overwhelmed) and expose batch-size metrics — which feeds Phase 4.

## Still deferred
- Metrics + dashboard (p50/p95/p99, prediction mix) → **Phase 4**
- Drift detection + chart → **Phase 5**
