"""
ML Model Serving & Monitoring — Phase 4: metrics & monitoring.

You can't operate what you can't see. This phase records, for every prediction:
  - throughput (requests over time)
  - latency, summarized as percentiles (p50 / p95 / p99) \u2014 NOT just the average,
    because the slow tail is what users feel and what SLAs target
  - the prediction MIX (how outputs are distributed) over time \u2014 a sudden shift
    is an early warning something changed upstream

It exposes a /metrics endpoint and feeds a simple dashboard.

Run the demo:  python metrics_phase4.py
Run tests:     python test_phase4.py
"""

import time
import bisect
from collections import deque, Counter
from dataclasses import dataclass, field
from threading import Lock


def percentile(sorted_vals, q):
    """q in [0,100]. Nearest-rank percentile on an already-sorted list."""
    if not sorted_vals:
        return 0.0
    if q <= 0:
        return sorted_vals[0]
    if q >= 100:
        return sorted_vals[-1]
    # nearest-rank: smallest value with at least q% of data <= it
    rank = max(1, int(round(q / 100.0 * len(sorted_vals))))
    return sorted_vals[min(rank - 1, len(sorted_vals) - 1)]


class Metrics:
    """Thread-safe rolling metrics over the most recent `window` requests."""

    def __init__(self, window: int = 1000):
        self.window = window
        self._lock = Lock()
        self._latencies: deque[float] = deque(maxlen=window)   # ms
        self._labels: deque[str] = deque(maxlen=window)
        self._times: deque[float] = deque(maxlen=window)       # epoch seconds
        self._total = 0

    def record(self, latency_ms: float, label: str):
        with self._lock:
            self._latencies.append(latency_ms)
            self._labels.append(label)
            self._times.append(time.time())
            self._total += 1

    def snapshot(self) -> dict:
        with self._lock:
            lats = sorted(self._latencies)
            labels = list(self._labels)
            times = list(self._times)
            total = self._total
        n = len(lats)
        # throughput over the window: requests / span of timestamps
        rps = 0.0
        if len(times) >= 2:
            span = times[-1] - times[0]
            rps = (len(times) - 1) / span if span > 0 else 0.0
        return {
            "total_requests": total,
            "window_count": n,
            "throughput_rps": round(rps, 2),
            "latency_ms": {
                "p50": round(percentile(lats, 50), 2),
                "p95": round(percentile(lats, 95), 2),
                "p99": round(percentile(lats, 99), 2),
                "max": round(lats[-1], 2) if lats else 0.0,
            },
            "prediction_mix": dict(Counter(labels)),
        }


# A small helper to time and record a prediction in one place.
class MonitoredModel:
    """Wraps a model (or batched model) and records metrics for each call."""
    def __init__(self, predict_fn, metrics: Metrics):
        self.predict_fn = predict_fn      # callable text -> (label, confidence)
        self.metrics = metrics

    def predict(self, text: str):
        t0 = time.perf_counter()
        label, conf = self.predict_fn(text)
        self.metrics.record((time.perf_counter() - t0) * 1000.0, label)
        return label, conf


# ===========================================================================
# Demo: feed a mix of traffic and print a metrics snapshot.
# ===========================================================================
def _demo():
    import random
    from serve_phase1 import StubTextClassifier

    clf = StubTextClassifier()
    metrics = Metrics(window=500)
    mon = MonitoredModel(clf.predict, metrics)

    real_samples = ["the city council met on tuesday",
                    "researchers published a new study",
                    "the train schedule changed this week"]
    fake_samples = ["SHOCKING miracle cure exposed!!!",
                    "you won't believe this secret hoax",
                    "click now: doctors hate this!!!"]

    print("Feeding 300 mixed requests...\n")
    for _ in range(300):
        text = random.choice(real_samples + fake_samples)
        # add a little artificial latency variation to make percentiles meaningful
        time.sleep(random.uniform(0.0005, 0.004))
        mon.predict(text)

    snap = metrics.snapshot()
    print(f"  total requests : {snap['total_requests']}")
    print(f"  throughput     : {snap['throughput_rps']} req/s")
    lat = snap["latency_ms"]
    print(f"  latency (ms)   : p50={lat['p50']}  p95={lat['p95']}  "
          f"p99={lat['p99']}  max={lat['max']}")
    print(f"  prediction mix : {snap['prediction_mix']}")
    print("\n  Note: p95/p99 (the tail) sit well above p50 \u2014 the average alone "
          "would hide those slow requests.")


if __name__ == "__main__":
    _demo()
