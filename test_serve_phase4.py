"""
Tests for Model Serving Phase 4 — metrics.
Run: python test_serve_phase4.py
"""
import time
import metrics_phase4 as m


def test_percentile_known_values():
    vals = list(range(1, 101))   # 1..100, sorted
    assert m.percentile(vals, 50) == 50, m.percentile(vals, 50)
    assert m.percentile(vals, 95) == 95, m.percentile(vals, 95)
    assert m.percentile(vals, 99) == 99, m.percentile(vals, 99)
    assert m.percentile(vals, 100) == 100
    assert m.percentile(vals, 0) == 1
    print("  percentile_known_values: PASS")


def test_percentile_empty():
    assert m.percentile([], 95) == 0.0
    print("  percentile_empty: PASS")


def test_percentiles_ordered():
    met = m.Metrics(window=1000)
    for i in range(1, 201):
        met.record(latency_ms=float(i), label="real")
    snap = met.snapshot()
    lat = snap["latency_ms"]
    assert lat["p50"] <= lat["p95"] <= lat["p99"] <= lat["max"], lat
    # p50 of 1..200 should be ~100
    assert 95 <= lat["p50"] <= 105, lat["p50"]
    print("  percentiles_ordered: PASS")


def test_prediction_mix_counts():
    met = m.Metrics(window=1000)
    for _ in range(7):
        met.record(1.0, "real")
    for _ in range(3):
        met.record(1.0, "fake")
    snap = met.snapshot()
    assert snap["prediction_mix"] == {"real": 7, "fake": 3}, snap["prediction_mix"]
    print("  prediction_mix_counts: PASS")


def test_window_rolls_off_old_data():
    met = m.Metrics(window=50)
    for i in range(200):
        met.record(latency_ms=1.0, label="real")
    snap = met.snapshot()
    assert snap["window_count"] == 50, snap["window_count"]   # only last 50 kept
    assert snap["total_requests"] == 200                      # but total counts all
    print("  window_rolls_off_old_data: PASS")


def test_throughput_positive_under_load():
    met = m.Metrics(window=100)
    for _ in range(20):
        met.record(1.0, "real")
        time.sleep(0.002)
    snap = met.snapshot()
    assert snap["throughput_rps"] > 0, snap["throughput_rps"]
    print("  throughput_positive_under_load: PASS")


def test_monitored_model_records():
    met = m.Metrics(window=100)
    mon = m.MonitoredModel(lambda text: ("real", 0.9), met)
    label, conf = mon.predict("hello")
    assert label == "real"
    snap = met.snapshot()
    assert snap["total_requests"] == 1
    assert snap["prediction_mix"] == {"real": 1}
    print("  monitored_model_records: PASS")


if __name__ == "__main__":
    print("Running Model Serving Phase 4 tests:")
    test_percentile_known_values()
    test_percentile_empty()
    test_percentiles_ordered()
    test_prediction_mix_counts()
    test_window_rolls_off_old_data()
    test_throughput_positive_under_load()
    test_monitored_model_records()
    print("All tests passed.")
