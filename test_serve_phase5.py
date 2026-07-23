"""
Tests for Model Serving Phase 5 — drift detection (PSI).
Run: python test_serve_phase5.py
"""
import random
import drift_phase5 as m


def test_psi_zero_for_identical():
    edges = m.make_edges(0, 1, 10)
    vals = [0.1, 0.3, 0.5, 0.7, 0.9] * 20
    # identical distributions -> PSI ~ 0
    val = m.psi(vals, vals, edges)
    assert val < 1e-6, val
    print("  psi_zero_for_identical: PASS")


def test_psi_small_for_same_distribution():
    random.seed(1)
    edges = m.make_edges(0, 1, 10)
    base = [random.gauss(0.5, 0.1) for _ in range(2000)]
    live = [random.gauss(0.5, 0.1) for _ in range(2000)]   # same dist, new sample
    val = m.psi(base, live, edges)
    assert val < 0.1, val          # should read as "no drift"
    assert m.classify_psi(val) == "none"
    print(f"  psi_small_for_same_distribution (psi={val:.3f}): PASS")


def test_psi_large_for_shifted():
    random.seed(2)
    edges = m.make_edges(0, 1, 10)
    base = [random.gauss(0.8, 0.08) for _ in range(2000)]
    live = [random.gauss(0.4, 0.10) for _ in range(2000)]  # clearly shifted
    val = m.psi(base, live, edges)
    assert val >= 0.25, val        # should read as "significant"
    assert m.classify_psi(val) == "significant"
    print(f"  psi_large_for_shifted (psi={val:.2f}): PASS")


def test_classify_bands():
    assert m.classify_psi(0.05) == "none"
    assert m.classify_psi(0.15) == "moderate"
    assert m.classify_psi(0.40) == "significant"
    print("  classify_bands: PASS")


def test_detector_no_alert_when_stable():
    random.seed(3)
    base = [random.gauss(0.8, 0.08) for _ in range(2000)]
    det = m.DriftDetector(base, alert_threshold=0.25)
    live = [random.gauss(0.8, 0.08) for _ in range(200)]
    res = det.score(live)
    assert res["alert"] is False, res
    assert res["level"] == "none"
    print(f"  detector_no_alert_when_stable (psi={res['psi']}): PASS")


def test_detector_alerts_on_drift():
    random.seed(4)
    base = [random.gauss(0.8, 0.08) for _ in range(2000)]
    det = m.DriftDetector(base, alert_threshold=0.25)
    live = [random.gauss(0.45, 0.10) for _ in range(200)]
    res = det.score(live)
    assert res["alert"] is True, res
    assert res["level"] == "significant"
    print(f"  detector_alerts_on_drift (psi={res['psi']}): PASS")


def test_empty_live_is_no_drift():
    det = m.DriftDetector([0.5] * 100)
    res = det.score([])
    assert res["alert"] is False and res["n"] == 0
    print("  empty_live_is_no_drift: PASS")


def test_empty_baseline_rejected():
    try:
        m.DriftDetector([])
        assert False, "should have raised"
    except ValueError:
        pass
    print("  empty_baseline_rejected: PASS")


def test_histogram_sums_to_one():
    edges = m.make_edges(0, 1, 10)
    h = m._histogram([0.1, 0.2, 0.9, 0.95], edges)
    assert abs(sum(h) - 1.0) < 1e-9, sum(h)
    print("  histogram_sums_to_one: PASS")


if __name__ == "__main__":
    print("Running Model Serving Phase 5 tests:")
    test_psi_zero_for_identical()
    test_psi_small_for_same_distribution()
    test_psi_large_for_shifted()
    test_classify_bands()
    test_detector_no_alert_when_stable()
    test_detector_alerts_on_drift()
    test_empty_live_is_no_drift()
    test_empty_baseline_rejected()
    test_histogram_sums_to_one()
    print("All tests passed.")
