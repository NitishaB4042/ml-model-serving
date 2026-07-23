"""
ML Model Serving & Monitoring — Phase 5: drift detection.

A model trained on yesterday's data silently gets worse when today's inputs look
different. The model doesn't error; it just quietly becomes wrong. Catching that
is a core ML-engineering job.

We detect drift by comparing the LIVE distribution of a tracked quantity (here,
the model's confidence / a feature value) against a fixed BASELINE distribution
captured from reference data. The comparison uses the Population Stability Index
(PSI), a standard, defensible drift measure. When PSI crosses a threshold, we
raise a drift alert.

Important framing: detecting drift in the INPUTS is an EARLY WARNING that
accuracy MAY be degrading \u2014 it is not a measured accuracy drop (we usually have
no live ground-truth labels). It prompts investigation or retraining.

Run the demo:  python drift_phase5.py
Run tests:     python test_phase5.py
"""

import math
import json
from dataclasses import dataclass


# ===========================================================================
# Population Stability Index
#   PSI = sum over bins of (live% - base%) * ln(live% / base%)
# Rule-of-thumb interpretation (industry-standard):
#   PSI < 0.1   : no significant drift
#   0.1 <= PSI < 0.25 : moderate drift (watch)
#   PSI >= 0.25 : significant drift (alert)
# ===========================================================================
def _histogram(values, edges):
    """Fraction of `values` falling in each bin defined by `edges`."""
    counts = [0] * (len(edges) - 1)
    for v in values:
        # find bin; clamp to the outer bins
        placed = False
        for i in range(len(edges) - 1):
            if edges[i] <= v < edges[i + 1]:
                counts[i] += 1
                placed = True
                break
        if not placed:
            if v < edges[0]:
                counts[0] += 1
            else:
                counts[-1] += 1
    total = sum(counts) or 1
    return [c / total for c in counts]


def make_edges(lo, hi, bins):
    step = (hi - lo) / bins
    return [lo + i * step for i in range(bins + 1)]


def psi(baseline_vals, live_vals, edges, eps=1e-4):
    """Population Stability Index between baseline and live distributions."""
    b = _histogram(baseline_vals, edges)
    l = _histogram(live_vals, edges)
    total = 0.0
    for bp, lp in zip(b, l):
        bp = max(bp, eps)        # avoid log(0) / divide-by-zero on empty bins
        lp = max(lp, eps)
        total += (lp - bp) * math.log(lp / bp)
    return total


def classify_psi(value):
    if value < 0.1:
        return "none"
    if value < 0.25:
        return "moderate"
    return "significant"


# ===========================================================================
# DriftDetector: holds a baseline, accepts live samples, reports drift.
# ===========================================================================
class DriftDetector:
    def __init__(self, baseline_vals, lo=0.0, hi=1.0, bins=10,
                 alert_threshold=0.25):
        if not baseline_vals:
            raise ValueError("baseline must be non-empty")
        self.edges = make_edges(lo, hi, bins)
        self.baseline = list(baseline_vals)
        self.alert_threshold = alert_threshold

    def score(self, live_vals) -> dict:
        if not live_vals:
            return {"psi": 0.0, "level": "none", "alert": False, "n": 0}
        value = psi(self.baseline, live_vals, self.edges)
        return {
            "psi": round(value, 4),
            "level": classify_psi(value),
            "alert": value >= self.alert_threshold,
            "n": len(live_vals),
        }


# ===========================================================================
# Demo: a stream that is stable for a while, then drifts. Chart PSI over time.
# ===========================================================================
def _demo():
    import random
    random.seed(7)

    # Baseline: confidences clustered around 0.8 (model is usually sure).
    baseline = [min(0.999, max(0.001, random.gauss(0.8, 0.08))) for _ in range(2000)]
    detector = DriftDetector(baseline, lo=0.0, hi=1.0, bins=10, alert_threshold=0.25)

    print("Baseline: confidence ~ N(0.80, 0.08). Streaming windows over time.\n")
    print(f"  {'window':>6} | {'mean conf':>9} | {'PSI':>6} | level        | alert")
    print("  " + "-" * 55)

    psi_series, alert_points = [], []
    for w in range(20):
        # windows 0-9 look like baseline; from 10 on, confidence drifts DOWN
        # (model is getting unsure -> inputs likely changed)
        if w < 10:
            center = 0.80
        else:
            center = 0.80 - 0.04 * (w - 9)     # gradual drift downward
        live = [min(0.999, max(0.001, random.gauss(center, 0.10))) for _ in range(150)]
        res = detector.score(live)
        psi_series.append(res["psi"])
        if res["alert"]:
            alert_points.append(w)
        flag = "  <-- ALERT" if res["alert"] else ""
        print(f"  {w:>6} | {center:>9.2f} | {res['psi']:>6.3f} | "
              f"{res['level']:<12} |{flag}")

    first_alert = alert_points[0] if alert_points else None
    print(f"\n  First drift alert at window: {first_alert}")

    with open("serving_drift_results.json", "w") as f:
        json.dump({"psi_series": [round(x, 4) for x in psi_series],
                   "first_alert_window": first_alert,
                   "threshold": detector.alert_threshold}, f, indent=2)
    print("  wrote serving_drift_results.json")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4.4))
        ax.plot(psi_series, "o-", color="#2E75B6", linewidth=2, label="PSI (drift score)")
        ax.axhline(0.25, ls="--", color="#C0392B", label="alert threshold (0.25)")
        ax.axhline(0.10, ls=":", color="#B9770E", label="watch threshold (0.10)")
        ax.axvline(9.5, ls="-", color="#7F7F7F", alpha=0.5, label="inputs start drifting")
        ax.set_xlabel("window (time)")
        ax.set_ylabel("PSI")
        ax.set_title("Drift detection: PSI rises and crosses the alert line as inputs shift")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig("serving_drift.png", dpi=130)
        print("  wrote serving_drift.png")
    except ImportError:
        print("  (matplotlib not installed \u2014 skipped chart)")

    print("\n  Note: PSI measures INPUT drift \u2014 an early warning that accuracy may "
          "be slipping, prompting investigation/retraining. It is not itself a "
          "measured accuracy drop (no live labels).")


if __name__ == "__main__":
    _demo()
