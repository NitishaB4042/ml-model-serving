"""Generate a small monitoring snapshot chart for the README."""
import random, time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import metrics_phase4 as m
from serve_phase1 import StubTextClassifier

clf = StubTextClassifier()
met = m.Metrics(window=600)
mon = m.MonitoredModel(clf.predict, met)

real = ["the city council met", "a new study was published", "the schedule changed"]
fake = ["SHOCKING cure exposed!!!", "you won't believe this hoax", "click now doctors hate this!!!"]

# feed traffic, sampling p95 latency and fake-fraction over time
p95_series, fakefrac_series = [], []
for step in range(60):
    for _ in range(20):
        # drift the mix partway through to make the prediction-mix line move
        bias = 0.7 if step > 35 else 0.4
        text = random.choice(fake) if random.random() < bias else random.choice(real)
        time.sleep(random.uniform(0.0003, 0.003))
        mon.predict(text)
    snap = met.snapshot()
    p95_series.append(snap["latency_ms"]["p95"])
    mix = snap["prediction_mix"]
    tot = sum(mix.values()) or 1
    fakefrac_series.append(100.0 * mix.get("fake", 0) / tot)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
ax1.plot(p95_series, color="#C0392B", linewidth=2)
ax1.set_title("p95 latency over time")
ax1.set_xlabel("time (samples)"); ax1.set_ylabel("ms"); ax1.grid(True, alpha=0.3)

ax2.plot(fakefrac_series, color="#2E75B6", linewidth=2)
ax2.axvline(35, ls="--", color="#7F7F7F", label="upstream shift")
ax2.set_title("prediction mix: % 'fake' over time")
ax2.set_xlabel("time (samples)"); ax2.set_ylabel("% fake"); ax2.set_ylim(0, 100)
ax2.grid(True, alpha=0.3); ax2.legend()
fig.tight_layout()
fig.savefig("serving_metrics.png", dpi=130)
print("wrote serving_metrics.png")
