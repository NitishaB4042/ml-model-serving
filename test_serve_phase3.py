"""
Tests for Model Serving Phase 3 — dynamic batching.
Run: python test_serve_phase3.py
"""
import asyncio
import time
import batch_phase3 as m
from serve_phase1 import Model


class EchoModel(Model):
    """Returns each input's text as the 'label', so we can verify that results
    are routed back to the CORRECT caller (no mix-ups across a batch)."""
    name = "echo"; labels = ("x",)
    def predict_batch(self, texts):
        return [(t, 1.0) for t in texts]
    def predict(self, text):
        return (text, 1.0)


class CountingModel(Model):
    name = "count"; labels = ("real",)
    def __init__(self):
        self.batch_sizes = []
    def predict_batch(self, texts):
        self.batch_sizes.append(len(texts))
        return [("real", 0.9)] * len(texts)
    def predict(self, text):
        return self.predict_batch([text])[0]


def test_each_caller_gets_its_own_result():
    async def go():
        bm = m.BatchedModel(EchoModel(), max_batch_size=16, max_delay_ms=10)
        bm.start()
        texts = [f"req-{i}" for i in range(20)]
        results = await asyncio.gather(*[bm.infer(t) for t in texts])
        await bm.stop()
        # each result's label must equal the text that caller submitted
        for t, (label, conf) in zip(texts, results):
            assert label == t, (t, label)
    asyncio.run(go())
    print("  each_caller_gets_its_own_result: PASS")


def test_requests_are_actually_batched():
    async def go():
        model = CountingModel()
        bm = m.BatchedModel(model, max_batch_size=32, max_delay_ms=20)
        bm.start()
        # fire 30 at once -> should run in far fewer than 30 batches
        await asyncio.gather(*[bm.infer("x") for _ in range(30)])
        await bm.stop()
        assert sum(model.batch_sizes) == 30
        assert len(model.batch_sizes) < 30, model.batch_sizes   # genuinely grouped
        assert max(model.batch_sizes) > 1
    asyncio.run(go())
    print("  requests_are_actually_batched: PASS")


def test_batch_size_cap_respected():
    async def go():
        model = CountingModel()
        bm = m.BatchedModel(model, max_batch_size=8, max_delay_ms=20)
        bm.start()
        await asyncio.gather(*[bm.infer("x") for _ in range(40)])
        await bm.stop()
        assert max(model.batch_sizes) <= 8, model.batch_sizes
    asyncio.run(go())
    print("  batch_size_cap_respected: PASS")


def test_single_request_still_works():
    async def go():
        bm = m.BatchedModel(EchoModel(), max_batch_size=16, max_delay_ms=5)
        bm.start()
        label, conf = await bm.infer("solo")
        await bm.stop()
        assert label == "solo"
    asyncio.run(go())
    print("  single_request_still_works: PASS")


def test_delay_bounds_wait_for_small_load():
    # a lone request should return within roughly max_delay, not hang
    async def go():
        bm = m.BatchedModel(EchoModel(), max_batch_size=64, max_delay_ms=30)
        bm.start()
        t0 = time.monotonic()
        await bm.infer("solo")
        elapsed = time.monotonic() - t0
        await bm.stop()
        assert elapsed < 0.5, elapsed     # generous upper bound; must not hang
    asyncio.run(go())
    print("  delay_bounds_wait_for_small_load: PASS")


def test_exception_propagates_to_callers():
    class BoomModel(Model):
        name = "boom"; labels = ("x",)
        def predict_batch(self, texts):
            raise RuntimeError("model failed")
        def predict(self, text):
            return self.predict_batch([text])[0]
    async def go():
        bm = m.BatchedModel(BoomModel(), max_batch_size=4, max_delay_ms=5)
        bm.start()
        try:
            await bm.infer("x")
            assert False, "should have raised"
        except RuntimeError:
            pass
        finally:
            await bm.stop()
    asyncio.run(go())
    print("  exception_propagates_to_callers: PASS")


if __name__ == "__main__":
    print("Running Model Serving Phase 3 tests:")
    test_each_caller_gets_its_own_result()
    test_requests_are_actually_batched()
    test_batch_size_cap_respected()
    test_single_request_still_works()
    test_delay_bounds_wait_for_small_load()
    test_exception_propagates_to_callers()
    print("All tests passed.")
