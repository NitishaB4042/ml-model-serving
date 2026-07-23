"""
ML Model Serving & Monitoring — Phase 3: dynamic batching for throughput.

The signature ML-serving optimization. A model processes many inputs almost as
cheaply as one (especially on a GPU), so instead of running it once per request
we collect requests that arrive within a short window into a BATCH, run the
model once on the whole batch, and route each result back to its caller.

This is a classic concurrency pattern: independent callers, one shared worker
that drains a queue in batches. Each caller waits on its own "future" until the
batch it joined completes.

Key knobs:
  max_batch_size : the most requests to run together
  max_delay_ms   : the longest a request waits to let a batch fill

Run the throughput demo:  python batch_phase3.py
Run the tests:            python test_phase3.py
"""

import asyncio
import time
from dataclasses import dataclass, field

from serve_phase1 import Model, StubTextClassifier


@dataclass
class _Pending:
    """One queued request: its text, and a future to receive its result."""
    text: str
    future: asyncio.Future


class BatchedModel:
    """Wraps a Model with dynamic batching.

    Callers `await infer(text)`. Internally, requests queue up; a background
    worker forms batches (up to max_batch_size, or after max_delay_ms) and runs
    the model's predict_batch once per batch, then resolves each caller's future.
    """

    def __init__(self, model: Model, max_batch_size: int = 16,
                 max_delay_ms: float = 5.0):
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_delay = max_delay_ms / 1000.0
        self._queue: asyncio.Queue[_Pending] = asyncio.Queue()
        self._worker: asyncio.Task | None = None
        self.batches_run = 0          # for observability / tests
        self.items_served = 0

    def start(self):
        if self._worker is None:
            self._worker = asyncio.create_task(self._run())

    async def stop(self):
        if self._worker:
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                pass
            self._worker = None

    async def infer(self, text: str):
        """Submit one request; await its result. Joins whatever batch forms."""
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        await self._queue.put(_Pending(text=text, future=fut))
        return await fut

    async def _run(self):
        """Background loop: gather a batch, run it, resolve futures."""
        while True:
            # block for the first item (no busy-waiting when idle)
            first = await self._queue.get()
            batch = [first]

            # try to fill the batch, but wait at most max_delay for stragglers
            deadline = time.monotonic() + self.max_delay
            while len(batch) < self.max_batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    nxt = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                    batch.append(nxt)
                except asyncio.TimeoutError:
                    break

            # run the model once on the whole batch
            texts = [p.text for p in batch]
            try:
                results = self.model.predict_batch(texts)
                for p, res in zip(batch, results):
                    if not p.future.done():
                        p.future.set_result(res)
            except Exception as e:        # never let the worker die silently
                for p in batch:
                    if not p.future.done():
                        p.future.set_exception(e)

            self.batches_run += 1
            self.items_served += len(batch)


# ===========================================================================
# Throughput demo: single-at-a-time vs batched, on a model with realistic
# per-CALL overhead (a fixed cost paid once per predict_batch call, plus a
# tiny per-item cost). This is what makes batching win.
# ===========================================================================
class SlowBatchModel(Model):
    """A stub whose predict_batch has a fixed per-CALL cost plus a small
    per-item cost \u2014 mimicking real model/GPU behaviour, where invoking the
    model is expensive but adding more items to a call is cheap."""
    name = "slow-batch-stub"
    labels = ("real", "fake")
    CALL_COST = 0.020        # 20ms fixed per predict_batch call
    ITEM_COST = 0.001        # 1ms per item

    def predict_batch(self, texts):
        time.sleep(self.CALL_COST + self.ITEM_COST * len(texts))
        return [("real", 0.7)] * len(texts)

    def predict(self, text):
        return self.predict_batch([text])[0]


async def _demo():
    N = 200
    model = SlowBatchModel()

    # --- unbatched: one model call per request (call cost paid every time) ---
    t0 = time.monotonic()
    for _ in range(N):
        model.predict("x")
    unbatched = time.monotonic() - t0

    # --- batched: requests collected and run together ---
    bm = BatchedModel(model, max_batch_size=32, max_delay_ms=5)
    bm.start()
    t0 = time.monotonic()
    await asyncio.gather(*[bm.infer("x") for _ in range(N)])
    batched = time.monotonic() - t0
    await bm.stop()

    print(f"  {N} requests, model call-cost {int(model.CALL_COST*1000)}ms + "
          f"{int(model.ITEM_COST*1000)}ms/item\n")
    print(f"  unbatched: {unbatched:.2f}s  ({N/unbatched:5.0f} req/s)")
    print(f"  batched:   {batched:.2f}s  ({N/batched:5.0f} req/s)  "
          f"in {bm.batches_run} batches")
    print(f"\n  speedup: {unbatched/batched:.1f}x")


if __name__ == "__main__":
    asyncio.run(_demo())
