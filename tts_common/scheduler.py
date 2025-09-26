import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional


@dataclass
class SynthFrame:
    audio_frame: bytes
    alignments: List[Dict[str, Any]]
    is_last: bool


@dataclass
class ScheduledRequest:
    request_id: str
    chunk_index: int
    text: str
    sample_rate_hz: int
    frame_ms: int
    params: Dict[str, Any] = field(default_factory=dict)
    enqueue_ts: float = field(default_factory=lambda: time.time())
    out_queue: "asyncio.Queue[SynthFrame]" = field(default_factory=asyncio.Queue)


class DynamicBatchScheduler:
    """
    Simple dynamic micro-batching scheduler.

    - Gathers requests up to scheduler_window_ms or until max_batch_size.
    - Packs FIFO to maintain fairness.
    - Calls backend.synthesize_batch(requests) which must return a list of lists of SynthFrame.
    - Streams frames for each request into its per-request queue.
    """

    def __init__(
        self,
        backend_synthesize_batch: Callable[[List[ScheduledRequest]], Awaitable[List[List[SynthFrame]]]],
        scheduler_window_ms: int = 20,
        max_batch_size: int = 8,
    ) -> None:
        self._backend_synthesize_batch = backend_synthesize_batch
        self._scheduler_window_ms = scheduler_window_ms
        self._max_batch_size = max_batch_size
        self._pending: "asyncio.Queue[ScheduledRequest]" = asyncio.Queue()
        self._stop = asyncio.Event()
        self._runner_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        if self._runner_task is None:
            self._runner_task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._stop.set()
        if self._runner_task:
            await self._runner_task

    async def submit(self, req: ScheduledRequest) -> asyncio.Queue:
        await self._pending.put(req)
        return req.out_queue

    async def _run(self) -> None:
        while not self._stop.is_set():
            batch: List[ScheduledRequest] = []
            try:
                # Wait for the first item
                first = await asyncio.wait_for(self._pending.get(), timeout=0.05)
                batch.append(first)
            except asyncio.TimeoutError:
                continue

            deadline = time.time() + (self._scheduler_window_ms / 1000.0)

            while len(batch) < self._max_batch_size and time.time() < deadline:
                try:
                    batch.append(await asyncio.wait_for(self._pending.get(), timeout=deadline - time.time()))
                except asyncio.TimeoutError:
                    break

            # FIFO fairness already preserved by queue order
            try:
                results = await self._backend_synthesize_batch(batch)
            except Exception as e:  # do not swallow; send error frames as last
                for req in batch:
                    await req.out_queue.put(
                        SynthFrame(audio_frame=b"", alignments=[], is_last=True)
                    )
                continue

            # Stream frames to each request's queue
            stream_tasks: List[asyncio.Task] = []
            for req, frames in zip(batch, results):
                async def _feed(q: asyncio.Queue, frames_seq: List[SynthFrame]):
                    for fr in frames_seq:
                        await q.put(fr)
                stream_tasks.append(asyncio.create_task(_feed(req.out_queue, frames)))

            if stream_tasks:
                await asyncio.gather(*stream_tasks)


