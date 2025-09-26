import asyncio
import base64
import json
import random
import statistics
import time
from typing import Dict, List

import websockets


async def run_client(idx: int, url: str, texts: List[str], results: List[float]):
    req_id = f"bench-{idx}"
    async with websockets.connect(url, ping_interval=20) as ws:
        # Note: The websocket protocol for the benchmark script seems different
        # from the example client. I'm assuming the 'model' parameter is no longer needed
        # in this initial message. If the gateway requires it, this may need adjustment.
        await ws.send(json.dumps({
            "type": "start", "request_id": req_id, "encoding": "pcm16", "sample_rate_hz": 16000
        }))
        # Choose random text
        text = random.choice(texts)
        await ws.send(json.dumps({"type": "text", "request_id": req_id, "chunk_index": 0, "text": text}))
        await ws.send(json.dumps({"type": "end", "request_id": req_id}))

        ttfa = None
        try:
            while True:
                msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
                data = json.loads(msg)
                if data.get("type") == "audio" and "ttfa_ms" in data:
                    ttfa = float(data["ttfa_ms"])
                    break
        except asyncio.TimeoutError:
            pass
        if ttfa is not None:
            results.append(ttfa)


async def bench(concurrency: int, url: str) -> Dict[str, float]:
    texts = [
        "Hello world.",
        "This is a slightly longer sentence for benchmarking.",
        "E equals m c squared is a famous equation in physics.",
        "The integral from zero to one of x squared d x.",
    ]
    results: List[float] = []
    tasks = [run_client(i, url, texts, results) for i in range(concurrency)]
    start = time.time()
    await asyncio.gather(*tasks)
    elapsed = time.time() - start
    if not results:
        return {"p50": float("nan"), "p95": float("nan"), "qps": 0.0}
    p50 = statistics.median(results)
    p95 = sorted(results)[max(0, int(len(results) * 0.95) - 1)]
    qps = len(results) / elapsed
    return {"p50": p50, "p95": p95, "qps": qps}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="ws://localhost:8000/ws?api_key=dev-key")
    parser.add_argument("--concurrency", type=int, default=10)
    args = parser.parse_args()
    res = asyncio.run(bench(args.concurrency, args.url))
    print(res)


