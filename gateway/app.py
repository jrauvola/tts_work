import asyncio
import base64
import json
import os
import time
from pathlib import Path
from typing import List, Optional

import grpc
import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, status
from fastapi.responses import HTMLResponse, JSONResponse
from prometheus_client import Counter, Gauge, Histogram, make_asgi_app
from opentelemetry import trace
try:
    import pynvml  # type: ignore
    pynvml.nvmlInit()
except Exception:
    pynvml = None

# Ensure generated protos are discoverable
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "protos", "gen", "python"))

import tts_pb2  # type: ignore
import tts_pb2_grpc  # type: ignore


APP_NAME = os.environ.get("APP_NAME", "gateway")
API_KEY = os.environ.get("API_KEY", "dev-key")
LOCAL_MODE = os.environ.get("LOCAL_MODE", "0") == "1"
if LOCAL_MODE:
    TEXTPROC_URL = os.environ.get("TEXTPROC_URL", "http://localhost:8001")
    TTS_NATIVE_ADDR = os.environ.get("TTS_NATIVE_ADDR", "localhost:50052")
else:
    TEXTPROC_URL = os.environ.get("TEXTPROC_URL", "http://textproc:8001")
    TTS_NATIVE_ADDR = os.environ.get("TTS_NATIVE_ADDR", "tts_native:50052")
FRAME_MS = int(os.environ.get("FRAME_MS", "20"))
MAX_INFLIGHT = int(os.environ.get("MAX_INFLIGHT", "2"))
TARGET_SAMPLE_RATE = int(os.environ.get("TARGET_SAMPLE_RATE", "44100"))
XTTS_SAMPLES_DIR = os.environ.get("XTTS_SAMPLES_DIR")

app = FastAPI(title="Gateway & Orchestrator", version="0.1.0")


# Metrics
WS_ACTIVE = Gauge("ws_active_conns", "Active websocket connections")
TTFA_MS = Histogram("ttfa_ms", "Time to first audio (ms)", buckets=(50, 100, 200, 300, 400, 600, 800, 1200, 2000))
ERRORS = Counter("errors_total", "Errors by code", ["code"])  # code labels e.g., BAD_INPUT
QPS = Counter("requests_total", "Total requests")
GPU_UTIL = Gauge("gpu_utilization", "GPU utilization percent")


@app.get("/livez")
async def livez() -> JSONResponse:
    return JSONResponse({"status": "ok", "service": APP_NAME})


@app.get("/healthz")
async def healthz() -> JSONResponse:
    return JSONResponse({"status": "ready", "service": APP_NAME})


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_PATH = Path(CURRENT_DIR).parent / "frontend" / "index.html"

app.mount("/metrics", make_asgi_app())


@app.get("/ui", response_class=HTMLResponse)
async def serve_ui_root() -> HTMLResponse:
    return HTMLResponse(FRONTEND_PATH.read_text(encoding="utf-8"))


@app.get("/ui/index.html", response_class=HTMLResponse)
async def serve_ui_index() -> HTMLResponse:
    return HTMLResponse(FRONTEND_PATH.read_text(encoding="utf-8"))


async def normalize_text(client: httpx.AsyncClient, text: str) -> str:
    tracer = trace.get_tracer("gateway")
    with tracer.start_as_current_span("normalize_call"):
        try:
            resp = await client.post(f"{TEXTPROC_URL}/normalize", json={"text": text}, timeout=3.0)
            resp.raise_for_status()
            return resp.json().get("normalized_text", text)
        except httpx.HTTPError:
            return text


def _encoding_to_proto(enc: str) -> int:
    if enc == "opus":
        return tts_pb2.OPUS
    return tts_pb2.PCM16


def _candidate_tts_targets() -> List[str]:
    candidates = [TTS_NATIVE_ADDR]
    fallbacks = ["localhost:50052", "127.0.0.1:50052"]
    for fb in fallbacks:
        if fb not in candidates:
            candidates.append(fb)
    return candidates


def _grpc_channel(addr: str) -> grpc.aio.Channel:
    if os.environ.get("TTS_TLS", "0") == "1":
        ca_path = os.environ.get("TTS_TLS_CA", "/certs/ca.pem")
        try:
            with open(ca_path, "rb") as f:
                creds = grpc.ssl_channel_credentials(root_certificates=f.read())
            return grpc.aio.secure_channel(addr, creds)
        except Exception:
            return grpc.aio.insecure_channel(addr)
    return grpc.aio.insecure_channel(addr)


async def synthesize_stream(channel: grpc.aio.Channel, req: tts_pb2.SynthesizeRequest):
    tracer = trace.get_tracer("gateway")
    with tracer.start_as_current_span("grpc_synthesize_stream"):
        stub = tts_pb2_grpc.TTSServiceStub(channel)
        return stub.SynthesizeStream(req)


@app.websocket("/ws")
async def ws_handler(ws: WebSocket):
    api_key = ws.query_params.get("api_key")
    if api_key != API_KEY:
        await ws.close(code=status.WS_1008_POLICY_VIOLATION)
        return
    await ws.accept()
    WS_ACTIVE.inc()
    QPS.inc()

    buffer: List[str] = []

    async with httpx.AsyncClient() as http_client:
        try:
            while True:
                try:
                    raw = await ws.receive_text()
                except WebSocketDisconnect:
                    break

                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError:
                    await ws.send_text(json.dumps({"error": "invalid_json"}))
                    ERRORS.labels(code="BAD_INPUT").inc()
                    continue

                text = payload.get("text", "")
                flush = bool(payload.get("flush", False))
                model = payload.get("model", "coqui").lower()
                speaker_wav = payload.get("speaker_wav", "")
                lang = payload.get("lang", "en")

                if text:
                    normalized = await normalize_text(http_client, text)
                    buffer.append(normalized)

                if not flush:
                    continue

                joined = "".join(buffer).strip()
                buffer.clear()

                if not joined:
                    await ws.close()
                    break

                req_id = f"ws-{int(time.time() * 1000)}"
                backend_enum = tts_pb2.MODEL_BACKEND_COQUI
                if model == "xtts":
                    backend_enum = tts_pb2.MODEL_BACKEND_XTTS

                request = tts_pb2.SynthesizeRequest(
                    request_id=req_id,
                    chunk_index=0,
                    text=joined,
                    voice="default",
                    encoding=tts_pb2.PCM16,
                    sample_rate_hz=TARGET_SAMPLE_RATE,
                    lang=lang,
                    seed=42,
                    backend=backend_enum,
                    speaker_wav=speaker_wav,
                )

                ttfa_started = time.time()
                next_alignment_idx = 0

                last_error: Optional[Exception] = None
                success = False
                aggregated_alignment = {
                    "chars": [],
                    "char_start_times_ms": [],
                    "char_durations_ms": [],
                }
                for target in _candidate_tts_targets():
                    try:
                        async with _grpc_channel(target) as channel:
                            call = await synthesize_stream(channel, request)
                            async for resp in call:
                                audio_b64 = base64.b64encode(resp.audio_frame).decode("ascii")
                                sorted_aligns = sorted(resp.alignments, key=lambda a: a.char_index)
                                new_aligns = []
                                for a in sorted_aligns:
                                    if a.char_index < next_alignment_idx:
                                        continue
                                    new_aligns.append(a)
                                if new_aligns:
                                    next_alignment_idx = new_aligns[-1].char_index + 1
                                aggregated_alignment["chars"].extend(a.grapheme for a in new_aligns)
                                aggregated_alignment["char_start_times_ms"].extend(a.start_ms for a in new_aligns)
                                aggregated_alignment["char_durations_ms"].extend(a.duration_ms for a in new_aligns)
                                alignment_payload = aggregated_alignment
                                message = {
                                    "audio": audio_b64,
                                    "alignment": alignment_payload,
                                    "format": {
                                        "encoding": "pcm16",
                                        "sample_rate_hz": TARGET_SAMPLE_RATE,
                                        "frame_ms": FRAME_MS,
                                    },
                                    "chunk_index": resp.frames_sent - 1,
                                }
                                if ttfa_started is not None:
                                    ttfa_ms = (time.time() - ttfa_started) * 1000.0
                                    message["ttfa_ms"] = ttfa_ms
                                    TTFA_MS.observe(ttfa_ms)
                                    ttfa_started = None
                                await ws.send_text(json.dumps(message))
                                if resp.is_last:
                                    break
                        success = True
                        break
                    except grpc.aio.AioRpcError as exc:
                        last_error = exc
                        continue
                if success:
                    await ws.send_text(json.dumps({"status": "done"}))
                else:
                    ERRORS.labels(code="TTS_NATIVE_UNAVAILABLE").inc()
                    err_payload = {"error": "tts_native_unavailable"}
                    if last_error is not None:
                        err_payload["details"] = last_error.details() if hasattr(last_error, "details") else str(last_error)
                    await ws.send_text(json.dumps(err_payload))
                    break
        finally:
            WS_ACTIVE.dec()
            try:
                await ws.close()
            except Exception:
                pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), loop="uvloop")


