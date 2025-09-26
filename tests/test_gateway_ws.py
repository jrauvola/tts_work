import asyncio
import json
from typing import Any, Dict, List

import pytest
from starlette.websockets import WebSocketDisconnect

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "protos" / "gen" / "python"))

import tts_pb2  # type: ignore
import numpy as np

from tts_native.synthesis import SynthesisResult
from tts_common.scheduler import ScheduledRequest
from tts_native.server import FRAME_MS, TARGET_SAMPLE_RATE, NativeAlignTTS


class FakeWebSocket:
    def __init__(self) -> None:
        self.sent: List[str] = []
        self._closed = False
        class _QueryParams:
            def get(self, key: str, default=None):
                if key == "api_key":
                    return "dev-key"
                return default

        self.query_params = _QueryParams()
        self._recv_queue: asyncio.Queue[str] = asyncio.Queue()

    async def accept(self) -> None:
        return None

    async def receive_text(self) -> str:
        if self._recv_queue.empty():
            raise WebSocketDisconnect()
        return await self._recv_queue.get()

    async def send_text(self, data: str) -> None:
        self.sent.append(data)

    async def close(self, code: int = 1000) -> None:
        self._closed = True

    async def queue_message(self, data: Dict[str, Any]) -> None:
        await self._recv_queue.put(json.dumps(data))


class DummyCall:
    def __init__(self, responses: List[tts_pb2.SynthesizeResponse]):
        self._responses = responses

    def __aiter__(self):
        async def iterator():
            for resp in self._responses:
                yield resp
        return iterator()


class DummyChannel:
    def __init__(self, call: DummyCall):
        self._call = call

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def __getattr__(self, name):
        if name == "SynthesizeStream":
            def _call(*args, **kwargs):
                return _call_wrapper(self._call)

            return _call
        raise AttributeError


def _call_wrapper(call: DummyCall):
    class _Awaitable:
        def __await__(self):
            async def _coro():
                return call
            return _coro().__await__()

    return _Awaitable()


@pytest.mark.asyncio
async def test_gateway_alignment_aggregation(monkeypatch):
    from gateway.app import ws_handler

    responses = [
        tts_pb2.SynthesizeResponse(
            request_id="test",
            chunk_index=0,
            audio_frame=b"abc",
            format=tts_pb2.AudioFormat(encoding=tts_pb2.PCM16, sample_rate_hz=44100, frame_ms=20),
            alignments=[
                tts_pb2.CharAlignment(char_index=0, grapheme="H", start_ms=0.0, duration_ms=100.0),
                tts_pb2.CharAlignment(char_index=1, grapheme="e", start_ms=100.0, duration_ms=80.0),
            ],
            is_last=False,
            frames_sent=1,
        ),
        tts_pb2.SynthesizeResponse(
            request_id="test",
            chunk_index=0,
            audio_frame=b"def",
            format=tts_pb2.AudioFormat(encoding=tts_pb2.PCM16, sample_rate_hz=44100, frame_ms=20),
            alignments=[
                tts_pb2.CharAlignment(char_index=2, grapheme="l", start_ms=180.0, duration_ms=120.0),
                tts_pb2.CharAlignment(char_index=3, grapheme="l", start_ms=300.0, duration_ms=120.0),
            ],
            is_last=False,
            frames_sent=2,
        ),
        tts_pb2.SynthesizeResponse(
            request_id="test",
            chunk_index=0,
            audio_frame=b"ghi",
            format=tts_pb2.AudioFormat(encoding=tts_pb2.PCM16, sample_rate_hz=44100, frame_ms=20),
            alignments=[
                tts_pb2.CharAlignment(char_index=4, grapheme="o", start_ms=420.0, duration_ms=150.0),
            ],
            is_last=True,
            frames_sent=3,
        ),
    ]

    dummy_call = DummyCall(responses)

    async def fake_synth_stream(channel, request):
        return dummy_call

    async def fake_grpc_channel(_addr):
        return DummyChannel(dummy_call)

    monkeypatch.setattr("gateway.app.synthesize_stream", fake_synth_stream)
    monkeypatch.setattr("gateway.app._candidate_tts_targets", lambda: ["tts_native:50052", "localhost:50052"])
    monkeypatch.setattr("gateway.app._grpc_channel", lambda addr: DummyChannel(dummy_call))
    async def fake_normalize(_client, text):
        return text

    monkeypatch.setattr("gateway.app.normalize_text", fake_normalize)

    ws = FakeWebSocket()
    await ws.queue_message({"text": "Hello", "flush": True})

    await ws_handler(ws)

    payloads = [json.loads(msg) for msg in ws.sent]
    audio_frames = [p for p in payloads if "audio" in p]
    assert len(audio_frames) == 3
    final_alignment = audio_frames[-1]["alignment"]
    assert final_alignment["chars"] == ["H", "e", "l", "l", "o"]
    assert final_alignment["char_start_times_ms"] == [0.0, 100.0, 180.0, 300.0, 420.0]
    assert final_alignment["char_durations_ms"] == [100.0, 80.0, 120.0, 120.0, 150.0]
    assert payloads[-1] == {"status": "done"}


def test_native_server_unique_alignments():
    svc = NativeAlignTTS()

    class DummyBackend:
        def synthesize(self, text: str) -> SynthesisResult:
            samples_per_frame = int(TARGET_SAMPLE_RATE * FRAME_MS / 1000)
            audio = np.zeros(samples_per_frame * 5, dtype=np.float32)
            chars = ["A", "B", "C"]
            alignments = [
                (0.0, FRAME_MS),
                (FRAME_MS, FRAME_MS),
                (FRAME_MS * 2, FRAME_MS),
            ]
            return SynthesisResult(
                audio_44k=audio,
                sample_rate=TARGET_SAMPLE_RATE,
                chars=chars,
                alignments_ms=alignments,
            )

    svc._backends["coqui"] = DummyBackend()  # type: ignore[index]
    req = ScheduledRequest(
        request_id="test",
        chunk_index=0,
        text="abc",
        sample_rate_hz=TARGET_SAMPLE_RATE,
        frame_ms=FRAME_MS,
        params={"backend": tts_pb2.MODEL_BACKEND_COQUI},
    )
    frames = svc._synthesize(req)
    emitted = [a["char_index"] for fr in frames for a in fr.alignments]
    assert emitted == list(range(3))
    assert len(emitted) == len(set(emitted))
    assert [a["grapheme"] for fr in frames for a in fr.alignments] == ["A", "B", "C"]


@pytest.mark.asyncio
async def test_gateway_handles_xtts_selection(monkeypatch):
    from gateway.app import ws_handler

    request_payloads = []

    class CollectingChannel(DummyChannel):
        pass

    async def fake_synth_stream(channel, request):
        request_payloads.append(request)
        return DummyCall([
            tts_pb2.SynthesizeResponse(
                request_id=request.request_id,
                chunk_index=0,
                audio_frame=b"",
                alignments=[],
                is_last=True,
                frames_sent=1,
            )
        ])

    monkeypatch.setattr("gateway.app.synthesize_stream", fake_synth_stream)
    monkeypatch.setattr("gateway.app._candidate_tts_targets", lambda: ["tts_native:50052"])
    monkeypatch.setattr("gateway.app._grpc_channel", lambda addr: CollectingChannel(DummyCall([])))

    async def fake_normalize(_client, text):
        return text
    monkeypatch.setattr("gateway.app.normalize_text", fake_normalize)

    ws = FakeWebSocket()
    await ws.queue_message({
        "text": "Hola",
        "flush": True,
        "model": "xtts",
        "speaker_wav": "/tmp/ref.wav",
        "lang": "es",
    })

    await ws_handler(ws)

    assert request_payloads, "No synthesize request sent"
    req = request_payloads[0]
    assert req.backend == tts_pb2.MODEL_BACKEND_XTTS
    assert req.speaker_wav == "/tmp/ref.wav"
    assert req.lang == "es"


def test_xtts_backend_not_initialized(monkeypatch):
    monkeypatch.setenv("TTS_BACKEND", "xtts")
    monkeypatch.delenv("XTTS_CONFIG_JSON", raising=False)
    with pytest.raises(RuntimeError):
        NativeAlignTTS()
