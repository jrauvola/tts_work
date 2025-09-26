import asyncio
import os
from typing import Dict, List, Optional

import grpc

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "protos", "gen", "python"))

import tts_pb2  # type: ignore
import tts_pb2_grpc  # type: ignore

from tts_common.scheduler import DynamicBatchScheduler, ScheduledRequest, SynthFrame
from prometheus_client import start_http_server
from aiohttp import web
import numpy as np
from tts_native.coqui_backend import CoquiNativeBackend
from tts_native.xtts_backend import XttsBackend, XttsBackendError
from tts_native.synthesis import SynthesisResult

FRAME_MS = int(os.environ.get("FRAME_MS", "20"))
TARGET_SAMPLE_RATE = 44100


def pcm16_from_float(audio: np.ndarray) -> bytes:
    audio = np.clip(audio, -1.0, 1.0)
    return (audio * 32767.0).astype(np.int16).tobytes()


def chunk_audio(audio: np.ndarray, sample_rate: int, frame_ms: int) -> List[np.ndarray]:
    samples_per_frame = int(sample_rate * frame_ms / 1000.0)
    if samples_per_frame <= 0:
        return [audio]
    return [audio[i : i + samples_per_frame] for i in range(0, len(audio), samples_per_frame)]


class NativeAlignTTS(tts_pb2_grpc.TTSServiceServicer):
    def __init__(self) -> None:
        self._backends: Dict[str, object] = {"coqui": CoquiNativeBackend()}
        configured_backend = os.environ.get("TTS_BACKEND", "coqui").lower()
        self._default_backend_name = configured_backend if configured_backend in {"coqui", "xtts"} else "coqui"
        if self._default_backend_name == "xtts":
            try:
                self._backends["xtts"] = XttsBackend()
            except XttsBackendError as exc:
                raise RuntimeError(f"Failed to initialize XTTS backend: {exc}")

        async def synth_batch(reqs: List[ScheduledRequest]) -> List[List[SynthFrame]]:
            out: List[List[SynthFrame]] = []
            loop = asyncio.get_running_loop()
            for r in reqs:
                frames = await loop.run_in_executor(None, self._synthesize, r)
                out.append(frames)
            return out

        self._scheduler = DynamicBatchScheduler(
            backend_synthesize_batch=synth_batch,
            scheduler_window_ms=int(os.environ.get("SCHED_WINDOW_MS", "20")),
            max_batch_size=int(os.environ.get("MAX_BATCH", "4")),
        )

    async def start(self) -> None:
        await self._scheduler.start()

    def _synthesize(self, req: ScheduledRequest) -> List[SynthFrame]:
        backend_name = self._resolve_backend(req.params.get("backend") if req.params else None)
        backend = self._ensure_backend(backend_name)
        text = req.text
        speaker_wav = req.params.get("speaker_wav") if req.params else None
        lang = req.params.get("lang") if req.params else None
        if backend_name == "xtts":
            result: SynthesisResult = backend.synthesize(text, speaker_wav=speaker_wav, language=lang)  # type: ignore[attr-defined]
        else:
            result = backend.synthesize(text)  # type: ignore[attr-defined]
        if result.audio_44k.size == 0:
            return [SynthFrame(audio_frame=b"", alignments=[], is_last=True)]
        frames = chunk_audio(result.audio_44k, result.sample_rate, FRAME_MS)
        synth_frames: List[SynthFrame] = []
        produced: set[int] = set()
        for idx, frame in enumerate(frames):
            frame_bytes = pcm16_from_float(frame)
            frame_start = idx * FRAME_MS
            frame_end = frame_start + FRAME_MS
            frame_aligns: List[dict] = []
            if result.alignments_ms:
                for char_idx, (start_ms, duration_ms) in enumerate(result.alignments_ms):
                    if char_idx in produced:
                        continue
                    char_end = start_ms + duration_ms
                    if char_end <= frame_start or start_ms >= frame_end:
                        continue
                    overlap_start = max(start_ms, frame_start)
                    overlap_end = min(char_end, frame_end)
                    frame_aligns.append(
                        {
                            "char_index": char_idx,
                            "grapheme": result.chars[char_idx],
                            "start_ms": start_ms,
                            "duration_ms": duration_ms,
                            "chunk_start_ms": overlap_start,
                            "chunk_duration_ms": max(overlap_end - overlap_start, 0.0),
                        }
                    )
                if frame_aligns:
                    produced.update(a["char_index"] for a in frame_aligns)
            synth_frames.append(
                SynthFrame(
                    audio_frame=frame_bytes,
                    alignments=frame_aligns,
                    is_last=(idx == len(frames) - 1),
                )
            )
        return synth_frames

    def _alignments_for_chunk(self, chunk_idx: int, chars: List[str], align_pairs: List[tuple], produced: set[int]) -> List[dict]:
        # legacy hook kept for compatibility but no longer used
        return []

    async def SynthesizeStream(self, request: tts_pb2.SynthesizeRequest, context: grpc.aio.ServicerContext):
        req = ScheduledRequest(
            request_id=request.request_id,
            chunk_index=request.chunk_index,
            text=request.text,
            sample_rate_hz=TARGET_SAMPLE_RATE,
            frame_ms=FRAME_MS,
            params={
                "backend": request.backend,
                "speaker_wav": request.speaker_wav,
                "lang": request.lang,
            },
        )
        q = await self._scheduler.submit(req)
        frames_sent = 0
        while True:
            fr: SynthFrame = await q.get()
            frames_sent += 1
            chunk_chars = [a["grapheme"] for a in fr.alignments]
            chunk_starts = [a["start_ms"] for a in fr.alignments]
            chunk_durations = [a["duration_ms"] for a in fr.alignments]
            scope_start = request.chunk_index * req.frame_ms + (frames_sent - 1) * req.frame_ms
            scope_end = scope_start + req.frame_ms
            yield tts_pb2.SynthesizeResponse(
                request_id=request.request_id,
                chunk_index=request.chunk_index,
                audio_frame=fr.audio_frame,
                format=tts_pb2.AudioFormat(
                    encoding=request.encoding or tts_pb2.PCM16,
                    sample_rate_hz=TARGET_SAMPLE_RATE,
                    frame_ms=req.frame_ms,
                ),
                alignments=[
                    tts_pb2.CharAlignment(
                        char_index=a["char_index"],
                        grapheme=a["grapheme"],
                        start_ms=a["start_ms"],
                        duration_ms=a["duration_ms"],
                    )
                    for a in fr.alignments
                ],
                is_last=fr.is_last,
                frames_sent=frames_sent,
            )
            if fr.is_last:
                break

    def _resolve_backend(self, backend_enum: Optional[int]) -> str:
        if backend_enum == tts_pb2.MODEL_BACKEND_XTTS:
            return "xtts"
        if backend_enum == tts_pb2.MODEL_BACKEND_COQUI:
            return "coqui"
        return self._default_backend_name

    def _ensure_backend(self, name: str):
        if name in self._backends:
            return self._backends[name]
        if name == "xtts":
            backend = XttsBackend()
            self._backends[name] = backend
            return backend
        if name == "coqui":
            backend = CoquiNativeBackend()
            self._backends[name] = backend
            return backend
        raise RuntimeError(f"Unsupported backend '{name}'")


def serve_coqui_backend():
    return NativeAlignTTS()


async def serve() -> None:
    server = grpc.aio.server(options=[("grpc.max_send_message_length", -1), ("grpc.max_receive_message_length", -1)])
    svc = serve_coqui_backend()
    await svc.start()
    tts_pb2_grpc.add_TTSServiceServicer_to_server(svc, server)
    listen_addr = os.environ.get("LISTEN_ADDR", "0.0.0.0:50052")
    if os.environ.get("TTS_TLS", "0") == "1":
        cert_path = os.environ.get("TTS_TLS_CERT", "/certs/server.crt")
        key_path = os.environ.get("TTS_TLS_KEY", "/certs/server.key")
        try:
            with open(cert_path, "rb") as f:
                cert = f.read()
            with open(key_path, "rb") as f:
                key = f.read()
            creds = grpc.ssl_server_credentials([(key, cert)])
            server.add_secure_port(listen_addr, creds)
        except Exception:
            server.add_insecure_port(listen_addr)
    else:
        server.add_insecure_port(listen_addr)
    start_http_server(int(os.environ.get("METRICS_PORT", "8004")))
    async def start_health_server() -> None:
        app = web.Application()
        async def livez(_):
            return web.json_response({"status": "ok"})
        async def healthz(_):
            return web.json_response({"status": "ready"})
        app.router.add_get('/livez', livez)
        app.router.add_get('/healthz', healthz)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', int(os.environ.get("HTTP_PORT", "8084")))
        await site.start()
    asyncio.create_task(start_health_server())
    await server.start()
    await server.wait_for_termination()


if __name__ == "__main__":
    import asyncio as _asyncio
    _asyncio.run(serve())


