import argparse
import asyncio
import base64
import json
import wave
from pathlib import Path

import websockets


def write_wav(path: Path, pcm_bytes: bytes, sample_rate: int) -> None:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)


async def synthesize(url: str, text: str) -> None:
    audio_chunks: list[bytes] = []
    alignment_accum = {
        "chars": [],
        "char_start_times_ms": [],
        "char_durations_ms": [],
    }
    sample_rate = 44100

    async with websockets.connect(url, ping_interval=20) as ws:
        # protocol: send whitespace keepalive, then text
        await ws.send(json.dumps({"text": " ", "flush": False}))
        await ws.send(json.dumps({"text": text, "flush": True}))

        while True:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
            except (asyncio.TimeoutError, websockets.ConnectionClosed):
                break

            data = json.loads(msg)

            if "audio" in data:
                fmt = data.get("format", {})
                sample_rate = int(fmt.get("sample_rate_hz", sample_rate))
                audio_chunks.append(base64.b64decode(data["audio"]))
                alignment = data.get("alignment", {})
                alignment_accum["chars"].extend(alignment.get("chars", []))
                alignment_accum["char_start_times_ms"].extend(alignment.get("char_start_times_ms", []))
                alignment_accum["char_durations_ms"].extend(alignment.get("char_durations_ms", []))
                if "ttfa_ms" in data:
                    print(f"TTFA: {data['ttfa_ms']:.1f} ms")
            elif data.get("status") == "done":
                await ws.close()
                break
            elif "error" in data:
                print("Gateway error:", data["error"])

    if audio_chunks:
        pcm_bytes = b"".join(audio_chunks)
        out_path = Path("coqui_native_output.wav")
        write_wav(out_path, pcm_bytes, sample_rate)
        print(f"Saved audio to {out_path} ({len(pcm_bytes)/2/sample_rate:.2f} s)")

    if any(alignment_accum.values()):
        print("Alignments:")
        print(json.dumps(alignment_accum, indent=2))


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("text", nargs="?", default="Hello, world!", help="Text to synthesize")
    parser.add_argument("--url", default="ws://localhost:8000/ws?api_key=dev-key")
    args = parser.parse_args()

    await synthesize(args.url, args.text)


if __name__ == "__main__":
    asyncio.run(main())


