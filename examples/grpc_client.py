import argparse
import asyncio
import os
import sys
import time
import wave
from pathlib import Path

import grpc

# Add generated protos to path
sys.path.append(str(Path(__file__).resolve().parents[1] / "protos" / "gen" / "python"))
import tts_pb2
import tts_pb2_grpc


def write_wav(path: Path, pcm_bytes: bytes, sample_rate: int) -> None:
    """Saves PCM bytes to a WAV file."""
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)


async def run_grpc_test(
    server_addr: str, text: str, speaker_wav: str, lang: str, output_path: Path
) -> None:
    """Connects to the gRPC server and runs a synthesis test."""
    print(f"Connecting to gRPC server at {server_addr}...")
    audio_chunks = []
    start_time = time.time()
    ttfa = None

    try:
        async with grpc.aio.insecure_channel(server_addr) as channel:
            stub = tts_pb2_grpc.TTSServiceStub(channel)
            request = tts_pb2.SynthesizeRequest(
                request_id="grpc-test-xtts",
                text=text,
                backend=tts_pb2.MODEL_BACKEND_XTTS,
                speaker_wav=speaker_wav,
                lang=lang,
                encoding=tts_pb2.PCM16,
                sample_rate_hz=44100,
            )

            print("Sending SynthesizeStream request...")
            async for resp in stub.SynthesizeStream(request, timeout=180.0):
                if ttfa is None:
                    ttfa = time.time() - start_time
                    print(f"Time to first audio chunk: {ttfa:.2f}s")

                if resp.audio_frame:
                    audio_chunks.append(resp.audio_frame)

                if resp.is_last:
                    print("Received final chunk.")
                    break
        if not audio_chunks:
            print("No audio received; retrying without speaker_wav to use backend default...")
            # Retry without external speaker path
            async with grpc.aio.insecure_channel(server_addr) as channel:
                stub = tts_pb2_grpc.TTSServiceStub(channel)
                request2 = tts_pb2.SynthesizeRequest(
                    request_id="grpc-test-xtts-retry",
                    text=text,
                    backend=tts_pb2.MODEL_BACKEND_XTTS,
                    speaker_wav="",
                    lang=lang,
                    encoding=tts_pb2.PCM16,
                    sample_rate_hz=44100,
                )
                async for resp in stub.SynthesizeStream(request2, timeout=180.0):
                    if resp.audio_frame:
                        audio_chunks.append(resp.audio_frame)
                    if resp.is_last:
                        break
    except grpc.aio.AioRpcError as e:
        print(f"gRPC call failed: {e.code()} - {e.details()}")
        return
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return

    end_time = time.time()
    print(f"Stream finished in {end_time - start_time:.2f}s")

    if not audio_chunks:
        print("Test failed: No audio received from the server.")
    else:
        pcm_bytes = b"".join(audio_chunks)
        write_wav(output_path, pcm_bytes, 44100)
        duration_s = len(pcm_bytes) / (44100 * 2)
        print(
            f"Test successful: Saved {duration_s:.2f}s of audio to {output_path}"
        )


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Direct gRPC client for the tts_native service."
    )
    parser.add_argument(
        "--server",
        default="localhost:50052",
        help="Address of the tts_native gRPC server.",
    )
    parser.add_argument(
        "--text",
        default="This is a test of the emergency broadcast system.",
        help="Text to synthesize.",
    )
    parser.add_argument(
        "--speaker_wav",
        default="voices/Josh.wav",
        help="Path to the reference speaker WAV file.",
    )
    parser.add_argument(
        "--lang", default="en", help="Language code for synthesis."
    )
    parser.add_argument(
        "--output",
        default="grpc_xtts_output.wav",
        help="Path to save the output WAV file.",
    )
    args = parser.parse_args()

    speaker_path = Path(args.speaker_wav)
    if not speaker_path.exists():
        print(f"Error: Speaker WAV not found at '{speaker_path.resolve()}'")
        sys.exit(1)

    await run_grpc_test(
        args.server, args.text, str(speaker_path.resolve()), args.lang, Path(args.output)
    )


if __name__ == "__main__":
    asyncio.run(main())
