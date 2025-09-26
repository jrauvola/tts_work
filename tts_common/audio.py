import math
from typing import List

PCM16_MAX = 32767


def pcm16_sine(frequency_hz: float, duration_ms: int, sample_rate: int = 16000, amplitude: float = 0.2) -> bytes:
    total_samples = int(sample_rate * (duration_ms / 1000.0))
    amp = int(PCM16_MAX * max(0.0, min(amplitude, 1.0)))
    buf = bytearray()
    for n in range(total_samples):
        value = int(amp * math.sin(2.0 * math.pi * frequency_hz * (n / sample_rate)))
        # little-endian signed 16-bit
        buf += value.to_bytes(2, byteorder="little", signed=True)
    return bytes(buf)


def frame_count_for_ms(total_ms: int, frame_ms: int) -> int:
    return max(1, math.ceil(total_ms / frame_ms))


def generate_pcm_silence_frames(total_ms: int, frame_ms: int = 20, sample_rate: int = 16000) -> List[bytes]:
    num_frames = frame_count_for_ms(total_ms, frame_ms)
    samples_per_frame = int(sample_rate * (frame_ms / 1000.0))
    frame = (0).to_bytes(2, byteorder="little", signed=True) * samples_per_frame
    return [frame for _ in range(num_frames)]


