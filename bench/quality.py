import numpy as np


def real_time_factor(samples: int, sample_rate_hz: int, wall_ms: float) -> float:
    audio_ms = (samples / sample_rate_hz) * 1000.0
    return (audio_ms / max(1.0, wall_ms))


def loudness_and_clip(pcm16_bytes: bytes) -> dict:
    arr = np.frombuffer(pcm16_bytes, dtype=np.int16)
    peak = float(np.max(np.abs(arr))) if arr.size else 0.0
    clipped = bool(np.any(arr == 32767) or np.any(arr == -32768))
    rms = float(np.sqrt(np.mean(arr.astype(np.float32) ** 2))) if arr.size else 0.0
    return {"peak": peak, "rms": rms, "clipped": clipped}


