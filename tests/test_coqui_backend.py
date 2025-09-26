import numpy as np
from tts_native.coqui_backend import CoquiNativeBackend

def test_attention_to_char_alignment_mock():
    backend = CoquiNativeBackend()
    attn = np.zeros((10, 4))
    peaks = [1, 3, 5, 7]
    for c, peak in enumerate(peaks):
        attn[peak - 1:peak + 2, c] = [0.2, 1.0, 0.2]
    alignments = backend._attention_to_char_alignment(attn, len(peaks))
    assert len(alignments) == len(peaks)
    starts = [start for start, _ in alignments]
    assert starts == sorted(starts)
    for start_ms, duration_ms in alignments:
        assert duration_ms >= backend._ap.hop_length / backend._ap.sample_rate * 1000.0


def test_short_text_alignment_duration():
    backend = CoquiNativeBackend()
    result = backend.synthesize("This")
    total_ms = len(result.audio_44k) / result.sample_rate * 1000 if result.sample_rate else 0
    assert total_ms < 1000
    for start_ms, duration_ms in result.alignments_ms:
        assert start_ms >= 0
        assert duration_ms >= 0
        assert start_ms + duration_ms <= total_ms + 1e-3
