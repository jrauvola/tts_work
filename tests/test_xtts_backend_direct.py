import os
import pathlib
import sys
from pathlib import Path

import pytest

# Add project root to path to allow importing tts_native
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tts_native.xtts_backend import XttsBackend


@pytest.mark.slow
def test_xtts_backend_direct_synthesis(tmp_path: pathlib.Path) -> None:
    """
    Tests the XttsBackend class directly, bypassing the server and scheduler.
    This helps verify that model loading and synthesis work in the current environment.
    """
    # This test will download the model to the default cache location on first run
    model_root = tmp_path / "models" / "XTTS-v2"
    speaker_wav = model_root / "reference_sample.wav"
    output_wav = tmp_path / "direct_output.wav"

    # Set environment variables to control model location
    os.environ["XTTS_MODEL_ROOT"] = str(model_root)

    print(f"Initializing XttsBackend with model root: {model_root}")
    backend = XttsBackend(
        model_root=str(model_root), speaker_wav=str(speaker_wav)
    )
    print("XttsBackend initialized.")

    text_to_synthesize = "Hello from the direct backend test."
    print(f"Synthesizing text: '{text_to_synthesize}'")

    result = backend.synthesize(text=text_to_synthesize)

    print("Synthesis complete.")
    assert result.audio_44k is not None, "Synthesis result should have audio data."
    assert (
        len(result.audio_44k) > 1000
    ), "Synthesized audio should not be empty."

    # For manual verification, you could save the file:
    # import wave
    # with wave.open(str(output_wav), "wb") as wf:
    #     wf.setnchannels(1)
    #     wf.setsampwidth(2)
    #     wf.setframerate(result.sample_rate)
    #     wf.writeframes((result.audio_44k * 32767).astype(np.int16).tobytes())
    # print(f"Audio saved to {output_wav}")

    print("Direct synthesis test passed.")
