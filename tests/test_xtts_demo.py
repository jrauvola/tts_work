import os
import pathlib
import shutil
import ssl
import urllib.request

import certifi
import pytest
from TTS.api import TTS
from TTS.utils.manage import ModelManager


MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
REFERENCE_URL = "https://huggingface.co/coqui/XTTS-v2/resolve/main/samples/en_sample.wav?download=1"
DEFAULT_TEXT = (
    "It took me quite a long time to develop a voice, "
    "and now that I have it I'm not going to be silent."
)


def _download_reference_clip(reference_path: pathlib.Path) -> None:
    if reference_path.exists():
        return
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(REFERENCE_URL, context=ssl_context) as response, reference_path.open("wb") as target:
        shutil.copyfileobj(response, target)


def _auto_accept_terms() -> None:
    def _ask_tos(model_full_path: pathlib.Path) -> bool:
        tos_path = model_full_path / "tos_agreed.txt"
        tos_path.parent.mkdir(parents=True, exist_ok=True)
        with tos_path.open("w", encoding="utf-8") as handle:
            handle.write("Agreed to Coqui terms via automated test.\n")
        return True

    ModelManager.ask_tos = staticmethod(_ask_tos)  # type: ignore[method-assign]


@pytest.mark.slow
def test_xtts_cpu_demo(tmp_path: pathlib.Path) -> None:
    os.environ.setdefault("COQUI_TOS_ACCEPTED", "1")
    os.environ.setdefault("COQUI_LICENSE_ACCEPTED", "1")

    reference_path = tmp_path / "reference_sample.wav"
    output_path = tmp_path / "xtts_output.wav"

    _download_reference_clip(reference_path)
    _auto_accept_terms()

    tts = TTS(model_name=MODEL_NAME, gpu=False)
    tts.tts_to_file(
        text=DEFAULT_TEXT,
        file_path=str(output_path),
        speaker_wav=str(reference_path),
        language="en",
    )

    assert output_path.exists(), "XTTS demo did not produce an output file"
    size_bytes = output_path.stat().st_size
    assert size_bytes > 10_000, f"Expected synthesized audio to be >10 KB, got {size_bytes} bytes"

