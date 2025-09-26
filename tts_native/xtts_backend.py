import os
from typing import Optional

import librosa
import numpy as np
import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

from tts_native.synthesis import SynthesisResult


if "weights_only" in torch.load.__code__.co_varnames:
    _original_torch_load = torch.load

    def _torch_load_weights_off(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _original_torch_load(*args, **kwargs)

    torch.load = _torch_load_weights_off  # type: ignore


class XttsBackendError(RuntimeError):
    """Raised when XTTS backend cannot be initialized."""


class XttsBackend:
    def __init__(
        self,
        config_path: Optional[str] = None,
        checkpoint_dir: Optional[str] = None,
        speaker_wav: Optional[str] = None,
    ) -> None:
        self._config_path = config_path or os.environ.get("XTTS_CONFIG_JSON")
        self._checkpoint_dir = checkpoint_dir or os.environ.get("XTTS_CHECKPOINT_DIR")
        self._default_speaker = speaker_wav or os.environ.get("XTTS_SPEAKER_WAV")
        if not self._config_path or not self._checkpoint_dir:
            raise XttsBackendError(
                "XTTS backend requires XTTS_CONFIG_JSON and XTTS_CHECKPOINT_DIR to be set"
            )
        if not os.path.exists(self._config_path):
            raise XttsBackendError(f"XTTS config not found at {self._config_path}")
        if not os.path.isdir(self._checkpoint_dir):
            raise XttsBackendError(f"XTTS checkpoint directory not found at {self._checkpoint_dir}")

        self._config = XttsConfig()
        self._config.load_json(self._config_path)
        self._model = Xtts.init_from_config(self._config)
        self._model.load_checkpoint(self._config, checkpoint_dir=self._checkpoint_dir, eval=True)

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        if self._device == "cuda":
            self._model.cuda()
        else:
            self._model.to(torch.device("cpu"))

        self._target_sr = int(os.environ.get("XTTS_TARGET_SR", "44100"))
        self._language = os.environ.get("XTTS_LANGUAGE", "en")
        self._gpt_cond_len = int(os.environ.get("XTTS_GPT_COND_LEN", "3"))
        self._temperature = float(os.environ.get("XTTS_TEMPERATURE", "0.75"))
        self._length_scale = float(os.environ.get("XTTS_LENGTH_SCALE", "1.0"))

    def synthesize(self, text: str, speaker_wav: Optional[str] = None, language: Optional[str] = None) -> SynthesisResult:
        ref_wav = speaker_wav or self._default_speaker
        if not ref_wav or not os.path.exists(ref_wav):
            raise XttsBackendError("XTTS reference speaker wav is required")

        outputs = self._model.synthesize(
            text,
            self._config,
            speaker_wav=ref_wav,
            language=language or self._language,
            gpt_cond_len=self._gpt_cond_len,
            temperature=self._temperature,
            length_scale=self._length_scale,
        )

        waveform = np.array(outputs["wav"], dtype=np.float32)
        sample_rate = outputs.get("sample_rate") or getattr(self._config.audio, "sample_rate", self._target_sr)

        if sample_rate != self._target_sr:
            waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=self._target_sr).astype(np.float32)
            sample_rate = self._target_sr

        chars = list(text)
        return SynthesisResult(audio_44k=waveform, sample_rate=sample_rate, chars=chars, alignments_ms=[])


