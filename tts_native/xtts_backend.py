import os
import pathlib
import shutil
import ssl
from typing import Optional

import certifi
import librosa
import numpy as np
import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts, XttsAudioConfig, XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.utils.manage import ModelManager

from tts_native.synthesis import SynthesisResult

torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs])


class XttsBackendError(RuntimeError):
    """Raised when XTTS backend cannot be initialized."""


class XttsBackend:
    MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
    REFERENCE_URL = (
        "https://huggingface.co/coqui/XTTS-v2/resolve/main/samples/en_sample.wav?download=1"
    )

    def __init__(
        self,
        config_path: Optional[str] = None,
        checkpoint_dir: Optional[str] = None,
        speaker_wav: Optional[str] = None,
        model_root: Optional[str] = None,
    ) -> None:
        os.environ.setdefault("COQUI_TOS_ACCEPTED", "1")
        os.environ.setdefault("COQUI_LICENSE_ACCEPTED", "1")

        self._model_root = pathlib.Path(model_root or os.environ.get("XTTS_MODEL_ROOT", "models/XTTS-v2")).resolve()
        self._model_root.mkdir(parents=True, exist_ok=True)

        self._config_path = pathlib.Path(config_path or os.environ.get("XTTS_CONFIG_JSON") or (self._model_root / "config.json"))
        self._checkpoint_dir = pathlib.Path(checkpoint_dir or os.environ.get("XTTS_CHECKPOINT_DIR") or self._model_root)
        self._default_speaker = speaker_wav or os.environ.get("XTTS_SPEAKER_WAV")
        if not self._default_speaker:
            self._default_speaker = str(self._model_root / "reference_sample.wav")

        self._ensure_model_assets()

        if not self._config_path.exists():
            raise XttsBackendError(f"XTTS config not found at {self._config_path}")
        if not self._checkpoint_dir.is_dir():
            raise XttsBackendError(f"XTTS checkpoint directory not found at {self._checkpoint_dir}")
        if not self._default_speaker or not os.path.exists(self._default_speaker):
            raise XttsBackendError("XTTS reference speaker wav is required")

        self._config = XttsConfig()
        self._config.load_json(str(self._config_path))
        self._model = Xtts.init_from_config(self._config)
        self._model.load_checkpoint(self._config, checkpoint_dir=str(self._checkpoint_dir), eval=True)

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

    def _ensure_model_assets(self) -> None:
        manager = ModelManager()

        if not self._config_path.exists() or not any(self._checkpoint_dir.glob("*.pth")):
            download_result = manager.download_model(self.MODEL_NAME)
            # Handle newer TTS versions returning a tuple (path, config)
            if isinstance(download_result, tuple):
                model_path = download_result[0]
            else:
                model_path = download_result

            if model_path is None:
                raise XttsBackendError(f"Failed to download XTTS model '{self.MODEL_NAME}'")
            model_path = pathlib.Path(model_path)
            # copy configs and checkpoints into model_root
            for item in model_path.iterdir():
                target = self._model_root / item.name
                if item.is_dir():
                    shutil.copytree(item, target, dirs_exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(item, target)

        reference_path = pathlib.Path(self._default_speaker)
        if not reference_path.exists():
            reference_path.parent.mkdir(parents=True, exist_ok=True)
            ssl_context = ssl.create_default_context(cafile=certifi.where())
            import urllib.request

            with urllib.request.urlopen(self.REFERENCE_URL, context=ssl_context) as response, reference_path.open("wb") as target:
                target.write(response.read())

    def synthesize(self, text: str, speaker_wav: Optional[str] = None, language: Optional[str] = None) -> SynthesisResult:
        ref_wav = speaker_wav or self._default_speaker
        if not ref_wav or not os.path.exists(ref_wav):
            raise XttsBackendError("XTTS reference speaker wav is required")

        outputs = self._model.synthesize(
            text,
            speaker_wav=ref_wav,
            language=language or self._language,
            gpt_cond_len=self._gpt_cond_len,
            temperature=self._temperature,
        )

        waveform = np.array(outputs["wav"], dtype=np.float32)
        sample_rate = self._config.audio.sample_rate

        if sample_rate != self._target_sr:
            waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=self._target_sr).astype(np.float32)
            sample_rate = self._target_sr

        chars = list(text)
        return SynthesisResult(audio_44k=waveform, sample_rate=sample_rate, chars=chars, alignments_ms=[])


