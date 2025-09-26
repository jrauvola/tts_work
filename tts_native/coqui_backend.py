import os
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple, Optional

import librosa
import numpy as np
import torch
from TTS.api import TTS
from tts_native.synthesis import SynthesisResult


class CoquiNativeBackend:
    def __init__(self, model_name: str = None) -> None:
        self._model_name = model_name or os.environ.get("COQUI_MODEL", "tts_models/en/ljspeech/tacotron2-DDC")
        requested_device = os.environ.get("COQUI_DEVICE", "auto").lower()
        if requested_device == "gpu":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif requested_device == "cpu":
            device = "cpu"
        elif requested_device == "cuda":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self._tts = TTS(self._model_name, gpu=device == "cuda")
        self._device = device
        self._model = self._tts.synthesizer.tts_model
        if hasattr(self._model, "to"):
            self._model.to(device)
        self._gate_threshold = float(os.environ.get("COQUI_GATE_THRESHOLD", "0.5"))
        self._model.decoder.max_decoder_steps = int(os.environ.get("COQUI_MAX_DECODER_STEPS", "400"))
        if hasattr(self._model.decoder, "gate_threshold"):
            self._model.decoder.gate_threshold = self._gate_threshold
        if hasattr(self._model.decoder, "stop_threshold"):
            self._model.decoder.stop_threshold = self._gate_threshold
        self._vocoder = self._tts.synthesizer.vocoder_model
        if hasattr(self._vocoder, "to"):
            self._vocoder.to(device)
        self._ap = self._model.ap
        self._vocoder_ap = self._tts.synthesizer.vocoder_ap
        self._target_sr = 44100

    def synthesize(self, text: str) -> SynthesisResult:
        tokens = self._model.tokenizer.text_to_ids(text)
        if not tokens:
            return SynthesisResult(audio_44k=np.zeros(0, dtype=np.float32), sample_rate=self._target_sr, chars=[], alignments_ms=[])
        token_tensor = torch.tensor(tokens, device=self._device).unsqueeze(0)
        aux = self._model._format_aux_input({"speaker_ids": None, "d_vectors": None})
        outputs = self._model.inference(token_tensor, aux)
        align = outputs["alignments"][0].detach().cpu().numpy()
        mel = outputs["model_outputs"][0].detach().cpu().numpy()
        stop_tokens = outputs.get("stop_tokens")
        stop_tokens_np = stop_tokens[0].detach().cpu().numpy() if stop_tokens is not None else None
        effective_steps = self._determine_effective_steps(align, len(tokens), stop_tokens_np)
        align = align[:effective_steps]
        mel = mel[:effective_steps]
        if stop_tokens_np is not None:
            stop_tokens_np = stop_tokens_np[:effective_steps]
        mel = self._ap.denormalize(mel.T).T
        vocoder_in = self._vocoder_ap.normalize(mel.T)
        vocoder_in = torch.tensor(vocoder_in, device=self._device).unsqueeze(0)
        self._vocoder.eval()
        with torch.no_grad():
            waveform = self._vocoder.inference(vocoder_in).detach().cpu().numpy().squeeze()
        waveform = self._resample_waveform(waveform, self._vocoder_ap.sample_rate)
        waveform, offset_ms, total_ms = self._trim_waveform(waveform)
        chars = [self._model.tokenizer.characters.id_to_char(i) for i in tokens]
        align_pairs = self._attention_to_char_alignment(align, len(chars))
        align_pairs = self._clip_alignments(align_pairs, offset_ms, total_ms if total_ms > 0.0 else len(waveform) / self._target_sr * 1000.0)
        return SynthesisResult(audio_44k=waveform, sample_rate=self._target_sr, chars=chars, alignments_ms=align_pairs)

    def _resample_waveform(self, waveform: np.ndarray, src_sr: int) -> np.ndarray:
        if src_sr == self._target_sr:
            return waveform.astype(np.float32)
        return librosa.resample(waveform, orig_sr=src_sr, target_sr=self._target_sr).astype(np.float32)

    def _trim_waveform(self, waveform: np.ndarray, top_db: float = 40.0) -> Tuple[np.ndarray, float, float]:
        if waveform.size == 0:
            return waveform.astype(np.float32), 0.0, 0.0
        trimmed, idx = librosa.effects.trim(waveform, top_db=top_db)
        start_sample, end_sample = int(idx[0]), int(idx[1])
        offset_ms = start_sample / self._target_sr * 1000.0
        total_ms = (end_sample - start_sample) / self._target_sr * 1000.0
        return trimmed.astype(np.float32), offset_ms, total_ms

    def _clip_alignments(self, align_pairs: List[Tuple[float, float]], offset_ms: float, total_ms: float) -> List[Tuple[float, float]]:
        if not align_pairs:
            return []
        adjusted: List[Tuple[float, float]] = []
        prev_end = 0.0
        limit = max(total_ms, 0.0)
        for start_ms, duration_ms in align_pairs:
            start_adj = max(0.0, start_ms - offset_ms)
            end_adj = start_adj + max(0.0, duration_ms)
            if limit > 0.0:
                end_adj = min(end_adj, limit)
            if start_adj < prev_end:
                start_adj = prev_end
            if end_adj < start_adj:
                end_adj = start_adj
            adjusted.append((start_adj, end_adj - start_adj))
            prev_end = end_adj
        if limit > 0.0 and adjusted:
            last_start, last_dur = adjusted[-1]
            last_end = min(limit, last_start + last_dur)
            adjusted[-1] = (last_start, max(0.0, last_end - last_start))
        return adjusted

    def _effective_length(self, stop_tokens: np.ndarray) -> int:
        if stop_tokens.size == 0:
            return 0
        threshold = self._gate_threshold
        for idx, val in enumerate(stop_tokens):
            if val > threshold:
                return max(1, idx + 1)
        return len(stop_tokens)

    def _determine_effective_steps(self, attn: np.ndarray, char_len: int, stop_tokens: Optional[np.ndarray]) -> int:
        steps = attn.shape[0]
        if stop_tokens is not None and stop_tokens.size:
            steps = min(steps, self._effective_length(stop_tokens))
        if char_len > 0 and steps > 0:
            step_assign = np.argmax(attn[:steps], axis=1)
            step_assign = np.clip(step_assign, 0, char_len - 1)
            step_assign = np.maximum.accumulate(step_assign)
            last_char = char_len - 1
            last_occurrences = np.where(step_assign == last_char)[0]
            if last_occurrences.size:
                plateau = int(os.environ.get("COQUI_LAST_CHAR_PLATEAU_STEPS", "20"))
                candidate = last_occurrences[0] + plateau
                steps = min(steps, max(1, candidate))
        return max(1, steps)

    def _attention_to_char_alignment(self, attn: np.ndarray, char_len: int) -> List[Tuple[float, float]]:
        decoder_steps, _ = attn.shape
        hop_ms = (self._ap.hop_length / self._ap.sample_rate) * 1000.0
        if decoder_steps == 0 or char_len == 0:
            return [(0.0, 0.0) for _ in range(char_len)]

        # Determine most likely token index per decoder step and enforce monotonicity
        step_assign = np.argmax(attn, axis=1)
        step_assign = np.clip(step_assign, 0, char_len - 1)
        step_assign = np.maximum.accumulate(step_assign)

        starts: List[float] = []
        ends: List[float] = []

        for token_idx in range(char_len):
            indices = np.where(step_assign == token_idx)[0]
            if indices.size == 0:
                if ends:
                    start_ms = ends[-1]
                else:
                    start_ms = token_idx * hop_ms
                end_ms = start_ms + hop_ms
            else:
                start_step = int(indices[0])
                end_step = int(indices[-1]) + 1
                start_ms = start_step * hop_ms
                end_ms = end_step * hop_ms
            starts.append(start_ms)
            ends.append(end_ms)

        # Enforce strictly non-decreasing boundaries and ensure minimum duration
        min_dur = hop_ms
        for i in range(char_len):
            if i > 0 and starts[i] < ends[i - 1]:
                starts[i] = ends[i - 1]
            if ends[i] < starts[i] + min_dur:
                ends[i] = starts[i] + min_dur

        alignments: List[Tuple[float, float]] = []
        for start_ms, end_ms in zip(starts, ends):
            duration_ms = max(min_dur, end_ms - start_ms)
            alignments.append((start_ms, duration_ms))

        return alignments
