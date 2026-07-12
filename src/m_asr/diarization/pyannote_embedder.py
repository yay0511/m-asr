from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

from m_asr.config import AppConfig
from m_asr.types import AudioChunk


class PyannoteSpeakerEmbedder:
    """Speaker embedding adapter backed by local pyannote embedding model."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.mode = config.speaker.mode
        self._embedding = None
        self._backend = "real"
        self._init_error: str | None = None

        if self.mode != "real":
            raise ValueError(f"speaker mode must be 'real', got {self.mode!r}")
        self._init_real()

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def init_error(self) -> str | None:
        return self._init_error

    def extract(self, chunk: AudioChunk) -> np.ndarray | None:
        if chunk.duration < self.config.speaker.min_embedding_duration:
            return None
        return self._extract_real(chunk)

    def _init_real(self) -> None:
        import torch

        pyannote_src = Path(self.config.paths.pyannote_audio_root) / "src"
        if pyannote_src.exists() and str(pyannote_src) not in sys.path:
            sys.path.insert(0, str(pyannote_src))

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"\s*torchcodec is not installed correctly.*",
                category=UserWarning,
            )
            from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding

        model_dir = Path(self.config.paths.pyannote_model_dir) / "embedding"
        if not model_dir.exists():
            raise FileNotFoundError(f"pyannote embedding model not found: {model_dir}")

        device_name = self.config.runtime.device
        if device_name == "cuda" and not torch.cuda.is_available():
            device_name = "cpu"
        device = torch.device(device_name)
        self._embedding = PretrainedSpeakerEmbedding(str(model_dir), device=device)

    def _extract_real(self, chunk: AudioChunk) -> np.ndarray | None:
        import torch

        if self._embedding is None:
            return None
        waveform = np.asarray(chunk.waveform, dtype=np.float32).reshape(1, 1, -1)
        if waveform.shape[-1] == 0:
            return None
        tensor = torch.from_numpy(waveform)
        embedding = self._embedding(tensor)
        if embedding is None or len(embedding) == 0:
            return None
        return np.asarray(embedding[0], dtype=np.float32)
