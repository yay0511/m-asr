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
        if device_name == "cuda" and not _torch_cuda_available(torch):
            device_name = "cpu"
        self._embedding = self._load_embedding(
            PretrainedSpeakerEmbedding,
            model_dir,
            torch.device(device_name),
        )

    def _load_embedding(self, factory: object, model_dir: Path, device: object) -> object:
        try:
            return factory(str(model_dir), device=device)
        except RuntimeError as exc:
            if str(device) == "cuda" and _is_cuda_runtime_error(exc):
                import torch

                self.config.runtime.device = "cpu"
                print(
                    "[warn  ] CUDA failed while loading pyannote embedding; "
                    f"falling back to CPU. {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
                return factory(str(model_dir), device=torch.device("cpu"))
            raise

    def _extract_real(self, chunk: AudioChunk) -> np.ndarray | None:
        import torch

        if self._embedding is None:
            return None
        waveform = np.asarray(chunk.waveform, dtype=np.float32).reshape(1, 1, -1)
        if waveform.shape[-1] == 0:
            return None
        tensor = torch.from_numpy(waveform)
        try:
            embedding = self._embedding(tensor)
        except RuntimeError as exc:
            if self.config.runtime.device == "cuda" and _is_cuda_runtime_error(exc):
                self.config.runtime.device = "cpu"
                print(
                    "[warn  ] CUDA failed during pyannote embedding inference; "
                    "falling back to CPU and retrying this chunk. "
                    f"{type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
                self._move_embedding_to_cpu()
                embedding = self._embedding(tensor)
            else:
                raise
        if embedding is None or len(embedding) == 0:
            return None
        return np.asarray(embedding[0], dtype=np.float32)

    def _move_embedding_to_cpu(self) -> None:
        if self._embedding is None:
            return
        try:
            import torch

            if hasattr(self._embedding, "to"):
                self._embedding.to(torch.device("cpu"))
        except RuntimeError:
            raise


def _torch_cuda_available(torch_module: object) -> bool:
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="CUDA initialization:.*", category=UserWarning)
            return bool(torch_module.cuda.is_available())
    except BaseException:
        return False


def _is_cuda_runtime_error(exc: RuntimeError) -> bool:
    message = str(exc).lower()
    return (
        "cuda" in message
        or "nvidia driver" in message
        or "driver on your system is too old" in message
    )
