from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class VadEvent:
    type: str
    sample: int


class SileroVadStream:
    """Small adapter around silero-vad's streaming VADIterator."""

    def __init__(
        self,
        sample_rate: int = 16000,
        threshold: float = 0.5,
        min_silence_duration_ms: int = 700,
        speech_pad_ms: int = 0,
    ):
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.min_silence_duration_ms = min_silence_duration_ms
        self.speech_pad_ms = speech_pad_ms
        self._sample_offset = 0

        try:
            import torch
            from silero_vad import VADIterator, load_silero_vad
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "silero-vad is required when chunker.vad_provider is 'silero'. "
                "Install it in this project environment with `uv pip install silero-vad` "
                "or run `uv sync` after adding it to pyproject.toml."
            ) from exc

        self._torch = torch
        self._model = load_silero_vad()
        self._iterator_cls = VADIterator
        self._iterator = self._make_iterator()

    def reset(self, sample_offset: int = 0) -> None:
        self._sample_offset = int(sample_offset)
        if hasattr(self._iterator, "reset_states"):
            self._iterator.reset_states()
        else:
            self._iterator = self._make_iterator()

    def accept(self, samples: np.ndarray) -> list[dict[str, int | str]]:
        frame = np.asarray(samples, dtype=np.float32).reshape(-1)
        if frame.size == 0:
            return []

        tensor = self._torch.from_numpy(frame)
        result = self._iterator(tensor, return_seconds=False)
        return self._normalize_result(result)

    def _make_iterator(self) -> Any:
        return self._iterator_cls(
            self._model,
            threshold=self.threshold,
            sampling_rate=self.sample_rate,
            min_silence_duration_ms=self.min_silence_duration_ms,
            speech_pad_ms=self.speech_pad_ms,
        )

    def _normalize_result(self, result: Any) -> list[dict[str, int | str]]:
        if result is None:
            return []
        raw_events = result if isinstance(result, list) else [result]
        events: list[dict[str, int | str]] = []
        for event in raw_events:
            if not isinstance(event, dict):
                continue
            if "start" in event:
                events.append({"type": "start", "sample": self._sample_offset + int(event["start"])})
            if "end" in event:
                events.append({"type": "end", "sample": self._sample_offset + int(event["end"])})
        return events
