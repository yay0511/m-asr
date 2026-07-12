from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(slots=True)
class AudioChunk:
    chunk_id: int
    start: float
    end: float
    waveform: np.ndarray
    sample_rate: int
    is_final: bool = True

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(slots=True)
class AsrResult:
    chunk_id: int
    text: str
    is_final: bool
    latency_ms: float | None = None


@dataclass(slots=True)
class SpeakerResult:
    chunk_id: int
    speaker_id: str
    confidence: float
    embedding: np.ndarray | None = None


@dataclass(slots=True)
class TranscriptTurn:
    start: float
    end: float
    speaker_id: str
    text: str
    confidence: float


@dataclass(slots=True)
class PipelineEvent:
    event_type: Literal["chunk_finalized", "partial", "speaker", "final", "error"]
    chunk_id: int
    start: float
    end: float
    speaker_id: str = "UNKNOWN"
    text: str = ""
    confidence: float = 0.0
    message: str = ""

    def format_turn(self) -> str:
        return f"[{self.start:.2f} - {self.end:.2f}] {self.speaker_id}: {self.text}"
