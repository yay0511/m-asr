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
    core_start: float | None = None
    core_end: float | None = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def speech_start(self) -> float:
        return self.core_start if self.core_start is not None else self.start

    @property
    def speech_end(self) -> float:
        return self.core_end if self.core_end is not None else self.end


@dataclass(slots=True)
class AsrResult:
    chunk_id: int
    text: str
    is_final: bool
    latency_ms: float | None = None


@dataclass(slots=True)
class TimestampedCharacter:
    text: str
    start: float
    end: float


@dataclass(slots=True)
class TimestampedWord:
    text: str
    start: float
    end: float
    speaker_id: str = "UNKNOWN"
    confidence: float = 0.0


@dataclass(slots=True)
class TimestampedAsrResult:
    chunk_id: int
    text: str
    characters: list[TimestampedCharacter]
    words: list[TimestampedWord]
    status: str = "matched"
    latency_ms: float | None = None


@dataclass(slots=True)
class LocalSpeakerSegment:
    local_id: str
    start: float
    end: float
    confidence: float = 0.0
    overlap: bool = False


@dataclass(slots=True)
class LocalSpeakerTrack:
    local_id: str
    segments: list[LocalSpeakerSegment]
    embedding: np.ndarray | None
    speech_duration: float
    confidence: float = 0.0


@dataclass(slots=True)
class LocalDiarizationResult:
    chunk_id: int
    start: float
    end: float
    tracks: list[LocalSpeakerTrack]
    latency_ms: float | None = None


@dataclass(slots=True)
class GlobalSpeakerSegment:
    start: float
    end: float
    speaker_id: str
    confidence: float = 0.0
    overlap: bool = False


@dataclass(slots=True)
class LocalSpeakerAssignment:
    local_id: str
    speaker_id: str
    confidence: float
    segment: LocalSpeakerSegment


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
    words: list[TimestampedWord] | None = None


@dataclass(slots=True)
class PipelineEvent:
    event_type: Literal[
        "chunk_finalized",
        "partial",
        "speaker",
        "timestamp",
        "final",
        "error",
    ]
    chunk_id: int
    start: float
    end: float
    speaker_id: str = "UNKNOWN"
    text: str = ""
    confidence: float = 0.0
    message: str = ""
    words: list[dict[str, object]] | None = None
    speaker_segments: list[dict[str, object]] | None = None
    timestamp_status: str = ""

    def format_turn(self) -> str:
        return f"[{self.start:.2f} - {self.end:.2f}] {self.speaker_id}: {self.text}"
