from __future__ import annotations

from collections import deque
from enum import Enum

import numpy as np

from .config import ChunkerConfig
from .types import AudioChunk


class ChunkerState(str, Enum):
    IDLE = "IDLE"
    IN_SPEECH = "IN_SPEECH"
    WAIT_SILENCE = "WAIT_SILENCE"


class SpeechChunker:
    """Energy/VAD-score based streaming utterance chunker.

    The first version follows the state machine from the design document and
    uses normalized frame RMS as a dependency-free VAD score. A neural VAD can
    replace `_speech_score` later without changing the chunk contract.
    """

    def __init__(self, config: ChunkerConfig, sample_rate: int = 16000):
        self.config = config
        self.sample_rate = sample_rate
        self.frame_samples = int(round(sample_rate * config.frame_ms / 1000.0))
        self.left_padding_samples = int(round(sample_rate * config.left_padding_ms / 1000.0))
        self.right_padding_samples = int(round(sample_rate * config.right_padding_ms / 1000.0))
        self.end_silence_samples = int(round(sample_rate * config.end_silence_ms / 1000.0))
        self.min_chunk_samples = int(round(sample_rate * config.min_chunk_duration))
        self.max_chunk_samples = int(round(sample_rate * config.max_chunk_duration))
        self._left_padding_frames = max(1, int(np.ceil(self.left_padding_samples / self.frame_samples)))
        self.reset()

    def reset(self) -> None:
        self.state = ChunkerState.IDLE
        self.chunk_id = 0
        self.sample_cursor = 0
        self._carry = np.zeros(0, dtype=np.float32)
        self._preroll: deque[np.ndarray] = deque(maxlen=self._left_padding_frames)
        self._current: list[np.ndarray] = []
        self._chunk_start_sample = 0
        self._silence_samples = 0

    def accept(self, waveform: np.ndarray) -> list[AudioChunk]:
        data = np.asarray(waveform, dtype=np.float32).reshape(-1)
        if self._carry.size:
            data = np.concatenate([self._carry, data])

        chunks: list[AudioChunk] = []
        usable = (data.size // self.frame_samples) * self.frame_samples
        for offset in range(0, usable, self.frame_samples):
            frame = data[offset : offset + self.frame_samples]
            chunks.extend(self._accept_frame(frame))

        self._carry = data[usable:].copy()
        return chunks

    def flush(self) -> list[AudioChunk]:
        chunks: list[AudioChunk] = []
        if self._carry.size:
            padded = np.zeros(self.frame_samples, dtype=np.float32)
            padded[: self._carry.size] = self._carry
            chunks.extend(self._accept_frame(padded))
            self._carry = np.zeros(0, dtype=np.float32)
        if self.state is not ChunkerState.IDLE and self._current:
            chunk = self._finalize(extra_silence_samples=0)
            if chunk is not None:
                chunks.append(chunk)
        self.state = ChunkerState.IDLE
        return chunks

    def _accept_frame(self, frame: np.ndarray) -> list[AudioChunk]:
        score = self._speech_score(frame)
        frame_start = self.sample_cursor
        self.sample_cursor += frame.size

        emitted: list[AudioChunk] = []
        if self.state is ChunkerState.IDLE:
            self._preroll.append(frame.copy())
            if score >= self.config.speech_onset_threshold:
                self._start_chunk(frame_start, frame)
            return emitted

        self._current.append(frame.copy())
        current_duration = self.sample_cursor - self._chunk_start_sample

        if score >= self.config.speech_offset_threshold:
            self._silence_samples = 0
            self.state = ChunkerState.IN_SPEECH
        else:
            self._silence_samples += frame.size
            self.state = ChunkerState.WAIT_SILENCE

        if current_duration >= self.max_chunk_samples:
            chunk = self._finalize(extra_silence_samples=0)
            if chunk is not None:
                emitted.append(chunk)
            self.state = ChunkerState.IDLE
            self._preroll.clear()
            return emitted

        if self._silence_samples >= self.end_silence_samples:
            extra = max(0, self._silence_samples - self.right_padding_samples)
            chunk = self._finalize(extra_silence_samples=extra)
            if chunk is not None:
                emitted.append(chunk)
            self.state = ChunkerState.IDLE
            self._preroll.clear()

        return emitted

    def _start_chunk(self, frame_start: int, frame: np.ndarray) -> None:
        preroll = list(self._preroll)
        preroll_samples = sum(part.size for part in preroll)
        self._chunk_start_sample = max(0, frame_start - preroll_samples)
        self._current = [part.copy() for part in preroll]
        if not self._current or not np.array_equal(self._current[-1], frame):
            self._current.append(frame.copy())
        self._silence_samples = 0
        self.state = ChunkerState.IN_SPEECH

    def _finalize(self, extra_silence_samples: int) -> AudioChunk | None:
        waveform = np.concatenate(self._current).astype(np.float32, copy=False)
        if extra_silence_samples > 0:
            keep = max(0, waveform.size - extra_silence_samples)
            waveform = waveform[:keep]

        start_sample = self._chunk_start_sample
        end_sample = start_sample + waveform.size
        self._current = []
        self._silence_samples = 0

        if waveform.size < self.min_chunk_samples:
            return None

        chunk = AudioChunk(
            chunk_id=self.chunk_id,
            start=start_sample / self.sample_rate,
            end=end_sample / self.sample_rate,
            waveform=waveform,
            sample_rate=self.sample_rate,
            is_final=True,
        )
        self.chunk_id += 1
        return chunk

    def _speech_score(self, frame: np.ndarray) -> float:
        rms = float(np.sqrt(np.mean(np.square(frame), dtype=np.float64)) + 1e-12)
        return max(0.0, min(1.0, rms / max(self.config.energy_reference, 1e-6)))
