from __future__ import annotations

from collections import deque
from enum import Enum

import numpy as np

from .config import ChunkerConfig
from .types import AudioChunk
from .vad import SileroVadStream


class ChunkerState(str, Enum):
    IDLE = "IDLE"
    TRIGGERED = "TRIGGERED"
    WAIT_SILENCE = "WAIT_SILENCE"


class _AudioPartsBuffer:
    def __init__(self, capacity_samples: int):
        self.capacity_samples = capacity_samples
        self.parts: deque[np.ndarray] = deque()
        self.num_samples = 0
        self.absolute_start_sample = 0

    @property
    def absolute_end_sample(self) -> int:
        return self.absolute_start_sample + self.num_samples

    def clear(self) -> None:
        self.parts.clear()
        self.num_samples = 0
        self.absolute_start_sample = 0

    def append(self, samples: np.ndarray) -> None:
        data = np.asarray(samples, dtype=np.float32).reshape(-1)
        if data.size == 0:
            return
        self.parts.append(data.copy())
        self.num_samples += data.size
        self._trim()

    def get_range(self, start_sample: int, end_sample: int) -> np.ndarray:
        start = max(start_sample, self.absolute_start_sample)
        end = min(end_sample, self.absolute_end_sample)
        if end <= start:
            return np.zeros(0, dtype=np.float32)

        data = np.concatenate(list(self.parts)) if self.parts else np.zeros(0, dtype=np.float32)
        offset0 = start - self.absolute_start_sample
        offset1 = end - self.absolute_start_sample
        return data[offset0:offset1].astype(np.float32, copy=False)

    def _trim(self) -> None:
        while self.num_samples > self.capacity_samples and self.parts:
            extra = self.num_samples - self.capacity_samples
            left = self.parts[0]
            if left.size <= extra:
                self.parts.popleft()
                self.num_samples -= left.size
                self.absolute_start_sample += left.size
            else:
                self.parts[0] = left[extra:]
                self.num_samples -= extra
                self.absolute_start_sample += extra


class SpeechChunker:
    """Streaming speech chunker backed by Silero VAD or an energy fallback.

    The public contract is intentionally unchanged: callers feed arbitrary
    waveform fragments to `accept`, receive finalized `AudioChunk` objects, and
    call `flush` when the input stream ends.
    """

    def __init__(self, config: ChunkerConfig, sample_rate: int = 16000):
        self.config = config
        self.sample_rate = sample_rate
        self.provider = str(config.vad_provider).lower()
        self.frame_samples = int(round(sample_rate * config.frame_ms / 1000.0))
        self.window_samples = self._resolve_window_samples()
        self.left_padding_samples = int(round(sample_rate * config.left_padding_ms / 1000.0))
        self.right_padding_samples = int(round(sample_rate * config.right_padding_ms / 1000.0))
        self.end_silence_samples = int(round(sample_rate * config.end_silence_ms / 1000.0))
        self.min_chunk_samples = int(round(sample_rate * config.min_chunk_duration))
        self.max_chunk_samples = int(round(sample_rate * config.max_chunk_duration))
        self._left_padding_frames = max(1, int(np.ceil(self.left_padding_samples / self.window_samples)))
        self._vad = self._build_vad()
        capacity = (
            self.max_chunk_samples
            + self.left_padding_samples
            + self.right_padding_samples
            + max(sample_rate * 5, self.end_silence_samples * 2)
        )
        self._buffer = _AudioPartsBuffer(int(capacity))
        self.reset()

    def reset(self) -> None:
        self.state = ChunkerState.IDLE
        self.chunk_id = 0
        self.sample_cursor = 0
        self._carry = np.zeros(0, dtype=np.float32)
        self._preroll: deque[np.ndarray] = deque(maxlen=self._left_padding_frames)
        self._current: list[np.ndarray] = []
        self._chunk_start_sample = 0
        self._speech_start_sample = 0
        self._speech_end_sample = 0
        self._silence_samples = 0
        self._buffer.clear()
        if self._vad is not None:
            self._vad.reset(sample_offset=0)

    def accept(self, waveform: np.ndarray) -> list[AudioChunk]:
        if self.provider == "energy":
            return self._accept_energy(waveform)
        if self.provider != "silero":
            raise ValueError(f"unsupported VAD provider: {self.provider!r}")
        return self._accept_silero(waveform)

    def flush(self) -> list[AudioChunk]:
        if self.provider == "energy":
            return self._flush_energy()

        chunks: list[AudioChunk] = []
        if self._carry.size:
            self._buffer.append(self._carry)
            self.sample_cursor += self._carry.size
            self._carry = np.zeros(0, dtype=np.float32)

        if self.state is ChunkerState.TRIGGERED:
            chunk = self._finalize_silero(self.sample_cursor)
            if chunk is not None:
                chunks.append(chunk)

        self.state = ChunkerState.IDLE
        if self._vad is not None:
            self._vad.reset(sample_offset=self.sample_cursor)
        return chunks

    def _accept_silero(self, waveform: np.ndarray) -> list[AudioChunk]:
        data = np.asarray(waveform, dtype=np.float32).reshape(-1)
        if self._carry.size:
            data = np.concatenate([self._carry, data])

        chunks: list[AudioChunk] = []
        usable = (data.size // self.window_samples) * self.window_samples
        for offset in range(0, usable, self.window_samples):
            frame = data[offset : offset + self.window_samples]
            frame_start = self.sample_cursor
            frame_end = frame_start + frame.size
            self._buffer.append(frame)

            assert self._vad is not None
            for event in self._vad.accept(frame):
                event_type = event["type"]
                sample = int(event["sample"])
                if event_type == "start":
                    self._on_silero_start(sample)
                elif event_type == "end":
                    chunk = self._on_silero_end(sample)
                    if chunk is not None:
                        chunks.append(chunk)

            if (
                self.state is ChunkerState.TRIGGERED
                and frame_end - self._chunk_start_sample >= self.max_chunk_samples
            ):
                chunk = self._finalize_silero(frame_end)
                if chunk is not None:
                    chunks.append(chunk)
                self._vad.reset(sample_offset=frame_end)

            self.sample_cursor = frame_end

        self._carry = data[usable:].copy()
        return chunks

    def _on_silero_start(self, start_sample: int) -> None:
        if self.state is ChunkerState.TRIGGERED:
            return
        self._chunk_start_sample = max(0, start_sample - self.left_padding_samples)
        self._speech_start_sample = start_sample
        self._speech_end_sample = start_sample
        self.state = ChunkerState.TRIGGERED

    def _on_silero_end(self, end_sample: int) -> AudioChunk | None:
        if self.state is not ChunkerState.TRIGGERED:
            return None
        self._speech_end_sample = end_sample
        return self._finalize_silero(end_sample + self.right_padding_samples)

    def _finalize_silero(self, end_sample: int) -> AudioChunk | None:
        start_sample = self._chunk_start_sample
        end = min(max(end_sample, start_sample), self._buffer.absolute_end_sample)
        if self._speech_end_sample <= self._speech_start_sample:
            self._speech_end_sample = end
        waveform = self._buffer.get_range(start_sample, end)
        self.state = ChunkerState.IDLE
        self._chunk_start_sample = 0

        if waveform.size < self.min_chunk_samples:
            return None

        chunk = AudioChunk(
            chunk_id=self.chunk_id,
            start=start_sample / self.sample_rate,
            end=(start_sample + waveform.size) / self.sample_rate,
            waveform=waveform,
            sample_rate=self.sample_rate,
            is_final=True,
            core_start=self._speech_start_sample / self.sample_rate,
            core_end=(self._speech_end_sample or start_sample + waveform.size) / self.sample_rate,
        )
        self._speech_start_sample = 0
        self._speech_end_sample = 0
        self.chunk_id += 1
        return chunk

    def _accept_energy(self, waveform: np.ndarray) -> list[AudioChunk]:
        data = np.asarray(waveform, dtype=np.float32).reshape(-1)
        if self._carry.size:
            data = np.concatenate([self._carry, data])

        chunks: list[AudioChunk] = []
        usable = (data.size // self.frame_samples) * self.frame_samples
        for offset in range(0, usable, self.frame_samples):
            frame = data[offset : offset + self.frame_samples]
            chunks.extend(self._accept_energy_frame(frame))

        self._carry = data[usable:].copy()
        return chunks

    def _flush_energy(self) -> list[AudioChunk]:
        chunks: list[AudioChunk] = []
        if self._carry.size:
            padded = np.zeros(self.frame_samples, dtype=np.float32)
            padded[: self._carry.size] = self._carry
            chunks.extend(self._accept_energy_frame(padded))
            self._carry = np.zeros(0, dtype=np.float32)
        if self.state is not ChunkerState.IDLE and self._current:
            chunk = self._finalize_energy(extra_silence_samples=0)
            if chunk is not None:
                chunks.append(chunk)
        self.state = ChunkerState.IDLE
        return chunks

    def _accept_energy_frame(self, frame: np.ndarray) -> list[AudioChunk]:
        score = self._energy_score(frame)
        frame_start = self.sample_cursor
        self.sample_cursor += frame.size

        emitted: list[AudioChunk] = []
        if self.state is ChunkerState.IDLE:
            self._preroll.append(frame.copy())
            if score >= self.config.speech_onset_threshold:
                self._start_energy_chunk(frame_start, frame)
            return emitted

        self._current.append(frame.copy())
        current_duration = self.sample_cursor - self._chunk_start_sample

        if score >= self.config.speech_offset_threshold:
            self._silence_samples = 0
            self.state = ChunkerState.TRIGGERED
        else:
            self._silence_samples += frame.size
            self.state = ChunkerState.WAIT_SILENCE

        if current_duration >= self.max_chunk_samples:
            chunk = self._finalize_energy(extra_silence_samples=0)
            if chunk is not None:
                emitted.append(chunk)
            self.state = ChunkerState.IDLE
            self._preroll.clear()
            return emitted

        if self._silence_samples >= self.end_silence_samples:
            extra = max(0, self._silence_samples - self.right_padding_samples)
            chunk = self._finalize_energy(extra_silence_samples=extra)
            if chunk is not None:
                emitted.append(chunk)
            self.state = ChunkerState.IDLE
            self._preroll.clear()

        return emitted

    def _start_energy_chunk(self, frame_start: int, frame: np.ndarray) -> None:
        preroll = list(self._preroll)
        preroll_samples = sum(part.size for part in preroll)
        self._chunk_start_sample = max(0, frame_start - preroll_samples)
        self._current = [part.copy() for part in preroll]
        if not self._current or not np.array_equal(self._current[-1], frame):
            self._current.append(frame.copy())
        self._silence_samples = 0
        self.state = ChunkerState.TRIGGERED

    def _finalize_energy(self, extra_silence_samples: int) -> AudioChunk | None:
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
            core_start=start_sample / self.sample_rate,
            core_end=end_sample / self.sample_rate,
        )
        self.chunk_id += 1
        return chunk

    def _energy_score(self, frame: np.ndarray) -> float:
        rms = float(np.sqrt(np.mean(np.square(frame), dtype=np.float64)) + 1e-12)
        return max(0.0, min(1.0, rms / max(self.config.energy_reference, 1e-6)))

    def _resolve_window_samples(self) -> int:
        if self.provider == "silero":
            window = int(self.config.silero_window_samples)
            if window <= 0:
                raise ValueError("silero_window_samples must be positive")
            return window
        return self.frame_samples

    def _build_vad(self) -> SileroVadStream | None:
        if self.provider != "silero":
            return None
        return SileroVadStream(
            sample_rate=self.sample_rate,
            threshold=self.config.silero_threshold,
            min_silence_duration_ms=self.config.silero_min_silence_ms,
            speech_pad_ms=self.config.silero_speech_pad_ms,
        )
