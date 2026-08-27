from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


def to_mono_float32(waveform: np.ndarray) -> np.ndarray:
    data = np.asarray(waveform)
    if data.ndim == 2:
        data = data.mean(axis=1)
    data = data.astype(np.float32, copy=False).reshape(-1)
    if data.size == 0:
        return data
    peak = float(np.max(np.abs(data)))
    if peak > 1.5:
        data = data / 32768.0
    return np.clip(data, -1.0, 1.0).astype(np.float32, copy=False)


def linear_resample(waveform: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    data = to_mono_float32(waveform)
    if source_rate == target_rate or data.size == 0:
        return data
    target_len = int(round(data.size * target_rate / source_rate))
    if target_len <= 0:
        return np.zeros(0, dtype=np.float32)
    src_x = np.linspace(0.0, 1.0, num=data.size, endpoint=False)
    dst_x = np.linspace(0.0, 1.0, num=target_len, endpoint=False)
    return np.interp(dst_x, src_x, data).astype(np.float32)


@dataclass(slots=True)
class AudioWrite:
    start_sample: int
    end_sample: int
    start_time: float
    end_time: float
    waveform: np.ndarray


class RingBuffer:
    def __init__(self, capacity_samples: int):
        if capacity_samples <= 0:
            raise ValueError("capacity_samples must be positive")
        self.capacity_samples = capacity_samples
        self._parts: deque[np.ndarray] = deque()
        self._num_samples = 0
        self.absolute_start_sample = 0

    @property
    def num_samples(self) -> int:
        return self._num_samples

    @property
    def absolute_end_sample(self) -> int:
        return self.absolute_start_sample + self._num_samples

    def append(self, samples: np.ndarray) -> None:
        data = to_mono_float32(samples)
        if data.size == 0:
            return
        self._parts.append(data)
        self._num_samples += data.size
        self._trim()

    def get_range(self, start_sample: int, end_sample: int) -> np.ndarray:
        if start_sample < self.absolute_start_sample or end_sample > self.absolute_end_sample:
            raise ValueError("requested range is outside ring buffer")
        if end_sample <= start_sample:
            return np.zeros(0, dtype=np.float32)
        data = np.concatenate(list(self._parts)) if self._parts else np.zeros(0, dtype=np.float32)
        offset0 = start_sample - self.absolute_start_sample
        offset1 = end_sample - self.absolute_start_sample
        return data[offset0:offset1].astype(np.float32, copy=False)

    def _trim(self) -> None:
        while self._num_samples > self.capacity_samples and self._parts:
            extra = self._num_samples - self.capacity_samples
            left = self._parts[0]
            if left.size <= extra:
                self._parts.popleft()
                self._num_samples -= left.size
                self.absolute_start_sample += left.size
            else:
                self._parts[0] = left[extra:]
                self._num_samples -= extra
                self.absolute_start_sample += extra


class AudioBuffer:
    def __init__(self, sample_rate: int = 16000, capacity_seconds: float = 60.0):
        self.sample_rate = sample_rate
        self.ring = RingBuffer(int(round(sample_rate * capacity_seconds)))
        self.total_samples = 0

    def append(self, waveform: np.ndarray, source_sample_rate: int | None = None) -> AudioWrite:
        source_rate = source_sample_rate or self.sample_rate
        samples = linear_resample(waveform, source_rate, self.sample_rate)
        start = self.total_samples
        end = start + samples.size
        self.ring.append(samples)
        self.total_samples = end
        return AudioWrite(
            start_sample=start,
            end_sample=end,
            start_time=start / self.sample_rate,
            end_time=end / self.sample_rate,
            waveform=samples,
        )

    @property
    def start_time(self) -> float:
        return self.ring.absolute_start_sample / self.sample_rate

    @property
    def end_time(self) -> float:
        return self.ring.absolute_end_sample / self.sample_rate

    def get_available_range_seconds(self, start: float, end: float) -> tuple[float, float, np.ndarray]:
        actual_start = max(float(start), self.start_time)
        actual_end = min(float(end), self.end_time)
        if actual_end <= actual_start:
            return actual_start, actual_start, np.zeros(0, dtype=np.float32)
        return actual_start, actual_end, self.get_range_seconds(actual_start, actual_end)

    def get_range_seconds(self, start: float, end: float) -> np.ndarray:
        return self.ring.get_range(int(round(start * self.sample_rate)), int(round(end * self.sample_rate)))
