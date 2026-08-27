from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import os
from pathlib import Path
import sys
from typing import Iterable, Iterator
import warnings

import numpy as np

from .asr import ParaformerTimestampClient, XAsrClient
from .audio_buffer import AudioBuffer
from .chunker import SpeechChunker
from .config import AppConfig
from .diarization import ChunkLocalPyannoteDiarizer, SpeakerRegistry
from .diarization.local_pyannote import AnalysisAudio
from .types import (
    AudioChunk,
    AsrResult,
    GlobalSpeakerSegment,
    LocalDiarizationResult,
    PipelineEvent,
    TimestampedAsrResult,
    TimestampedWord,
    TranscriptTurn,
)


@dataclass(slots=True)
class PipelineBackends:
    asr: str
    speaker: str


class StreamingCascadePipeline:
    def __init__(self, config: AppConfig):
        self.config = config
        resolve_runtime(config)
        self.audio_buffer = AudioBuffer(
            sample_rate=config.runtime.sample_rate,
            capacity_seconds=config.audio_buffer.retention_seconds,
        )
        self.chunker = SpeechChunker(config.chunker, sample_rate=config.runtime.sample_rate)
        self.asr = XAsrClient(config)
        self.local_diarizer = ChunkLocalPyannoteDiarizer(config)
        self.timestamp_asr = ParaformerTimestampClient(config)
        self.registry = SpeakerRegistry(config.speaker)
        self.transcript: list[TranscriptTurn] = []

    @property
    def backends(self) -> PipelineBackends:
        return PipelineBackends(asr=self.asr.backend, speaker=self.local_diarizer.backend)

    def process_frames(
        self,
        frames: Iterable[np.ndarray],
        source_sample_rate: int | None = None,
    ) -> Iterator[PipelineEvent]:
        with ThreadPoolExecutor(max_workers=self.config.runtime.max_workers) as executor:
            for frame in frames:
                write = self.audio_buffer.append(frame, source_sample_rate)
                for chunk in self.chunker.accept(write.waveform):
                    yield from self._process_chunk(chunk, executor)

            for chunk in self.chunker.flush():
                yield from self._process_chunk(chunk, executor)

    def process_waveform(
        self,
        waveform: np.ndarray,
        source_sample_rate: int | None = None,
        frame_seconds: float = 0.2,
    ) -> Iterator[PipelineEvent]:
        source_rate = source_sample_rate or self.config.runtime.sample_rate
        frame_samples = max(1, int(round(source_rate * frame_seconds)))
        for event in self.process_frames(_iter_array_frames(waveform, frame_samples), source_rate):
            yield event

    def _process_chunk(
        self,
        chunk: AudioChunk,
        executor: ThreadPoolExecutor,
    ) -> Iterator[PipelineEvent]:
        yield PipelineEvent("chunk_finalized", chunk.chunk_id, chunk.start, chunk.end)

        asr_future = executor.submit(self.asr.recognize, chunk)
        local_future = executor.submit(self.analyze_local, chunk)
        timestamp_future = executor.submit(self.analyze_timestamp, chunk)

        asr_result: AsrResult
        try:
            asr_result = asr_future.result()
        except Exception as exc:
            yield PipelineEvent(
                "error",
                chunk.chunk_id,
                chunk.start,
                chunk.end,
                message=f"ASR failed: {type(exc).__name__}: {exc}",
            )
            asr_result = AsrResult(chunk.chunk_id, "", True, None)

        if asr_result.text:
            yield PipelineEvent(
                "partial",
                chunk.chunk_id,
                chunk.start,
                chunk.end,
                speaker_id="UNKNOWN",
                text=asr_result.text,
                confidence=0.0,
            )

        speaker_segments: list[GlobalSpeakerSegment] = []
        try:
            local_result = local_future.result()
            speaker_segments, _ = self.registry.assign_local_tracks(local_result)
            speaker_id, speaker_confidence = _display_speaker(words=[], segments=speaker_segments)
            yield PipelineEvent(
                "speaker",
                chunk.chunk_id,
                chunk.start,
                chunk.end,
                speaker_id=speaker_id,
                confidence=speaker_confidence,
                speaker_segments=[_segment_to_dict(segment) for segment in speaker_segments],
            )
        except Exception as exc:
            yield PipelineEvent(
                "error",
                chunk.chunk_id,
                chunk.start,
                chunk.end,
                message=f"speaker embedding failed: {type(exc).__name__}: {exc}",
            )
            speaker_id, speaker_confidence = "UNKNOWN", 0.0

        timestamp_result = TimestampedAsrResult(chunk.chunk_id, "", [], [], "unavailable")
        try:
            timestamp_result = self._align_timestamp(asr_result.text, timestamp_future.result())
            if timestamp_result.words:
                yield PipelineEvent(
                    "timestamp",
                    chunk.chunk_id,
                    chunk.start,
                    chunk.end,
                    text=timestamp_result.text,
                    words=[_word_to_dict(word) for word in timestamp_result.words],
                    timestamp_status=timestamp_result.status,
                )
        except Exception as exc:
            yield PipelineEvent(
                "error",
                chunk.chunk_id,
                chunk.start,
                chunk.end,
                message=f"timestamp branch failed: {type(exc).__name__}: {exc}",
            )

        if not asr_result.text:
            return

        attached_words = _attach_speakers(
            timestamp_result.words,
            speaker_segments,
            min_run_words=self.config.transcript.min_speaker_run_words,
            min_run_duration=self.config.transcript.min_speaker_run_duration,
        )
        turns = _words_to_turns(chunk, asr_result.text, attached_words, speaker_id, speaker_confidence)
        self.transcript.extend(turns)
        event_speaker, event_confidence = _display_speaker(attached_words, speaker_segments)
        yield PipelineEvent(
            "final",
            chunk.chunk_id,
            chunk.start,
            chunk.end,
            speaker_id=event_speaker,
            text=asr_result.text,
            confidence=event_confidence,
            words=[_word_to_dict(word) for word in attached_words],
            speaker_segments=[_segment_to_dict(segment) for segment in speaker_segments],
        )

    def analyze_local(self, chunk: AudioChunk) -> LocalDiarizationResult:
        return self.local_diarizer.analyze(chunk, self._analysis_audio(chunk))

    def analyze_timestamp(self, chunk: AudioChunk) -> TimestampedAsrResult:
        return self.timestamp_asr.recognize(chunk)

    def _align_timestamp(self, text: str, result: TimestampedAsrResult) -> TimestampedAsrResult:
        return self.timestamp_asr.align_to_xasr(text, result)

    def _analysis_audio(self, chunk: AudioChunk) -> AnalysisAudio:
        start = max(0.0, chunk.speech_start - self.config.local_pyannote.left_context_seconds)
        end = chunk.speech_end + self.config.local_pyannote.right_context_seconds
        actual_start, actual_end, waveform = self.audio_buffer.get_available_range_seconds(start, end)
        if waveform.size == 0:
            return AnalysisAudio(chunk.waveform, chunk.start, chunk.end, chunk.speech_start, chunk.speech_end, chunk.sample_rate)
        return AnalysisAudio(
            waveform=waveform,
            start=actual_start,
            end=actual_end,
            core_start=max(actual_start, chunk.speech_start),
            core_end=min(actual_end, chunk.speech_end),
            sample_rate=chunk.sample_rate,
        )


def _iter_array_frames(waveform: np.ndarray, frame_samples: int) -> Iterator[np.ndarray]:
    data = np.asarray(waveform)
    for offset in range(0, data.shape[0], frame_samples):
        yield data[offset : offset + frame_samples]


def _display_speaker(
    words: list[TimestampedWord],
    segments: list[GlobalSpeakerSegment],
) -> tuple[str, float]:
    speaker_ids = {word.speaker_id for word in words if word.speaker_id != "UNKNOWN"}
    if not speaker_ids:
        speaker_ids = {segment.speaker_id for segment in segments if segment.speaker_id != "UNKNOWN"}
    if len(speaker_ids) == 1:
        speaker = next(iter(speaker_ids))
        confidence = max(
            [word.confidence for word in words if word.speaker_id == speaker]
            + [segment.confidence for segment in segments if segment.speaker_id == speaker]
            + [0.0]
        )
        return speaker, confidence
    if len(speaker_ids) > 1:
        return "MULTI", max([item.confidence for item in segments] + [0.0])
    return "UNKNOWN", 0.0


def _words_to_turns(
    chunk: AudioChunk,
    text: str,
    words: list[TimestampedWord],
    fallback_speaker: str,
    fallback_confidence: float,
) -> list[TranscriptTurn]:
    if not words:
        return [TranscriptTurn(chunk.start, chunk.end, fallback_speaker, text, fallback_confidence, [])]

    turns: list[TranscriptTurn] = []
    current: list[TimestampedWord] = []
    for word in words:
        if current:
            previous = current[-1]
            same_speaker = word.speaker_id == previous.speaker_id
            close_in_time = word.start - previous.end <= 1.0
            if not same_speaker or not close_in_time:
                turns.append(_turn_from_words(current))
                current = []
        current.append(word)
    if current:
        turns.append(_turn_from_words(current))
    return turns


def _turn_from_words(words: list[TimestampedWord]) -> TranscriptTurn:
    speaker = words[0].speaker_id or "UNKNOWN"
    return TranscriptTurn(
        start=words[0].start,
        end=words[-1].end,
        speaker_id=speaker,
        text=_join_words(words),
        confidence=max((word.confidence for word in words), default=0.0),
        words=list(words),
    )


def _join_words(words: list[TimestampedWord]) -> str:
    result = ""
    for word in words:
        if not result:
            result = word.text
        elif result[-1:].isascii() and word.text[:1].isascii():
            result += " " + word.text
        else:
            result += word.text
    return result


def _segment_to_dict(segment: GlobalSpeakerSegment) -> dict[str, object]:
    return {
        "start": segment.start,
        "end": segment.end,
        "speaker_id": segment.speaker_id,
        "confidence": segment.confidence,
        "overlap": segment.overlap,
    }


def _attach_speakers(
    words: list[TimestampedWord],
    segments: list[GlobalSpeakerSegment],
    min_run_words: int = 2,
    min_run_duration: float = 0.35,
) -> list[TimestampedWord]:
    attached: list[TimestampedWord] = []
    for word in words:
        overlaps: dict[str, float] = {}
        for segment in segments:
            overlap = max(0.0, min(word.end, segment.end) - max(word.start, segment.start))
            if overlap:
                overlaps[segment.speaker_id] = overlaps.get(segment.speaker_id, 0.0) + overlap
        speaker = max(overlaps, key=overlaps.get) if overlaps else "UNKNOWN"
        attached.append(
            TimestampedWord(word.text, word.start, word.end, speaker, word.confidence)
        )
    return _smooth_word_speakers(attached, min_run_words, min_run_duration)


def _smooth_word_speakers(
    words: list[TimestampedWord],
    min_run_words: int,
    min_run_duration: float,
) -> list[TimestampedWord]:
    if len(words) < 3:
        return words
    smoothed = list(words)
    runs: list[tuple[int, int, str, float]] = []
    start = 0
    for index in range(1, len(words) + 1):
        if index == len(words) or words[index].speaker_id != words[start].speaker_id:
            end = index
            duration = max(0.0, words[end - 1].end - words[start].start)
            runs.append((start, end, words[start].speaker_id, duration))
            start = index
    for run_index in range(1, len(runs) - 1):
        start, end, speaker, duration = runs[run_index]
        previous_speaker = runs[run_index - 1][2]
        next_speaker = runs[run_index + 1][2]
        if (
            previous_speaker == next_speaker
            and speaker != previous_speaker
            and (end - start) < max(1, min_run_words)
            and duration < min_run_duration
        ):
            for index in range(start, end):
                word = smoothed[index]
                smoothed[index] = TimestampedWord(
                    word.text,
                    word.start,
                    word.end,
                    previous_speaker,
                    word.confidence,
                )
    return smoothed


def _word_to_dict(word: TimestampedWord) -> dict[str, object]:
    return {
        "text": word.text,
        "start": word.start,
        "end": word.end,
        "speaker_id": word.speaker_id,
        "confidence": word.confidence,
    }


def resolve_runtime(config: AppConfig) -> None:
    config.runtime.device = str(config.runtime.device).lower()
    config.runtime.asr_provider = str(config.runtime.asr_provider).lower()

    wants_torch_cuda = config.runtime.device == "cuda"
    wants_asr_cuda = config.runtime.asr_provider in {"auto", "cuda"}

    torch_module = None
    cuda_available = False
    cuda_error: BaseException | None = None
    if wants_torch_cuda or wants_asr_cuda:
        try:
            import torch

            torch_module = torch
            cuda_available, cuda_error = _torch_cuda_available(torch)
        except Exception as exc:
            cuda_error = exc

    if wants_torch_cuda and not cuda_available:
        _fallback_torch_device_to_cpu(config, _cuda_reason(torch_module, cuda_error))

    if not wants_asr_cuda:
        return

    if config.runtime.asr_provider == "cuda" and _force_sherpa_cuda():
        if cuda_available:
            return
        _fallback_asr_provider_to_cpu(
            config,
            "M_ASR_FORCE_SHERPA_CUDA=1 is set, but CUDA is not available. "
            f"{_cuda_reason(torch_module, cuda_error)}",
        )
        return

    if cuda_available and _sherpa_onnx_has_cuda_provider():
        config.runtime.asr_provider = "cuda"
        return

    if config.runtime.asr_provider == "cuda":
        _fallback_asr_provider_to_cpu(
            config,
            "the installed sherpa-onnx package does not expose a CUDA provider. "
            "Install/compile a GPU-enabled sherpa-onnx build, or set "
            "M_ASR_FORCE_SHERPA_CUDA=1 only after that build is installed.",
        )
    config.runtime.asr_provider = "cpu"


def _torch_cuda_available(torch_module: object) -> tuple[bool, BaseException | None]:
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="CUDA initialization:.*", category=UserWarning)
            return bool(torch_module.cuda.is_available()), None
    except BaseException as exc:
        return False, exc


def _cuda_reason(torch_module: object | None, cuda_error: BaseException | None) -> str:
    if torch_module is None:
        return f"torch cannot be imported: {type(cuda_error).__name__}: {cuda_error}"
    cuda_build = getattr(torch_module.version, "cuda", None)
    reason = "torch.cuda.is_available() is False."
    if cuda_error:
        reason = f"torch CUDA check failed: {type(cuda_error).__name__}: {cuda_error}"
    return (
        f"{reason} "
        f"Installed torch={torch_module.__version__}, torch CUDA build={cuda_build}. "
        "Run `uv run --offline --extra asr python scripts/check_cuda.py` for details."
    )


def _sherpa_onnx_has_cuda_provider() -> bool:
    try:
        import sherpa_onnx
    except Exception:
        return False

    package_dir = Path(sherpa_onnx.__file__).resolve().parent
    return any(
        "onnxruntime_providers_cuda" in path.name.lower()
        for path in package_dir.rglob("*")
    )


def _force_sherpa_cuda() -> bool:
    return os.getenv("M_ASR_FORCE_SHERPA_CUDA", "").lower() in {"1", "true", "yes", "on"}


def _fallback_torch_device_to_cpu(config: AppConfig, reason: str) -> None:
    print(
        f"[warn  ] CUDA requested for pyannote but unavailable; using CPU. {reason}",
        file=sys.stderr,
    )
    config.runtime.device = "cpu"


def _fallback_asr_provider_to_cpu(config: AppConfig, reason: str) -> None:
    print(
        f"[warn  ] CUDA requested for X-ASR but unavailable; using sherpa-onnx CPU provider. {reason}",
        file=sys.stderr,
    )
    config.runtime.asr_provider = "cpu"
