from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import sys
from typing import Iterable, Iterator
import warnings

import numpy as np

from .asr import XAsrClient
from .audio_buffer import AudioBuffer
from .chunker import SpeechChunker
from .config import AppConfig
from .diarization import PyannoteSpeakerEmbedder, SpeakerRegistry
from .types import AudioChunk, AsrResult, PipelineEvent, SpeakerResult, TranscriptTurn


@dataclass(slots=True)
class PipelineBackends:
    asr: str
    speaker: str


class StreamingCascadePipeline:
    def __init__(self, config: AppConfig):
        self.config = config
        resolve_runtime(config)
        self.audio_buffer = AudioBuffer(sample_rate=config.runtime.sample_rate)
        self.chunker = SpeechChunker(config.chunker, sample_rate=config.runtime.sample_rate)
        self.asr = XAsrClient(config)
        self.embedder = PyannoteSpeakerEmbedder(config)
        self.registry = SpeakerRegistry(config.speaker)
        self.transcript: list[TranscriptTurn] = []

    @property
    def backends(self) -> PipelineBackends:
        return PipelineBackends(asr=self.asr.backend, speaker=self.embedder.backend)

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
        speaker_future = executor.submit(self._identify_speaker, chunk)

        asr_result: AsrResult
        speaker_result: SpeakerResult
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

        try:
            speaker_result = speaker_future.result()
            yield PipelineEvent(
                "speaker",
                chunk.chunk_id,
                chunk.start,
                chunk.end,
                speaker_id=speaker_result.speaker_id,
                confidence=speaker_result.confidence,
            )
        except Exception as exc:
            yield PipelineEvent(
                "error",
                chunk.chunk_id,
                chunk.start,
                chunk.end,
                message=f"speaker embedding failed: {type(exc).__name__}: {exc}",
            )
            speaker_result = SpeakerResult(chunk.chunk_id, "UNKNOWN", 0.0, None)

        if not asr_result.text:
            return

        turn = TranscriptTurn(
            start=chunk.start,
            end=chunk.end,
            speaker_id=speaker_result.speaker_id,
            text=asr_result.text,
            confidence=speaker_result.confidence,
        )
        self.transcript.append(turn)
        yield PipelineEvent(
            "final",
            chunk.chunk_id,
            chunk.start,
            chunk.end,
            speaker_id=turn.speaker_id,
            text=turn.text,
            confidence=turn.confidence,
        )

    def _identify_speaker(self, chunk: AudioChunk) -> SpeakerResult:
        embedding = self.embedder.extract(chunk)
        return self.registry.match(chunk, embedding, confidence=1.0)


def _iter_array_frames(waveform: np.ndarray, frame_samples: int) -> Iterator[np.ndarray]:
    data = np.asarray(waveform)
    for offset in range(0, data.shape[0], frame_samples):
        yield data[offset : offset + frame_samples]


def resolve_runtime(config: AppConfig) -> None:
    wants_cuda = config.runtime.device == "cuda" or config.runtime.asr_provider == "cuda"
    if not wants_cuda:
        return

    try:
        import torch
    except Exception as exc:
        _fallback_to_cpu(config, f"torch cannot be imported: {type(exc).__name__}: {exc}")
        return

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="CUDA initialization:.*", category=UserWarning)
        cuda_available = torch.cuda.is_available()

    if cuda_available:
        return

    cuda_build = getattr(torch.version, "cuda", None)
    _fallback_to_cpu(
        config,
        "torch.cuda.is_available() is False. "
        f"Installed torch={torch.__version__}, torch CUDA build={cuda_build}. "
        "Run `uv run --offline --extra asr python scripts/check_cuda.py` for details.",
    )


def _fallback_to_cpu(config: AppConfig, reason: str) -> None:
    print(
        f"[warn  ] CUDA requested but unavailable; falling back to CPU. {reason}",
        file=sys.stderr,
    )
    config.runtime.device = "cpu"
    config.runtime.asr_provider = "cpu"
