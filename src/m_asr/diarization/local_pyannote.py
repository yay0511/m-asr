from __future__ import annotations

from dataclasses import dataclass
import time
from pathlib import Path
from typing import Iterable

import numpy as np

from m_asr.config import AppConfig
from m_asr.types import AudioChunk, LocalDiarizationResult, LocalSpeakerSegment, LocalSpeakerTrack


@dataclass(slots=True)
class AnalysisAudio:
    waveform: np.ndarray
    start: float
    end: float
    core_start: float
    core_end: float
    sample_rate: int


class ChunkLocalPyannoteDiarizer:
    """Extract local speakers from one chunk using pyannote building blocks.

    This deliberately reuses pyannote's Model/Inference and masked speaker
    embedding implementations, but does not run pyannote's global clustering.
    Local channel identities are valid only inside the returned chunk result.
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self.local_config = config.local_pyannote
        self.sample_rate = config.runtime.sample_rate
        self._segmentation = None
        self._embedding = None
        self._segmentation_duration = 10.0
        self._backend = "real"
        if not self.local_config.enabled:
            self._backend = "disabled"
            return
        self._init_real()

    @property
    def backend(self) -> str:
        return self._backend

    def _init_real(self) -> None:
        import torch
        from pyannote.audio import Inference, Model
        from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding

        model_root = Path(self.config.paths.pyannote_model_dir)
        segmentation_dir = model_root / "segmentation"
        embedding_dir = model_root / "embedding"
        if not (segmentation_dir / "pytorch_model.bin").exists():
            raise FileNotFoundError(f"pyannote segmentation model not found: {segmentation_dir}")
        if not embedding_dir.exists():
            raise FileNotFoundError(f"pyannote embedding model not found: {embedding_dir}")

        device_name = str(self.config.runtime.device).lower()
        device = torch.device(device_name if device_name == "cuda" and torch.cuda.is_available() else "cpu")
        segmentation_model = Model.from_pretrained(str(segmentation_dir), map_location=device)
        self._segmentation_duration = float(segmentation_model.specifications.duration)
        self._segmentation = Inference(
            segmentation_model,
            window="sliding",
            duration=self._segmentation_duration,
            step=self._segmentation_duration,
            skip_aggregation=True,
            device=device,
            batch_size=1,
        )
        self._embedding = PretrainedSpeakerEmbedding(str(embedding_dir), device=device)

    def analyze(
        self,
        chunk: AudioChunk,
        analysis_audio: AnalysisAudio | None = None,
    ) -> LocalDiarizationResult:
        started = time.perf_counter()
        if self._segmentation is None or self._embedding is None:
            return LocalDiarizationResult(chunk.chunk_id, chunk.start, chunk.end, [], 0.0)

        audio = analysis_audio or AnalysisAudio(
            waveform=np.asarray(chunk.waveform, dtype=np.float32),
            start=chunk.start,
            end=chunk.end,
            core_start=chunk.speech_start,
            core_end=chunk.speech_end,
            sample_rate=chunk.sample_rate,
        )
        padded_waveform = self._pad_to_segmentation_duration(audio.waveform)
        file = {
            "waveform": self._waveform_tensor(padded_waveform),
            "sample_rate": audio.sample_rate,
        }
        segmentations = self._segmentation(file)
        tracks = self._tracks_from_segmentations(segmentations, audio)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return LocalDiarizationResult(
            chunk_id=chunk.chunk_id,
            start=chunk.speech_start,
            end=chunk.speech_end,
            tracks=tracks,
            latency_ms=elapsed_ms,
        )

    def _tracks_from_segmentations(self, segmentations: object, audio: AnalysisAudio) -> list[LocalSpeakerTrack]:
        data = np.asarray(segmentations.data, dtype=np.float32)
        if data.ndim == 2:
            data = data[None, ...]
        if data.ndim != 3 or data.shape[-1] == 0:
            return []

        local_tracks: list[LocalSpeakerTrack] = []
        for local_index in range(min(data.shape[-1], self.local_config.max_local_speakers)):
            mask = np.max(data[:, :, local_index], axis=0) if data.shape[0] else np.zeros(0)
            mask = np.nan_to_num(mask, nan=0.0)
            frame_duration = self._segmentation_duration / max(1, mask.size)
            raw_regions = _mask_to_regions(
                mask,
                frame_duration=frame_duration,
                threshold=self.local_config.segmentation_threshold,
                offset=self.local_config.segmentation_offset,
                min_duration=self.local_config.min_local_speaker_duration,
            )
            regions = []
            for start, end, confidence in raw_regions:
                absolute_start = max(audio.core_start, audio.start + start)
                absolute_end = min(audio.core_end, audio.start + end)
                if absolute_end > absolute_start:
                    regions.append(
                        LocalSpeakerSegment(
                            local_id=f"local_{local_index}",
                            start=absolute_start,
                            end=absolute_end,
                            confidence=confidence,
                        )
                    )
            if not regions:
                continue

            speech_duration = sum(max(0.0, r.end - r.start) for r in regions)
            if speech_duration < self.local_config.min_local_speaker_duration:
                continue
            embedding_mask = _clip_mask_to_core(
                mask,
                frame_duration=frame_duration,
                audio=audio,
            )
            if self.local_config.exclude_overlap:
                all_activity = np.max(data, axis=0)
                overlap_frames = np.sum(
                    all_activity >= self.local_config.segmentation_threshold,
                    axis=1,
                ) >= 2
                clean_mask = mask.copy()
                clean_mask[overlap_frames] = 0.0
                if np.count_nonzero(clean_mask >= self.local_config.segmentation_offset) >= np.count_nonzero(mask >= self.local_config.segmentation_offset) * 0.25:
                    embedding_mask = clean_mask
            embedding = self._extract_masked_embedding(
                padded_mask=_expand_mask(embedding_mask, data.shape[0]),
                waveform=self._pad_to_segmentation_duration(audio.waveform),
            )
            if embedding is None:
                continue
            overlap = _has_overlap(regions, self._all_regions(data, audio))
            for region in regions:
                region.overlap = overlap
            local_tracks.append(
                LocalSpeakerTrack(
                    local_id=f"local_{local_index}",
                    segments=regions,
                    embedding=embedding,
                    speech_duration=speech_duration,
                    confidence=float(np.mean([r.confidence for r in regions])),
                )
            )
        return local_tracks

    def _all_regions(self, data: np.ndarray, audio: AnalysisAudio) -> list[LocalSpeakerSegment]:
        result: list[LocalSpeakerSegment] = []
        for local_index in range(data.shape[-1]):
            mask = np.max(data[:, :, local_index], axis=0)
            for start, end, confidence in _mask_to_regions(
                mask,
                self._segmentation_duration / max(1, mask.size),
                self.local_config.segmentation_threshold,
                self.local_config.segmentation_offset,
                self.local_config.min_local_speaker_duration,
            ):
                absolute_start = max(audio.core_start, audio.start + start)
                absolute_end = min(audio.core_end, audio.start + end)
                if absolute_end > absolute_start:
                    result.append(
                        LocalSpeakerSegment(
                            local_id=f"local_{local_index}",
                            start=absolute_start,
                            end=absolute_end,
                            confidence=confidence,
                        )
                    )
        return result

    def _extract_masked_embedding(self, padded_mask: np.ndarray, waveform: np.ndarray) -> np.ndarray | None:
        import torch

        try:
            waveform_tensor = torch.from_numpy(waveform).reshape(1, 1, -1)
            mask_tensor = torch.from_numpy(padded_mask.astype(np.float32)).reshape(1, -1)
            embedding = self._embedding(waveform_tensor, masks=mask_tensor)
        except RuntimeError:
            return None
        if embedding is None or len(embedding) == 0:
            return None
        vector = np.asarray(embedding[0], dtype=np.float32).reshape(-1)
        if vector.size == 0 or not np.all(np.isfinite(vector)):
            return None
        return vector

    def _pad_to_segmentation_duration(self, waveform: np.ndarray) -> np.ndarray:
        data = np.asarray(waveform, dtype=np.float32).reshape(-1)
        target = int(round(self._segmentation_duration * self.sample_rate))
        if data.size >= target:
            return data[:target]
        padded = np.zeros(target, dtype=np.float32)
        padded[: data.size] = data
        return padded

    @staticmethod
    def _waveform_tensor(waveform: np.ndarray):
        import torch

        return torch.from_numpy(waveform).reshape(1, -1)


def _expand_mask(mask: np.ndarray, num_chunks: int) -> np.ndarray:
    if num_chunks <= 1:
        return mask.astype(np.float32, copy=False)
    return np.tile(mask.astype(np.float32, copy=False), num_chunks)


def _clip_mask_to_core(
    mask: np.ndarray,
    frame_duration: float,
    audio: AnalysisAudio,
) -> np.ndarray:
    """Keep context for segmentation but exclude it from speaker embedding."""
    clipped = np.zeros_like(mask, dtype=np.float32)
    start = max(0, int(np.floor((audio.core_start - audio.start) / frame_duration)))
    end = min(mask.size, int(np.ceil((audio.core_end - audio.start) / frame_duration)))
    if end > start:
        clipped[start:end] = mask[start:end]
    return clipped


def _mask_to_regions(
    mask: np.ndarray,
    frame_duration: float,
    threshold: float,
    offset: float,
    min_duration: float,
) -> list[tuple[float, float, float]]:
    regions: list[tuple[float, float, float]] = []
    start: int | None = None
    active = False
    for index, value in enumerate(np.append(np.asarray(mask, dtype=np.float32), 0.0)):
        if not active and value >= threshold:
            start = index
            active = True
        elif active and value < offset and start is not None:
            end = index
            duration = (end - start) * frame_duration
            if duration >= min_duration:
                values = np.asarray(mask[start:end], dtype=np.float32)
                regions.append((start * frame_duration, end * frame_duration, float(values.mean())))
            start = None
            active = False
    return regions


def _has_overlap(regions: Iterable[LocalSpeakerSegment], all_regions: Iterable[LocalSpeakerSegment]) -> bool:
    for region in regions:
        for other in all_regions:
            if region.local_id == other.local_id:
                continue
            if min(region.end, other.end) - max(region.start, other.start) > 0.0:
                return True
    return False
