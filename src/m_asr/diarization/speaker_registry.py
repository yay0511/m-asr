from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from m_asr.config import SpeakerConfig
from m_asr.types import AudioChunk, SpeakerResult


@dataclass(slots=True)
class SpeakerProfile:
    speaker_id: str
    centroid: np.ndarray
    num_embeddings: int
    last_seen_time: float


class SpeakerRegistry:
    def __init__(self, config: SpeakerConfig):
        self.config = config
        self._profiles: list[SpeakerProfile] = []
        self._last_speaker_id: str | None = None

    @property
    def profiles(self) -> tuple[SpeakerProfile, ...]:
        return tuple(self._profiles)

    def match(
        self,
        chunk: AudioChunk,
        embedding: np.ndarray | None,
        confidence: float = 1.0,
    ) -> SpeakerResult:
        if embedding is None:
            return SpeakerResult(chunk.chunk_id, "UNKNOWN", 0.0, None)

        vector = l2_normalize(np.asarray(embedding, dtype=np.float32).reshape(-1))
        if vector.size == 0 or not np.all(np.isfinite(vector)):
            return SpeakerResult(chunk.chunk_id, "UNKNOWN", 0.0, None)

        if not self._profiles:
            if chunk.duration < self._dynamic_min_new_speaker_duration(chunk):
                return SpeakerResult(chunk.chunk_id, "UNKNOWN", 0.0, vector)
            profile = self._create_profile(vector, chunk.end)
            self._last_speaker_id = profile.speaker_id
            return SpeakerResult(chunk.chunk_id, profile.speaker_id, 1.0, vector)

        similarities = np.asarray([float(np.dot(vector, p.centroid)) for p in self._profiles])
        best_index = int(np.argmax(similarities))
        best_score = float(similarities[best_index])

        last_index = self._last_profile_index()
        if last_index is not None:
            last_score = float(similarities[last_index])
            if (
                last_score >= self.config.last_speaker_threshold
                and best_score - last_score <= self.config.last_speaker_margin
            ):
                profile = self._profiles[last_index]
                if self._should_update(chunk, confidence, last_score):
                    self._update_profile(profile, vector, chunk.end)
                self._last_speaker_id = profile.speaker_id
                return SpeakerResult(chunk.chunk_id, profile.speaker_id, last_score, vector)

        if best_score >= self.config.same_speaker_threshold:
            profile = self._profiles[best_index]
            if self._should_update(chunk, confidence, best_score):
                self._update_profile(profile, vector, chunk.end)
            self._last_speaker_id = profile.speaker_id
            return SpeakerResult(chunk.chunk_id, profile.speaker_id, best_score, vector)

        if (
            best_score > self._dynamic_new_speaker_max_similarity(chunk)
            or chunk.duration < self._dynamic_min_new_speaker_duration(chunk)
        ):
            if self.config.assign_uncertain_to_best:
                profile = self._profiles[best_index]
                self._last_speaker_id = profile.speaker_id
                return SpeakerResult(chunk.chunk_id, profile.speaker_id, best_score, vector)
            return SpeakerResult(chunk.chunk_id, "UNKNOWN", best_score, vector)

        profile = self._create_profile(vector, chunk.end)
        self._last_speaker_id = profile.speaker_id
        return SpeakerResult(chunk.chunk_id, profile.speaker_id, best_score, vector)

    def _create_profile(self, vector: np.ndarray, last_seen_time: float) -> SpeakerProfile:
        speaker_id = f"SPEAKER_{len(self._profiles) + 1:02d}"
        profile = SpeakerProfile(
            speaker_id=speaker_id,
            centroid=vector,
            num_embeddings=1,
            last_seen_time=last_seen_time,
        )
        self._profiles.append(profile)
        return profile

    def _update_profile(self, profile: SpeakerProfile, vector: np.ndarray, last_seen_time: float) -> None:
        profile.centroid = l2_normalize(
            self.config.centroid_update_alpha * profile.centroid
            + (1.0 - self.config.centroid_update_alpha) * vector
        )
        profile.num_embeddings += 1
        profile.last_seen_time = last_seen_time

    def _should_update(self, chunk: AudioChunk, confidence: float, similarity: float) -> bool:
        return (
            chunk.duration >= self.config.min_embedding_duration
            and confidence >= self.config.min_update_confidence
            and similarity >= self.config.min_centroid_update_similarity
        )

    def _last_profile_index(self) -> int | None:
        if self._last_speaker_id is None:
            return None
        for index, profile in enumerate(self._profiles):
            if profile.speaker_id == self._last_speaker_id:
                return index
        return None

    def _dynamic_new_speaker_max_similarity(self, chunk: AudioChunk) -> float:
        progress = self._warmup_progress(chunk)
        return _lerp(
            self.config.new_speaker_initial_max_similarity,
            self.config.new_speaker_final_max_similarity,
            progress,
        )

    def _dynamic_min_new_speaker_duration(self, chunk: AudioChunk) -> float:
        progress = self._warmup_progress(chunk)
        return _lerp(
            self.config.min_new_speaker_duration_initial,
            self.config.min_new_speaker_duration_final,
            progress,
        )

    def _warmup_progress(self, chunk: AudioChunk) -> float:
        warmup = max(1e-6, float(self.config.new_speaker_warmup_seconds))
        return max(0.0, min(1.0, chunk.end / warmup))


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return vector.astype(np.float32, copy=False)
    return (vector / norm).astype(np.float32, copy=False)


def _lerp(start: float, end: float, progress: float) -> float:
    return start * (1.0 - progress) + end * progress
