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
            profile = self._create_profile(vector, chunk.end)
            return SpeakerResult(chunk.chunk_id, profile.speaker_id, 1.0, vector)

        similarities = np.asarray([float(np.dot(vector, p.centroid)) for p in self._profiles])
        best_index = int(np.argmax(similarities))
        best_score = float(similarities[best_index])

        if best_score >= self.config.same_speaker_threshold:
            profile = self._profiles[best_index]
            if self._should_update(chunk, confidence):
                profile.centroid = l2_normalize(
                    self.config.centroid_update_alpha * profile.centroid
                    + (1.0 - self.config.centroid_update_alpha) * vector
                )
                profile.num_embeddings += 1
                profile.last_seen_time = chunk.end
            return SpeakerResult(chunk.chunk_id, profile.speaker_id, best_score, vector)

        profile = self._create_profile(vector, chunk.end)
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

    def _should_update(self, chunk: AudioChunk, confidence: float) -> bool:
        return (
            chunk.duration >= self.config.min_embedding_duration
            and confidence >= self.config.min_update_confidence
        )


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return vector.astype(np.float32, copy=False)
    return (vector / norm).astype(np.float32, copy=False)
