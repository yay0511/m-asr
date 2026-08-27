from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from m_asr.config import SpeakerConfig
from m_asr.types import (
    AudioChunk,
    GlobalSpeakerSegment,
    LocalDiarizationResult,
    LocalSpeakerAssignment,
    LocalSpeakerTrack,
    SpeakerResult,
)


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

    def assign_local_tracks(
        self,
        result: LocalDiarizationResult,
    ) -> tuple[list[GlobalSpeakerSegment], list[LocalSpeakerAssignment]]:
        """Map chunk-local pyannote tracks to persistent online speakers.

        Pyannote local labels are intentionally discarded after this method. The
        registry is the only component allowed to create or update global IDs.
        Tracks are processed by duration so a short noisy track cannot steal a
        profile from the main track in the same chunk.
        """
        tracks = [
            track for track in result.tracks
            if track.embedding is not None
            and track.speech_duration >= self.config.local_min_track_duration
        ]
        tracks.sort(key=lambda track: track.speech_duration, reverse=True)
        used_profiles: set[str] = set()
        assignments: list[LocalSpeakerAssignment] = []
        previous_last_speaker_id = self._last_speaker_id

        for track in tracks:
            vector = l2_normalize(np.asarray(track.embedding, dtype=np.float32).reshape(-1))
            if vector.size == 0 or not np.all(np.isfinite(vector)):
                continue
            profile, score = self._match_local(
                vector,
                used_profiles,
                last_speaker_id=previous_last_speaker_id,
            )
            if profile is None:
                force_new_for_unoccupied_track = (
                    bool(self._profiles) and len(used_profiles) >= len(self._profiles)
                )
                if self._can_create_local(track, result) or force_new_for_unoccupied_track:
                    profile = self._create_profile(vector, result.end)
                    score = 1.0
                elif self.config.assign_uncertain_to_best:
                    profile, score = self._best_profile(vector, used_profiles)
                else:
                    continue
            if profile is None:
                continue
            used_profiles.add(profile.speaker_id)
            if score >= self.config.min_centroid_update_similarity:
                self._update_profile(profile, vector, result.end)
            for segment in track.segments:
                assignments.append(
                    LocalSpeakerAssignment(track.local_id, profile.speaker_id, score, segment)
                )

        if assignments:
            latest = max(assignments, key=lambda item: item.segment.end)
            self._last_speaker_id = latest.speaker_id

        segments = [
            GlobalSpeakerSegment(
                start=item.segment.start,
                end=item.segment.end,
                speaker_id=item.speaker_id,
                confidence=item.confidence,
                overlap=item.segment.overlap,
            )
            for item in assignments
        ]
        segments.sort(key=lambda item: (item.start, item.end, item.speaker_id))
        return segments, assignments

    def _match_local(
        self,
        vector: np.ndarray,
        used_profiles: set[str],
        last_speaker_id: str | None = None,
    ) -> tuple[SpeakerProfile | None, float]:
        if not self._profiles:
            return None, 0.0
        candidates = [
            (profile, float(np.dot(vector, profile.centroid)))
            for profile in self._profiles
            if profile.speaker_id not in used_profiles
        ]
        if not candidates:
            return None, 0.0
        candidates.sort(key=lambda item: item[1], reverse=True)
        best, best_score = candidates[0]
        last = next((item for item in candidates if item[0].speaker_id == last_speaker_id), None)
        if last is not None and len(self._profiles) > 1:
            last_profile, last_score = last
            if (
                last_score >= self.config.local_last_speaker_threshold
                and best.speaker_id == last_profile.speaker_id
                and best_score - last_score <= self.config.local_last_speaker_margin
            ):
                return last_profile, last_score
        if best_score >= self.config.local_match_threshold:
            return best, best_score
        return None, best_score

    def _best_profile(
        self,
        vector: np.ndarray,
        used_profiles: set[str],
    ) -> tuple[SpeakerProfile | None, float]:
        candidates = [
            (profile, float(np.dot(vector, profile.centroid)))
            for profile in self._profiles
            if profile.speaker_id not in used_profiles
        ]
        return max(candidates, key=lambda item: item[1], default=(None, 0.0))

    def _can_create_local(self, track: LocalSpeakerTrack, result: LocalDiarizationResult) -> bool:
        if track.speech_duration < self.config.local_min_track_duration:
            return False
        progress = max(0.0, min(1.0, result.end / max(self.config.new_speaker_warmup_seconds, 1e-6)))
        limit = _lerp(
            self.config.new_speaker_initial_max_similarity,
            self.config.new_speaker_final_max_similarity,
            progress,
        )
        if not self._profiles:
            return True
        vector = l2_normalize(np.asarray(track.embedding, dtype=np.float32).reshape(-1))
        best = max(float(np.dot(vector, profile.centroid)) for profile in self._profiles)
        return best <= max(limit, self.config.local_new_speaker_threshold)

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
