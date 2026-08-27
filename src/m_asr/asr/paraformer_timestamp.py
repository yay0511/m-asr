from __future__ import annotations

import re
import inspect
import time
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

from m_asr.config import AppConfig
from m_asr.types import (
    AudioChunk,
    TimestampedAsrResult,
    TimestampedCharacter,
    TimestampedWord,
)


class ParaformerTimestampClient:
    """Offline Paraformer branch used only to provide word/character timing.

    X-ASR remains the display ASR. This branch is deliberately independent so
    timestamp failures cannot stop the primary streaming decoder.
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self._model = None
        self._backend = "disabled"
        self._init_error: str | None = None
        if not config.timestamp_asr.enabled:
            return
        model_dir = Path(config.paths.paraformer_model_dir)
        if not model_dir.exists():
            raise FileNotFoundError(f"Paraformer model not found: {model_dir}")
        checkpoint = model_dir / "model.pt"
        if _is_git_lfs_pointer(checkpoint):
            self._backend = "unavailable"
            self._init_error = (
                f"Paraformer checkpoint is still a Git-LFS pointer: {checkpoint}. "
                "Download the real model.pt before enabling timestamp inference."
            )
            return
        try:
            from funasr import AutoModel
        except ModuleNotFoundError as exc:
            self._backend = "unavailable"
            self._init_error = (
                "FunASR is not installed; timestamp branch is unavailable. "
                "Install it with `python -m pip install funasr`."
            )
            return

        device = self._resolve_device()
        self._model = _load_funasr_model(AutoModel, model_dir, device)
        self._backend = "real"

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def init_error(self) -> str | None:
        return self._init_error

    def recognize(self, chunk: AudioChunk) -> TimestampedAsrResult:
        started = time.perf_counter()
        if self._model is None or chunk.duration < self.config.timestamp_asr.min_chunk_duration:
            return TimestampedAsrResult(chunk.chunk_id, "", [], [], "disabled", 0.0)
        result = self._model.generate(
            input=np.asarray(chunk.waveform, dtype=np.float32),
            fs=chunk.sample_rate,
            pred_timestamp=self.config.timestamp_asr.pred_timestamp,
            disable_pbar=True,
        )
        record = result[0] if isinstance(result, list) and result else result
        if not isinstance(record, dict):
            return TimestampedAsrResult(chunk.chunk_id, "", [], [], "invalid", _elapsed(started))
        text = str(record.get("text", "") or "")
        raw_timestamps = record.get("timestamp", record.get("time_stamp", [])) or []
        characters = _make_characters(text, raw_timestamps, chunk.start)
        words = _characters_to_words(characters)
        return TimestampedAsrResult(
            chunk_id=chunk.chunk_id,
            text=text,
            characters=characters,
            words=words,
            status="matched" if raw_timestamps else "no_timestamp",
            latency_ms=_elapsed(started),
        )

    def align_to_xasr(self, xasr_text: str, result: TimestampedAsrResult) -> TimestampedAsrResult:
        if not xasr_text or not result.text:
            return result
        lhs = _compact(xasr_text)
        rhs = _compact(result.text)
        ratio = SequenceMatcher(None, lhs, rhs).ratio()
        if ratio < 0.45:
            result.status = "unmatched"
            return result
        result.status = "matched" if ratio >= 0.75 else "weak_match"
        return result

    def _resolve_device(self) -> str:
        requested = str(self.config.timestamp_asr.device or self.config.runtime.device).lower()
        if requested != "cuda":
            return "cpu"
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        print("[warn  ] Paraformer CUDA unavailable; using CPU for timestamps.")
        return "cpu"


def _make_characters(text: str, raw: object, offset: float) -> list[TimestampedCharacter]:
    spans: list[tuple[float, float]] = []
    if isinstance(raw, (list, tuple)):
        for item in raw:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                try:
                    spans.append((float(item[0]) / 1000.0 + offset, float(item[1]) / 1000.0 + offset))
                except (TypeError, ValueError):
                    continue
    if not text:
        return []
    if not spans:
        duration = max(0.01, len(text) and 0.01)
        spans = [(offset + i * duration, offset + (i + 1) * duration) for i in range(len(text))]
    if len(spans) != len(text):
        total_start, total_end = spans[0][0], spans[-1][1]
        step = max(0.01, (total_end - total_start) / max(1, len(text)))
        spans = [(total_start + i * step, total_start + (i + 1) * step) for i in range(len(text))]
    return [TimestampedCharacter(ch, start, max(start, end)) for ch, (start, end) in zip(text, spans)]


def _characters_to_words(characters: list[TimestampedCharacter]) -> list[TimestampedWord]:
    words: list[TimestampedWord] = []
    current: list[TimestampedCharacter] = []
    for item in characters:
        if item.text.isspace() or re.match(r"[，。！？；：,.!?;:]", item.text):
            if current:
                words.append(TimestampedWord("".join(x.text for x in current), current[0].start, current[-1].end))
                current = []
            continue
        current.append(item)
    if current:
        words.append(TimestampedWord("".join(x.text for x in current), current[0].start, current[-1].end))
    return words


def _compact(text: str) -> str:
    return "".join(ch for ch in text.lower() if not ch.isspace() and not re.match(r"[，。！？；：,.!?;:]", ch))


def _elapsed(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0


def _is_git_lfs_pointer(path: Path) -> bool:
    try:
        if path.stat().st_size > 1024:
            return False
        prefix = path.read_bytes()[:128]
    except OSError:
        return False
    return prefix.startswith(b"version https://git-lfs.github.com/spec/v1")


def _load_funasr_model(auto_model: object, model_dir: Path, device: str) -> object:
    """Load trusted local FunASR checkpoints across PyTorch 2.6+.

    FunASR 1.4.4 still calls ``torch.load`` without ``weights_only``. The
    local checkpoint is a trusted project asset, so temporarily request the
    legacy full-checkpoint behavior during model construction only.
    """
    import torch

    original_load = torch.load
    supports_weights_only = "weights_only" in inspect.signature(original_load).parameters
    if supports_weights_only:
        def trusted_load(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return original_load(*args, **kwargs)

        torch.load = trusted_load
    try:
        return auto_model(
            model=str(model_dir),
            device=device,
            disable_update=True,
            disable_log=True,
        )
    finally:
        if supports_weights_only:
            torch.load = original_load
