from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class PathConfig:
    x_asr_root: str = "/root/shared-nvme/yuxinliu/X-ASR"
    pyannote_audio_root: str = "/root/shared-nvme/yuxinliu/pyannote-audio-4.0.7"
    pyannote_model_dir: str = "/root/shared-nvme/yuxinliu/pyannote-speaker-diarization-community-1"
    x_asr_model_dir: str = (
        "/root/shared-nvme/yuxinliu/X-ASR/X-ASR-zh-en/deployment/models/chunk-960ms-model"
    )


@dataclass(slots=True)
class RuntimeConfig:
    sample_rate: int = 16000
    device: str = "cuda"
    asr_provider: str = "cuda"
    max_workers: int = 2


@dataclass(slots=True)
class ChunkerConfig:
    frame_ms: int = 20
    speech_onset_threshold: float = 0.5
    speech_offset_threshold: float = 0.35
    energy_reference: float = 0.008
    end_silence_ms: int = 700
    min_chunk_duration: float = 0.8
    max_chunk_duration: float = 12.0
    left_padding_ms: int = 300
    right_padding_ms: int = 300


@dataclass(slots=True)
class SpeakerConfig:
    same_speaker_threshold: float = 0.65
    centroid_update_alpha: float = 0.85
    min_embedding_duration: float = 1.0
    min_update_confidence: float = 0.6
    mode: str = "real"


@dataclass(slots=True)
class AsrConfig:
    mode: str = "real"
    text_format: str = "none"
    tail_padding_seconds: float = 1.0


@dataclass(slots=True)
class AppConfig:
    paths: PathConfig = field(default_factory=PathConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    chunker: ChunkerConfig = field(default_factory=ChunkerConfig)
    speaker: SpeakerConfig = field(default_factory=SpeakerConfig)
    asr: AsrConfig = field(default_factory=AsrConfig)


def load_config(path: str | os.PathLike[str] | None = None) -> AppConfig:
    config = AppConfig()
    _apply_env(config)

    if path is None:
        return config

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"config not found: {config_path}")

    data = _load_mapping(config_path)
    _apply_mapping(config, data)
    _apply_env(config)
    return config


def _apply_env(config: AppConfig) -> None:
    env_map = {
        "X_ASR_ROOT": (config.paths, "x_asr_root", str),
        "PYANNOTE_AUDIO_ROOT": (config.paths, "pyannote_audio_root", str),
        "PYANNOTE_MODEL_DIR": (config.paths, "pyannote_model_dir", str),
        "X_ASR_MODEL_DIR": (config.paths, "x_asr_model_dir", str),
        "M_ASR_DEVICE": (config.runtime, "device", str),
        "M_ASR_ASR_PROVIDER": (config.runtime, "asr_provider", str),
        "M_ASR_ASR_MODE": (config.asr, "mode", str),
        "M_ASR_SPEAKER_MODE": (config.speaker, "mode", str),
    }
    for key, (obj, attr, caster) in env_map.items():
        value = os.getenv(key)
        if value:
            setattr(obj, attr, caster(value))


def _load_mapping(path: Path) -> dict[str, Any]:
    try:
        import yaml

        with path.open("r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"top-level config must be a mapping: {path}")
        return loaded
    except ModuleNotFoundError:
        return _load_simple_yaml(path)


def _load_simple_yaml(path: Path) -> dict[str, Any]:
    root: dict[str, Any] = {}
    current: dict[str, Any] | None = None

    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.split("#", 1)[0].rstrip()
            if not line.strip():
                continue
            if not line.startswith(" "):
                key = line.rstrip(":")
                current = {}
                root[key] = current
                continue
            if current is None or ":" not in line:
                continue
            key, value = line.strip().split(":", 1)
            current[key] = _parse_scalar(value.strip())
    return root


def _parse_scalar(value: str) -> Any:
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.lower() in {"null", "none"}:
        return None
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value.strip("\"'")


def _apply_mapping(config: AppConfig, data: dict[str, Any]) -> None:
    sections = {
        "paths": config.paths,
        "runtime": config.runtime,
        "chunker": config.chunker,
        "speaker": config.speaker,
        "asr": config.asr,
    }
    for section_name, values in data.items():
        target = sections.get(section_name)
        if target is None or not isinstance(values, dict):
            continue
        for key, value in values.items():
            if hasattr(target, key):
                setattr(target, key, value)
