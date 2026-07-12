from __future__ import annotations

import argparse
import math
import sys
import wave
from pathlib import Path

import numpy as np

from .audio_buffer import linear_resample, to_mono_float32
from .config import load_config
from .pipeline import StreamingCascadePipeline
from .types import PipelineEvent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Multi-speaker streaming ASR cascade demo")
    parser.add_argument("--config", default="configs/local.yaml", help="YAML config path")
    parser.add_argument("--audio", default=None, help="WAV file to process; omitted uses synthetic demo audio")
    parser.add_argument("--frame-seconds", type=float, default=0.2, help="stream feed frame size")
    args = parser.parse_args(argv)

    config = load_config(args.config if args.config else None)

    pipeline = StreamingCascadePipeline(config)
    print(f"[m_asr] ASR backend={pipeline.backends.asr} speaker backend={pipeline.backends.speaker}")

    if args.audio:
        waveform, sample_rate = read_wav(args.audio)
        print(f"[input] {args.audio} sr={sample_rate} samples={len(waveform)}")
    else:
        sample_rate = config.runtime.sample_rate
        waveform = synthetic_demo_audio(sample_rate)
        print("[input] synthetic demo audio")

    chunk_count = 0
    for event in pipeline.process_waveform(waveform, sample_rate, frame_seconds=args.frame_seconds):
        if event.event_type == "chunk_finalized":
            chunk_count += 1
        print(format_event(event))

    print("[transcript]")
    for turn in pipeline.transcript:
        print(f"[{turn.start:.2f} - {turn.end:.2f}] {turn.speaker_id}: {turn.text}")
    if chunk_count == 0:
        print("[warn  ] no speech chunks emitted; check audio level or chunker.energy_reference")
    elif not pipeline.transcript:
        print("[warn  ] speech chunks were emitted, but ASR returned no final text")
    return 0


def format_event(event: PipelineEvent) -> str:
    if event.event_type == "chunk_finalized":
        return f"[chunk ] #{event.chunk_id} {event.start:.2f}-{event.end:.2f}"
    if event.event_type == "partial":
        return f"[partial] {event.format_turn()}"
    if event.event_type == "speaker":
        return (
            f"[speaker] #{event.chunk_id} {event.speaker_id} "
            f"confidence={event.confidence:.3f}"
        )
    if event.event_type == "final":
        return f"[final ] {event.format_turn()}"
    return f"[error ] #{event.chunk_id} {event.message}"


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    try:
        import soundfile as sf

        data, sample_rate = sf.read(path, dtype="float32", always_2d=True)
        return to_mono_float32(data), int(sample_rate)
    except ModuleNotFoundError:
        return _read_wav_stdlib(path)


def _read_wav_stdlib(path: str | Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        raw = wf.readframes(wf.getnframes())

    if sample_width == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    elif sample_width == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"unsupported WAV sample width: {sample_width}")

    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data.astype(np.float32), sample_rate


def synthetic_demo_audio(sample_rate: int = 16000) -> np.ndarray:
    parts = [
        np.zeros(int(0.5 * sample_rate), dtype=np.float32),
        _tone(sample_rate, 1.3, 220.0, 0.08),
        np.zeros(int(0.9 * sample_rate), dtype=np.float32),
        _tone(sample_rate, 1.5, 440.0, 0.08),
        np.zeros(int(0.9 * sample_rate), dtype=np.float32),
        _tone(sample_rate, 1.2, 220.0, 0.08),
        np.zeros(int(1.0 * sample_rate), dtype=np.float32),
    ]
    return np.concatenate(parts)


def _tone(sample_rate: int, seconds: float, frequency: float, amplitude: float) -> np.ndarray:
    samples = int(round(seconds * sample_rate))
    x = np.arange(samples, dtype=np.float32) / sample_rate
    envelope = np.minimum(1.0, np.arange(samples, dtype=np.float32) / max(1, int(0.05 * sample_rate)))
    envelope = np.minimum(envelope, envelope[::-1])
    return (amplitude * envelope * np.sin(2.0 * math.pi * frequency * x)).astype(np.float32)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
