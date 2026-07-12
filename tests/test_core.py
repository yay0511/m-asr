import numpy as np

from m_asr.chunker import SpeechChunker
from m_asr.config import AppConfig
from m_asr.diarization.speaker_registry import SpeakerRegistry
from m_asr.types import AudioChunk


def test_chunker_emits_speech_chunk():
    config = AppConfig()
    chunker = SpeechChunker(config.chunker, sample_rate=config.runtime.sample_rate)
    sr = config.runtime.sample_rate
    waveform = np.concatenate(
        [
            np.zeros(int(0.4 * sr), dtype=np.float32),
            np.ones(int(1.0 * sr), dtype=np.float32) * 0.08,
            np.zeros(int(1.0 * sr), dtype=np.float32),
        ]
    )
    chunks = chunker.accept(waveform) + chunker.flush()
    assert len(chunks) == 1
    assert chunks[0].duration >= config.chunker.min_chunk_duration


def test_speaker_registry_matches_existing_speaker():
    config = AppConfig()
    registry = SpeakerRegistry(config.speaker)
    chunk = AudioChunk(0, 0.0, 1.5, np.zeros(24000, dtype=np.float32), 16000)
    first = registry.match(chunk, np.asarray([1.0, 0.0], dtype=np.float32))
    second = registry.match(chunk, np.asarray([0.99, 0.01], dtype=np.float32))
    assert first.speaker_id == "SPEAKER_01"
    assert second.speaker_id == "SPEAKER_01"


def test_config_defaults_to_real_backends():
    config = AppConfig()
    assert config.asr.mode == "real"
    assert config.speaker.mode == "real"
