import numpy as np

from m_asr.chunker import SpeechChunker
from m_asr.config import AppConfig
from m_asr.diarization.speaker_registry import SpeakerRegistry
from m_asr.types import AudioChunk
from m_asr.web_app import _select_streaming_final_text


def test_chunker_emits_speech_chunk():
    config = AppConfig()
    config.chunker.vad_provider = "energy"
    config.chunker.frame_ms = 20
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
    config.speaker.min_new_speaker_duration_initial = 1.0
    config.speaker.min_new_speaker_duration_final = 1.0
    registry = SpeakerRegistry(config.speaker)
    chunk = AudioChunk(0, 0.0, 1.5, np.zeros(24000, dtype=np.float32), 16000)
    first = registry.match(chunk, np.asarray([1.0, 0.0], dtype=np.float32))
    second = registry.match(chunk, np.asarray([0.99, 0.01], dtype=np.float32))
    assert first.speaker_id == "SPEAKER_01"
    assert second.speaker_id == "SPEAKER_01"


def test_speaker_registry_is_conservative_during_warmup():
    config = AppConfig()
    registry = SpeakerRegistry(config.speaker)
    short_chunk = AudioChunk(0, 0.0, 0.3, np.zeros(4800, dtype=np.float32), 16000)
    result = registry.match(short_chunk, np.asarray([1.0, 0.0], dtype=np.float32))
    assert result.speaker_id == "UNKNOWN"
    assert registry.profiles == ()


def test_speaker_registry_prefers_recent_speaker_with_margin():
    config = AppConfig()
    config.speaker.min_new_speaker_duration_initial = 1.0
    config.speaker.min_new_speaker_duration_final = 1.0
    config.speaker.min_centroid_update_similarity = 0.9
    registry = SpeakerRegistry(config.speaker)

    first_chunk = AudioChunk(0, 0.0, 2.0, np.zeros(32000, dtype=np.float32), 16000)
    second_chunk = AudioChunk(1, 2.2, 4.2, np.zeros(32000, dtype=np.float32), 16000)
    registry.match(first_chunk, np.asarray([1.0, 0.0], dtype=np.float32))

    result = registry.match(second_chunk, np.asarray([0.57, 0.821645], dtype=np.float32))
    assert result.speaker_id == "SPEAKER_01"


def test_config_defaults_to_real_backends():
    config = AppConfig()
    assert config.asr.mode == "real"
    assert config.speaker.mode == "real"
    assert config.runtime.device == "cuda"
    assert config.runtime.asr_provider == "auto"


def test_streaming_final_text_keeps_longer_partial_when_finish_drops_tail():
    text = _select_streaming_final_text("今天天气很", "今天天气很好")

    assert text == "今天天气很好"


def test_streaming_final_text_prefers_finished_when_it_extends_partial():
    text = _select_streaming_final_text("今天天气很好", "今天天气很")

    assert text == "今天天气很好"
