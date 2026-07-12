from __future__ import annotations

import glob
import os
import re
import time
from pathlib import Path

import numpy as np

from m_asr.config import AppConfig
from m_asr.types import AsrResult, AudioChunk


_CJK_RANGE = r"\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
_CJK_PUNCT = re.escape("，。！？；：、（）《》〈〉【】「」『』“”‘’")
_ASCII_PUNCT_NO_LEADING_SPACE = re.escape(",.!?;:%)]}")


def normalize_cjk_spacing(text: str) -> str:
    text = re.sub(rf"(?<=[{_CJK_RANGE}])\s+(?=[{_CJK_RANGE}])", "", text)
    text = re.sub(rf"(?<=[{_CJK_RANGE}])\s+(?=[{_CJK_PUNCT}])", "", text)
    text = re.sub(rf"(?<=[{_CJK_PUNCT}])\s+(?=[{_CJK_RANGE}])", "", text)
    text = re.sub(rf"(?<=[{_CJK_PUNCT}])\s+(?=[{_CJK_PUNCT}])", "", text)
    text = re.sub(rf"\s+(?=[{_ASCII_PUNCT_NO_LEADING_SPACE}])", "", text)
    return text


class XAsrClient:
    """Chunk-level X-ASR adapter backed by local sherpa-onnx model files."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.mode = config.asr.mode
        self._recognizer = None
        self._kind = "real"
        self._files: dict[str, str] = {}

        if self.mode != "real":
            raise ValueError(f"X-ASR mode must be 'real', got {self.mode!r}")
        self._init_real()

    @property
    def backend(self) -> str:
        return self._kind

    def recognize(self, chunk: AudioChunk) -> AsrResult:
        started = time.perf_counter()
        text = self._recognize_real(chunk)
        latency_ms = (time.perf_counter() - started) * 1000.0
        return AsrResult(chunk.chunk_id, text.strip(), True, latency_ms)

    def _init_real(self) -> None:
        import sherpa_onnx

        model_dir = Path(self.config.paths.x_asr_model_dir)
        kind, files = _find_asr_files(model_dir)
        common = {
            "num_threads": 2,
            "provider": self.config.runtime.asr_provider,
            "decoding_method": "greedy_search",
            "enable_endpoint_detection": False,
        }
        if kind == "transducer":
            self._recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
                tokens=files["tokens"],
                encoder=files["encoder"],
                decoder=files["decoder"],
                joiner=files["joiner"],
                sample_rate=self.config.runtime.sample_rate,
                feature_dim=80,
                model_type="zipformer2",
                **common,
            )
        elif kind == "wenet-ctc":
            self._recognizer = sherpa_onnx.OnlineRecognizer.from_wenet_ctc(
                tokens=files["tokens"],
                model=files["model"],
                chunk_size=16,
                num_left_chunks=4,
                **common,
            )
        else:
            raise ValueError(f"unsupported X-ASR model kind: {kind}")
        self._files = files

    def _recognize_real(self, chunk: AudioChunk) -> str:
        if self._recognizer is None:
            return ""

        stream = self._recognizer.create_stream()
        samples = np.asarray(chunk.waveform, dtype=np.float32).reshape(-1)
        stream.accept_waveform(chunk.sample_rate, samples)
        tail = np.zeros(
            int(round(self.config.asr.tail_padding_seconds * chunk.sample_rate)),
            dtype=np.float32,
        )
        if tail.size:
            stream.accept_waveform(chunk.sample_rate, tail)
        stream.input_finished()

        while self._recognizer.is_ready(stream):
            self._recognizer.decode_stream(stream)

        text = self._recognizer.get_result(stream)
        if self.config.asr.text_format == "lower":
            text = text.lower()
        elif self.config.asr.text_format == "capitalize" and text:
            text = text[:1].upper() + text[1:].lower()
        return normalize_cjk_spacing(text)


def _find_asr_files(model_dir: Path) -> tuple[str, dict[str, str]]:
    if not model_dir.exists():
        raise FileNotFoundError(f"X-ASR model directory not found: {model_dir}")
    onnx = sorted(glob.glob(str(model_dir / "*.onnx")))
    onnx = [path for path in onnx if "vad" not in os.path.basename(path).lower()]
    if not onnx:
        raise FileNotFoundError(f"no .onnx ASR model files found under {model_dir}")

    tokens = str(model_dir / "tokens.txt")
    if not os.path.isfile(tokens):
        raise FileNotFoundError(f"missing tokens file: {tokens}")

    def pick(substr: str) -> str | None:
        candidates = [p for p in onnx if substr in os.path.basename(p).lower()]
        no_int8 = [p for p in candidates if "int8" not in os.path.basename(p).lower()]
        return (no_int8 or candidates or [None])[0]

    encoder = pick("encoder")
    decoder = pick("decoder")
    joiner = pick("joiner")
    if encoder and decoder and joiner:
        return "transducer", {
            "tokens": tokens,
            "encoder": encoder,
            "decoder": decoder,
            "joiner": joiner,
        }

    model = pick("model") or pick("ctc") or encoder
    if model:
        return "wenet-ctc", {"tokens": tokens, "model": model}

    raise FileNotFoundError(f"could not infer ASR model layout under {model_dir}")
