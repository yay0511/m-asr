from __future__ import annotations

import importlib
import sys
from pathlib import Path

from m_asr.config import load_config


def exists(label: str, path: str) -> None:
    status = "OK" if Path(path).exists() else "MISSING"
    print(f"{status:7} {label}: {path}")


def module(label: str, name: str) -> None:
    try:
        importlib.import_module(name)
        found = True
        error = ""
    except Exception as exc:
        found = False
        error = f" ({type(exc).__name__}: {exc})"
    status = "OK" if found else "MISSING"
    print(f"{status:7} module {label}: {name}{error}")


def main() -> int:
    config = load_config("configs/local.yaml")
    pyannote_src = Path(config.paths.pyannote_audio_root) / "src"
    if pyannote_src.exists() and str(pyannote_src) not in sys.path:
        sys.path.insert(0, str(pyannote_src))

    exists("X-ASR root", config.paths.x_asr_root)
    exists("X-ASR model", config.paths.x_asr_model_dir)
    exists("pyannote audio root", config.paths.pyannote_audio_root)
    exists("pyannote model", config.paths.pyannote_model_dir)
    exists("pyannote embedding", str(Path(config.paths.pyannote_model_dir) / "embedding"))
    module("numpy", "numpy")
    module("torch", "torch")
    module("pyannote.audio", "pyannote.audio")
    module("sherpa_onnx", "sherpa_onnx")
    module("soundfile", "soundfile")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
