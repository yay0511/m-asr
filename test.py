from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import torch
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook


DEFAULT_MODEL = "/root/shared-nvme/yuxinliu/pyannote-speaker-diarization-community-1"
DEFAULT_AUDIO = "/root/shared-nvme/yuxinliu/test_10s.wav"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run pyannote diarization on GPU.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--audio", default=DEFAULT_AUDIO)
    parser.add_argument("--allow-cpu", action="store_true", help="fall back to CPU when CUDA is unavailable")
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        raise FileNotFoundError(f"audio file not found: {audio_path}")
    if audio_path.suffix.lower() not in {".wav", ".flac", ".mp3", ".m4a", ".ogg"}:
        raise ValueError(f"audio input must be an audio file, got: {audio_path}")

    device = resolve_device(allow_cpu=args.allow_cpu)
    print(f"[test] torch={torch.__version__} cuda_build={torch.version.cuda} device={device}")

    pipeline = Pipeline.from_pretrained(args.model, token=os.environ.get("HF_TOKEN"))
    pipeline.to(device)

    with ProgressHook() as hook:
        output = pipeline(str(audio_path), hook=hook)

    diarization = getattr(output, "speaker_diarization", output)
    for turn, _track, speaker in diarization.itertracks(yield_label=True):
        print(f"start={turn.start:.1f}s stop={turn.end:.1f}s {speaker}")
    return 0


def resolve_device(allow_cpu: bool) -> torch.device:
    try:
        cuda_available = torch.cuda.is_available()
    except Exception as exc:
        cuda_available = False
        print(f"[cuda] torch CUDA check failed: {type(exc).__name__}: {exc}", file=sys.stderr)

    if cuda_available:
        print(f"[cuda] device_count={torch.cuda.device_count()}")
        print(f"[cuda] device_0={torch.cuda.get_device_name(0)}")
        return torch.device("cuda")

    print_cuda_diagnostics()
    if allow_cpu:
        print("[warn] CUDA unavailable; falling back to CPU because --allow-cpu was set.", file=sys.stderr)
        return torch.device("cpu")

    raise RuntimeError(
        "CUDA is unavailable in this environment. "
        "Use a GPU-enabled container/pod with /dev/nvidia* devices and install a torch CUDA wheel "
        "compatible with the host NVIDIA driver, or rerun with --allow-cpu."
    )


def print_cuda_diagnostics() -> None:
    print(f"[cuda] torch={torch.__version__}", file=sys.stderr)
    print(f"[cuda] torch CUDA build={torch.version.cuda}", file=sys.stderr)
    print(f"[cuda] cuda available={torch.cuda.is_available()}", file=sys.stderr)
    nvidia_devices = sorted(Path("/dev").glob("nvidia*"))
    print(f"[cuda] /dev/nvidia*: {[str(path) for path in nvidia_devices]}", file=sys.stderr)
    try:
        result = subprocess.run(["nvidia-smi"], text=True, capture_output=True, check=False)
        output = (result.stdout or result.stderr).strip()
        print(f"[cuda] nvidia-smi exit={result.returncode}", file=sys.stderr)
        if output:
            print(output, file=sys.stderr)
    except FileNotFoundError:
        print("[cuda] nvidia-smi not found", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
