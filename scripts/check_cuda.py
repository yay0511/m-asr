from __future__ import annotations

import shutil
import subprocess


def main() -> int:
    print("[cuda] nvidia-smi")
    if shutil.which("nvidia-smi"):
        result = subprocess.run(["nvidia-smi"], text=True, capture_output=True, check=False)
        print(result.stdout.strip() or result.stderr.strip())
    else:
        print("nvidia-smi not found")

    print("\n[cuda] torch")
    import torch

    print(f"torch version: {torch.__version__}")
    print(f"torch CUDA build: {torch.version.cuda}")
    print(f"cuda available: {torch.cuda.is_available()}")
    print(f"device count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            print(f"device {index}: {torch.cuda.get_device_name(index)}")

    print("\n[cuda] sherpa-onnx")
    try:
        import sherpa_onnx

        print(f"sherpa_onnx import: ok ({getattr(sherpa_onnx, '__version__', 'unknown')})")
        print("note: pip sherpa-onnx wheels may be CPU-only; CUDA provider requires a GPU-enabled build.")
    except Exception as exc:
        print(f"sherpa_onnx import: failed ({type(exc).__name__}: {exc})")
        return 1

    return 0 if torch.cuda.is_available() else 2


if __name__ == "__main__":
    raise SystemExit(main())
