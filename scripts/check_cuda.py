from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import site


def configure_nvidia_library_path() -> None:
    import os

    paths: list[str] = []
    for site_dir in site.getsitepackages():
        nvidia_root = Path(site_dir) / "nvidia"
        for package in ("cublas", "cudnn", "cufft", "curand", "cuda_runtime", "cuda_nvrtc"):
            lib_dir = nvidia_root / package / "lib"
            if lib_dir.is_dir():
                paths.append(str(lib_dir))
    if paths:
        os.environ["LD_LIBRARY_PATH"] = ":".join(paths + [os.environ.get("LD_LIBRARY_PATH", "")])


def main() -> int:
    configure_nvidia_library_path()

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
        package_dir = Path(sherpa_onnx.__file__).resolve().parent
        provider_files = [
            path
            for path in package_dir.rglob("*")
            if "onnxruntime_providers_cuda" in path.name.lower()
        ]
        has_cuda_provider = bool(provider_files)
        print(f"inferred CUDA provider files: {has_cuda_provider}")
        if not has_cuda_provider:
            print("note: this sherpa-onnx install appears CPU-only; CUDA provider requires a GPU-enabled build.")
        for provider_file in provider_files:
            result = subprocess.run(["ldd", str(provider_file)], text=True, capture_output=True, check=False)
            missing = [
                line.strip()
                for line in result.stdout.splitlines()
                if "not found" in line
            ]
            if missing:
                print(f"missing dynamic libraries for {provider_file.name}:")
                for line in missing:
                    print(f"  {line}")
            else:
                print(f"dynamic libraries for {provider_file.name}: ok")
    except Exception as exc:
        print(f"sherpa_onnx import: failed ({type(exc).__name__}: {exc})")
        return 1

    return 0 if torch.cuda.is_available() else 2


if __name__ == "__main__":
    raise SystemExit(main())
