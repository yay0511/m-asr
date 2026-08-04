#!/usr/bin/env bash
set -euo pipefail

ROOT="${M_ASR_ROOT:-/root/shared-nvme/yuxinliu/m_asr}"
PYANNOTE_PYTHON="${PYANNOTE_GPU_PYTHON:-/root/.conda/envs/pyannote/bin/python}"

export M_ASR_DEVICE="${M_ASR_DEVICE:-cuda}"
export M_ASR_ASR_PROVIDER="${M_ASR_ASR_PROVIDER:-auto}"
export PYTHONPATH="$ROOT/src:${X_ASR_ROOT:-/root/shared-nvme/yuxinliu/X-ASR}:${PYANNOTE_AUDIO_ROOT:-/root/shared-nvme/yuxinliu/pyannote-audio-4.0.7}/src:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/m_asr_matplotlib}"
mkdir -p "$MPLCONFIGDIR"

if [[ ! -x "$PYANNOTE_PYTHON" ]]; then
  echo "[web-gpu] missing python: $PYANNOTE_PYTHON" >&2
  exit 2
fi

PYANNOTE_SITE="$("$PYANNOTE_PYTHON" - <<'PY'
import site

print(site.getsitepackages()[0])
PY
)"
NVIDIA_LIB_ROOT="$PYANNOTE_SITE/nvidia"
CUDA_LIB_PATHS=(
  "$NVIDIA_LIB_ROOT/cublas/lib"
  "$NVIDIA_LIB_ROOT/cudnn/lib"
  "$NVIDIA_LIB_ROOT/cufft/lib"
  "$NVIDIA_LIB_ROOT/curand/lib"
  "$NVIDIA_LIB_ROOT/cuda_runtime/lib"
  "$NVIDIA_LIB_ROOT/cuda_nvrtc/lib"
)
for lib_path in "${CUDA_LIB_PATHS[@]}"; do
  if [[ -d "$lib_path" ]]; then
    export LD_LIBRARY_PATH="$lib_path:${LD_LIBRARY_PATH:-}"
  fi
done

"$PYANNOTE_PYTHON" - <<'PY'
from pathlib import Path
import sys
import torch

print(f"[web-gpu] torch={torch.__version__} cuda_build={torch.version.cuda}")
if not list(Path("/dev").glob("nvidia*")):
    print(
        "[web-gpu] no /dev/nvidia* devices are visible in this container; "
        "start this environment with GPU access before launching the GPU web server.",
        file=sys.stderr,
    )
    raise SystemExit(2)
if not torch.cuda.is_available():
    print("[web-gpu] CUDA is not available; refusing to start GPU web server.", file=sys.stderr)
    raise SystemExit(2)
print(f"[web-gpu] device_count={torch.cuda.device_count()}")
print(f"[web-gpu] device_0={torch.cuda.get_device_name(0)}")
PY

cd "$ROOT"
exec "$PYANNOTE_PYTHON" -m m_asr.web_app --config configs/local.yaml "$@"
