#!/usr/bin/env bash
set -euo pipefail

ROOT="${M_ASR_ROOT:-/root/shared-nvme/yuxinliu/m_asr}"

if [[ "${M_ASR_REQUIRE_GPU:-0}" == "1" ]]; then
  exec "$ROOT/scripts/run_web_gpu.sh" "$@"
fi

export PYTHONPATH="$ROOT/src:${X_ASR_ROOT:-/root/shared-nvme/yuxinliu/X-ASR}:${PYANNOTE_AUDIO_ROOT:-/root/shared-nvme/yuxinliu/pyannote-audio-4.0.7}/src:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/m_asr_matplotlib}"
mkdir -p "$MPLCONFIGDIR"

cd "$ROOT"
PYTHON_BIN="${PYANNOTE_GPU_PYTHON:-/root/.conda/envs/pyannote/bin/python}"
if [[ -x "$PYTHON_BIN" ]]; then
  exec "$PYTHON_BIN" -m m_asr.web_app --config configs/local.yaml "$@"
fi
exec uv run --offline --extra asr python -m m_asr.web_app --config configs/local.yaml "$@"
