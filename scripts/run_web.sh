#!/usr/bin/env bash
set -euo pipefail

ROOT="${M_ASR_ROOT:-/root/shared-nvme/yuxinliu/m_asr}"
export PYTHONPATH="$ROOT/src:${X_ASR_ROOT:-/root/shared-nvme/yuxinliu/X-ASR}:${PYANNOTE_AUDIO_ROOT:-/root/shared-nvme/yuxinliu/pyannote-audio-4.0.7}/src:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/m_asr_matplotlib}"
mkdir -p "$MPLCONFIGDIR"

cd "$ROOT"
uv run --offline --extra asr python -m m_asr.web_app --config configs/local.yaml "$@"
