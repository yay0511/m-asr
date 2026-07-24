#!/usr/bin/env bash
set -euo pipefail

ROOT="${M_ASR_ROOT:-/root/shared-nvme/yuxinliu/m_asr}"
PYTHON_BIN="${M_ASR_EVAL_PYTHON:-/root/.conda/envs/pyannote/bin/python}"
DATA_ROOT="${AMI_DATA_ROOT:-/root/shared-nvme/yuxinliu/data_a}"
OUTPUT_DIR="${M_ASR_EVAL_OUTPUT_DIR:-eval/ami_array1_01_quick}"
CONFIG="${M_ASR_EVAL_CONFIG:-configs/local.yaml}"
MEETINGS="${M_ASR_EVAL_MEETINGS:-EN2001a}"
START_SECONDS="${M_ASR_EVAL_START_SECONDS:-0}"
MAX_SECONDS="${M_ASR_EVAL_MAX_SECONDS:-300}"
WARMUP_SECONDS="${M_ASR_EVAL_WARMUP_SECONDS:-0}"
BOUNDARY_POLICY="${M_ASR_EVAL_BOUNDARY_POLICY:-drop}"
FRAME_SECONDS="${M_ASR_EVAL_FRAME_SECONDS:-0.2}"
PROGRESS_SECONDS="${M_ASR_EVAL_PROGRESS_SECONDS:-30}"
METRICS="${M_ASR_EVAL_METRICS:-all}"
RESUME="${M_ASR_EVAL_RESUME:-0}"

export PYTHONPATH="$ROOT/src:${X_ASR_ROOT:-/root/shared-nvme/yuxinliu/X-ASR}:${PYANNOTE_AUDIO_ROOT:-/root/shared-nvme/yuxinliu/pyannote-audio-4.0.7}/src:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/m_asr_matplotlib}"
mkdir -p "$MPLCONFIGDIR"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[eval] missing python: $PYTHON_BIN" >&2
  exit 2
fi

PYTHON_SITE="$("$PYTHON_BIN" - <<'PY'
import site

print(site.getsitepackages()[0])
PY
)"
NVIDIA_LIB_ROOT="$PYTHON_SITE/nvidia"
for lib_path in \
  "$NVIDIA_LIB_ROOT/cublas/lib" \
  "$NVIDIA_LIB_ROOT/cudnn/lib" \
  "$NVIDIA_LIB_ROOT/cufft/lib" \
  "$NVIDIA_LIB_ROOT/curand/lib" \
  "$NVIDIA_LIB_ROOT/cuda_runtime/lib" \
  "$NVIDIA_LIB_ROOT/cuda_nvrtc/lib"; do
  if [[ -d "$lib_path" ]]; then
    export LD_LIBRARY_PATH="$lib_path:${LD_LIBRARY_PATH:-}"
  fi
done

cd "$ROOT"

echo "[eval] output=$OUTPUT_DIR"
echo "[eval] meetings=$MEETINGS start=${START_SECONDS}s max=${MAX_SECONDS}s warmup=${WARMUP_SECONDS}s boundary=${BOUNDARY_POLICY}"
echo "[eval] metrics=$METRICS"
echo "[eval] resume=$RESUME"

"$PYTHON_BIN" scripts/ami_prepare_refs.py \
  --data-root "$DATA_ROOT" \
  --output-dir "$OUTPUT_DIR/refs" \
  --meetings "$MEETINGS" \
  --start-seconds "$START_SECONDS" \
  --max-seconds "$MAX_SECONDS"

RUN_ARGS=(
  scripts/eval_run_ami.py
  --manifest "$OUTPUT_DIR/refs/manifest.jsonl" \
  --output-dir "$OUTPUT_DIR/pred" \
  --config "$CONFIG" \
  --meetings "$MEETINGS" \
  --start-seconds "$START_SECONDS" \
  --max-seconds "$MAX_SECONDS" \
  --warmup-seconds "$WARMUP_SECONDS" \
  --boundary-policy "$BOUNDARY_POLICY" \
  --frame-seconds "$FRAME_SECONDS" \
  --progress-seconds "$PROGRESS_SECONDS"
)
if [[ "$RESUME" == "1" || "$RESUME" == "true" || "$RESUME" == "yes" || "$RESUME" == "on" ]]; then
  RUN_ARGS+=(--resume)
fi

"$PYTHON_BIN" "${RUN_ARGS[@]}"

"$PYTHON_BIN" scripts/eval_metrics.py \
  --ref-jsonl "$OUTPUT_DIR/refs/ref.jsonl" \
  --hyp-jsonl "$OUTPUT_DIR/pred/pred.jsonl" \
  --ref-rttm "$OUTPUT_DIR/refs/ref.rttm" \
  --hyp-rttm "$OUTPUT_DIR/pred/pred.rttm" \
  --output-json "$OUTPUT_DIR/metrics.json" \
  --metrics "$METRICS" \
  --audio-scope strict
