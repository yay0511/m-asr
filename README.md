# m_asr

多说话人流式 ASR 级联系统。当前版本保留真实 X-ASR 和真实 pyannote speaker embedding，不再提供 mock 主流程。

## 当前流程

实时话筒 Web 首页 `/`：

```text
Browser microphone
-> WebSocket /ws/live
-> 16k mono PCM frames
-> X-ASR streaming session -> partial text immediately
-> AudioBuffer
-> Silero VAD SpeechChunker endpoint
-> final text
-> pyannote speaker embedding
-> SpeakerRegistry
-> patch speaker id in Transcript
```

文件上传页面 `/upload` 和命令行 WAV 处理：

```text
WAV / waveform
-> AudioBuffer
-> Silero VAD SpeechChunker
-> chunk-level X-ASR recognition
-> pyannote speaker embedding
-> SpeakerRegistry
-> [start - end] SPEAKER_N: text
```

也就是说，Web 话筒路径会边收音频边 decode 并持续显示 partial 文本；Silero VAD 主要负责检测 endpoint/finalize。上传和 CLI 仍是 chunk-level cascade。

## 当前能力

- 统一音频为 16k mono float32，并维护全局时间轴
- WebSocket 实时话筒输入，浏览器端重采样为 16k PCM 发送
- X-ASR 长期 streaming session，音频帧进入后立即产出 partial 文本
- 基于 Silero VAD 的流式 SpeechChunker，用于切句和 endpoint
- 真实 pyannote speaker embedding
- 在线 SpeakerRegistry：动态新建 speaker 阈值、last-speaker bias、低置信分配但不更新 centroid
- Web Transcript：先显示文字，speaker 确定后补 speaker id；相邻同 speaker 片段会合并显示
- 文件上传和命令行 WAV 演示

## 快速运行

```bash
cd /root/shared-nvme/yuxinliu/m_asr
export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
uv sync --extra asr
bash scripts/run_web.sh --host 0.0.0.0 --port 8000
```

浏览器打开：

```text
http://localhost:8000
```

首页是实时话筒模式。文件上传页面：

```text
http://localhost:8000/upload
```

命令行处理 WAV：

```bash
cd /root/shared-nvme/yuxinliu/m_asr
bash scripts/run_dev.sh --audio /path/to/audio.wav
```

检查环境：

```bash
cd /root/shared-nvme/yuxinliu/m_asr
uv run --offline --extra asr python scripts/check_env.py
uv run --offline --extra asr python scripts/check_cuda.py
```

## 配置

默认配置文件：

```text
configs/local.yaml
```

关键外部路径：

```text
/root/shared-nvme/yuxinliu/X-ASR
/root/shared-nvme/yuxinliu/pyannote-audio-4.0.7
/root/shared-nvme/yuxinliu/pyannote-speaker-diarization-community-1
```

外部目录只读引用，不复制进本项目，也不直接修改。

真实后端配置：

```yaml
asr:
  mode: real
speaker:
  mode: real
```

真实依赖或模型路径不可用时，程序会直接报错。

默认运行设备：

```yaml
runtime:
  device: cpu
  asr_provider: auto
```

`device: cpu` 会让 pyannote/torch 默认走 CPU，避免 NVIDIA driver 不匹配时触发 CUDA RuntimeError。需要手动尝试 GPU 时，可以临时设置 `M_ASR_DEVICE=cuda`。

`asr_provider: auto` 表示 X-ASR 只有在当前 `sherpa-onnx` 安装能看到 CUDA provider 时才传 `cuda`，否则显式传 `cpu`，避免 C++ 层反复 fallback。当前 pip 版 `sherpa-onnx` 常见为 CPU-only；X-ASR 要真正跑 CUDA，需要安装或编译 GPU-enabled sherpa-onnx，安装后可用 `M_ASR_FORCE_SHERPA_CUDA=1` 强制验证。

如果 CUDA 在 pyannote embedding 模型加载或推理时才失败，程序会打印 warning、把 pyannote 切到 CPU，并重试当前 chunk，不会把 NVIDIA driver 错误返回到 Web 页面作为识别失败。

## Silero VAD

默认切句使用 Silero VAD：

```yaml
chunker:
  vad_provider: silero
  frame_ms: 32
  silero_threshold: 0.35
  silero_min_silence_ms: 200
  silero_speech_pad_ms: 0
  silero_window_samples: 512
  min_chunk_duration: 0.35
  max_chunk_duration: 1.8
  left_padding_ms: 200
  right_padding_ms: 80
```

`silero_threshold` 越低越容易判为人声，能减少漏检，但更容易把相邻句子合并；`silero_min_silence_ms` 决定连续非语音多久后结束一个 chunk；`max_chunk_duration` 用来防止多人连续说话时长时间粘成一个大 chunk。

需要临时回到旧能量阈值方式：

```bash
M_ASR_VAD_PROVIDER=energy bash scripts/run_web.sh --host 0.0.0.0 --port 8000
```

## 说话人匹配

默认 speaker 配置：

```yaml
speaker:
  same_speaker_threshold: 0.68
  last_speaker_threshold: 0.56
  last_speaker_margin: 0.08
  new_speaker_initial_max_similarity: 0.08
  new_speaker_final_max_similarity: 0.28
  new_speaker_warmup_seconds: 12.0
  min_new_speaker_duration_initial: 2.0
  min_new_speaker_duration_final: 1.5
  min_centroid_update_similarity: 0.78
  centroid_update_alpha: 0.85
  min_embedding_duration: 0.7
  min_update_confidence: 0.6
  assign_uncertain_to_best: true
```

流式开始阶段 embedding 容易漂移，所以新建 speaker 更保守。系统会随着 `new_speaker_warmup_seconds` 逐步恢复到正常新建门槛；同时，上一个 chunk 的 speaker 有独立的较低匹配阈值和 margin 保护，用来增加 speaker 切换难度。

已有 speaker 时，不确定 embedding 会先分配给最相似 speaker，减少 `UNKNOWN`。低置信匹配只影响显示结果，不会更新 centroid；只有相似度超过 `min_centroid_update_similarity` 的高质量 chunk 才会更新 speaker centroid。

如果多人说话仍被合并，优先把 `max_chunk_duration` 调小到 `1.2` 或把 `silero_min_silence_ms` 调小到 `150`；如果切得太碎，则反向调大。

## 文档

主要代码和关键函数说明见：

```text
docs/ARCHITECTURE.md
```
