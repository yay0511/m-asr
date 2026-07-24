# 多说话人流式 ASR 级联系统代码说明

## 目标

当前项目实现两条入口：

```text
Web realtime:
Browser mic -> /ws/live -> X-ASR streaming partials
                         -> Silero endpoint -> pyannote embedding -> speaker id patch

CLI / upload:
WAV -> AudioBuffer -> Silero SpeechChunker -> chunk-level X-ASR
                                       \-> pyannote embedding -> SpeakerRegistry
```

Web 话筒路径的目标是“说多少出多少”：X-ASR streaming session 在音频帧进入时持续返回 partial 文本；一段语音被 Silero VAD 判定结束后，再生成 final 文本并异步补说话人。CLI 和文件上传路径仍然按 chunk 级级联处理。

## 关键模块

### `m_asr.types`

定义项目内部事件和数据结构：

- `AudioChunk`
- `AsrResult`
- `SpeakerResult`
- `TranscriptTurn`
- `PipelineEvent`

`PipelineEvent` 用于表达 `chunk_finalized`、`partial`、`speaker`、`final`、`error` 等事件。

### `m_asr.audio_buffer`

`AudioBuffer` 负责把输入音频统一成 16k mono float32，并维护全局采样点编号。

主要函数：

- `to_mono_float32`
- `linear_resample`
- `AudioBuffer.append`

### `m_asr.chunker`

`SpeechChunker` 是流式切句器，默认使用 Silero VAD。外部调用接口保持简单：

```text
accept(waveform) -> list[AudioChunk]
flush() -> list[AudioChunk]
```

Silero 路径使用 `SileroVadStream` 的 start/end 事件：

```text
IDLE -> TRIGGERED -> finalize -> IDLE
```

核心行为：

- 输入被拆成 `silero_window_samples=512` 的窗口
- Silero 检测 speech start/end
- start 时向左补 `left_padding_ms`
- end 时向右补 `right_padding_ms`
- 小于 `min_chunk_duration` 的 chunk 会被丢弃
- 超过 `max_chunk_duration` 会强制 finalize，避免多人连续说话粘成一个大 chunk

项目仍保留 `vad_provider: energy`，用于回滚和轻量单元测试；生产默认不是能量门限。

### `m_asr.vad.silero`

封装 `silero-vad` 的流式 VAD iterator。它负责加载 Silero 模型、按 16k 音频窗口计算语音事件，并把事件转换成项目内部使用的绝对采样点位置。

### `m_asr.asr.x_asr_client`

`XAsrClient` 封装真实 X-ASR 调用，底层使用本地 X-ASR deployment 的 `sherpa-onnx` 模型。

默认模型目录：

```text
/root/shared-nvme/yuxinliu/X-ASR/X-ASR-zh-en/deployment/models/chunk-960ms-model
```

两种调用方式：

- `recognize(chunk)`：为 CLI 和上传路径创建临时 stream，完成 chunk-level recognition
- `create_streaming_session()`：为 WebSocket 实时话筒创建长期 `XAsrStreamingSession`

`XAsrStreamingSession.accept_waveform()` 会持续向同一个 sherpa stream 追加音频，并在 recognizer ready 时 decode，返回当前 partial 文本。`finish()` 用于流结束时追加尾部 padding 并收尾。

当前实现只接受：

```yaml
asr:
  mode: real
```

依赖或模型不可用时会报错。

### `m_asr.diarization.pyannote_embedder`

`PyannoteSpeakerEmbedder` 封装真实 pyannote speaker embedding。

默认加载：

```text
/root/shared-nvme/yuxinliu/pyannote-speaker-diarization-community-1/embedding
```

识别时会把 `AudioChunk.waveform` reshape 成 pyannote 需要的张量格式并提取 embedding。小于 `speaker.min_embedding_duration` 的 chunk 会返回 `None`，避免把太短片段用于不可靠的 speaker 判断。

如果用户请求 CUDA 但当前 torch CUDA 不可用，或 CUDA 在模型加载/推理时触发 driver 错误，代码会打印 warning 并切到 CPU 重试。

### `m_asr.diarization.speaker_registry`

`SpeakerRegistry` 维护在线 speaker profile：

- `speaker_id`
- `centroid`
- `num_embeddings`
- `last_seen_time`

匹配流程：

1. embedding 做 L2 normalize
2. 和所有 speaker centroid 计算 cosine similarity
3. 优先检查上一个 speaker：达到 `last_speaker_threshold` 且和最优 speaker 差距不超过 `last_speaker_margin` 时，继续沿用上一个 speaker
4. 否则最佳相似度超过 `same_speaker_threshold` 时匹配已有 speaker
5. 如果不满足已有 speaker 匹配，再根据动态新建阈值判断是否允许创建新 speaker
6. 已有 speaker 且结果不确定时，默认分配给最佳已有 speaker，减少 `UNKNOWN`
7. 只有满足 `min_centroid_update_similarity`、`min_embedding_duration`、`min_update_confidence` 的高置信 chunk 才更新 centroid

流式开头使用更保守的新建 speaker 策略：`new_speaker_initial_max_similarity` 和 `min_new_speaker_duration_initial` 会随着 `new_speaker_warmup_seconds` 逐步过渡到 final 配置，降低刚开始 embedding 漂移导致频繁新建 speaker 的概率。

### `m_asr.pipeline`

`StreamingCascadePipeline` 串联 CLI 和文件上传路径：

```text
AudioBuffer -> SpeechChunker -> XAsrClient
                          \-> PyannoteSpeakerEmbedder -> SpeakerRegistry
                          \-> TranscriptTurn
```

每个 finalized chunk 内，ASR 和 speaker embedding 使用线程池并发执行。事件顺序：

1. `chunk_finalized`
2. `partial`：先输出 `UNKNOWN: text`
3. `speaker`：输出匹配到的 `SPEAKER_N`
4. `final`：输出带 speaker id 的最终 turn

这条路径是 chunk-level streaming cascade，不是 token-level realtime decoding。

### `m_asr.web_app`

`web_app.py` 提供 HTTP 页面和 WebSocket 服务：

- `/`：实时话筒页面
- `/upload`：文件上传页面
- `/health`：健康检查
- `/api/transcribe`：上传音频识别
- `/ws/live`：实时话筒 WebSocket

实时话筒路径由 `LiveWebSocketSession` 处理：

```text
browser 100ms PCM frame
-> XAsrStreamingSession.accept_waveform()
-> emit partial immediately
-> AudioBuffer.append()
-> SpeechChunker.accept()
-> on endpoint: final text + speaker future
-> emit speaker/final events
```

前端 Transcript 对事件的处理方式：

- `partial`：立即显示文字，speaker 位置暂时为空
- `speaker`：更新对应 chunk 的 speaker id
- `final`：固化该 chunk 文本
- 相邻同 speaker 且间隔不超过 1.2 秒的 turn 会合并显示

## 默认配置

默认配置文件：

```text
configs/local.yaml
```

运行设备：

```yaml
runtime:
  sample_rate: 16000
  device: cuda
  asr_provider: auto
  max_workers: 2
```

Silero VAD：

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

Speaker matching：

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

真实模型：

```yaml
asr:
  mode: real
speaker:
  mode: real
```

## 运行方式

安装或同步依赖：

```bash
cd /root/shared-nvme/yuxinliu/m_asr
export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
uv sync --extra asr
```

环境检查：

```bash
cd /root/shared-nvme/yuxinliu/m_asr
uv run --offline --extra asr python scripts/check_env.py
uv run --offline --extra asr python scripts/check_cuda.py
```

处理 WAV：

```bash
cd /root/shared-nvme/yuxinliu/m_asr
bash scripts/run_dev.sh --audio /path/to/audio.wav
```

启动 Web 页面：

```bash
cd /root/shared-nvme/yuxinliu/m_asr
bash scripts/run_web.sh --host 0.0.0.0 --port 8000
```

## CUDA 说明

默认 pyannote/torch device 是 CUDA，X-ASR provider 是 auto。GPU 网页推荐通过 `scripts/run_web_gpu.sh` 启动；该脚本使用 `/root/.conda/envs/pyannote/bin/python` 并在启动前强制检查 `torch.cuda.is_available()`。

启动 GPU 网页：

```bash
bash scripts/run_web_gpu.sh --host 0.0.0.0 --port 8001
```

`StreamingCascadePipeline` 初始化时会检查 `torch.cuda.is_available()`。如果使用普通 `run_web.sh` 且 CUDA 不可用，会打印 warning 并把 pyannote device 切换为 CPU 继续运行；如果使用 `run_web_gpu.sh`，CUDA 不可用会在启动前直接失败。

X-ASR 的 CUDA 能力取决于 `sherpa-onnx` 是否包含 ONNX Runtime CUDA provider。pip 版 `sherpa-onnx` 常见为 CPU-only，此时 `asr_provider: auto` 会显式选择 CPU provider，避免 C++ 层反复 fallback。X-ASR 要真正跑 CUDA，需要安装或编译 GPU-enabled sherpa-onnx；安装后可用 `M_ASR_FORCE_SHERPA_CUDA=1` 强制验证。
