# 多说话人流式 ASR 级联系统代码说明

## 目标

本项目把音频流切成带时间戳的稳定 chunk，再对每个 chunk 并行执行：

- X-ASR：chunk waveform -> text
- pyannote embedding：chunk waveform -> speaker embedding
- SpeakerRegistry：embedding -> 全局 `SPEAKER_N`

最终输出：

```text
[start - end] SPEAKER_N: text
```

## 关键模块

### `m_asr.types`

定义设计文档要求的数据结构：

- `AudioChunk`
- `AsrResult`
- `SpeakerResult`
- `TranscriptTurn`
- `PipelineEvent`

`PipelineEvent` 用于表达异步事件流：chunk finalized、partial、speaker、final、error。

### `m_asr.audio_buffer`

`AudioBuffer` 负责把输入音频统一成 16k mono float32，并维护全局采样点编号。

主要函数：

- `to_mono_float32`
- `linear_resample`
- `AudioBuffer.append`

### `m_asr.chunker`

`SpeechChunker` 是第一版 chunk 状态机，状态包括：

- `IDLE`
- `IN_SPEECH`
- `WAIT_SILENCE`

它按文档配置实现：

- 20ms frame
- speech onset / offset threshold
- 700ms trailing silence
- 0.8s 最短 chunk
- 12s 最长 chunk
- 左右 padding

当前默认 VAD 是能量门限，后续可把 `_speech_score` 替换为 pyannote segmentation 或神经 VAD。

### `m_asr.asr.x_asr_client`

`XAsrClient` 封装 X-ASR 调用。

当前实现只接受 `real` 模式：使用本地 X-ASR deployment 的 sherpa-onnx onnx 模型。依赖或模型不可用时直接报错。

默认模型目录：

```text
/root/shared-nvme/yuxinliu/X-ASR/X-ASR-zh-en/deployment/models/chunk-960ms-model
```

### `m_asr.diarization.pyannote_embedder`

`PyannoteSpeakerEmbedder` 封装 pyannote speaker embedding。

当前实现只接受 `real` 模式：加载本地 `pyannote-speaker-diarization-community-1/embedding`。依赖或模型不可用时直接报错。

### `m_asr.diarization.speaker_registry`

`SpeakerRegistry` 维护全局 speaker profile：

- `speaker_id`
- `centroid`
- `num_embeddings`
- `last_seen_time`

匹配逻辑：

1. embedding 做 L2 normalize
2. 和所有 centroid 算 cosine similarity
3. 超过 `same_speaker_threshold` 则匹配已有 speaker
4. 否则新建 `SPEAKER_N`
5. 满足时长和置信度要求才更新 centroid

### `m_asr.pipeline`

`StreamingCascadePipeline` 串联完整流程：

```text
AudioBuffer -> SpeechChunker -> XAsrClient
                          \-> PyannoteSpeakerEmbedder -> SpeakerRegistry
                          \-> TranscriptTurn
```

每个 chunk 内 ASR 和 speaker embedding 使用线程池并发执行。输出顺序保持为：

1. `chunk_finalized`
2. `partial`：`UNKNOWN: text`
3. `speaker`：`SPEAKER_N`
4. `final`：`SPEAKER_N: text`

## 运行方式

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

当前版本只运行真实 X-ASR 和真实 pyannote speaker embedding。若依赖暂时不可用，请先修复环境或模型路径。

默认运行 provider 是 CUDA。`StreamingCascadePipeline` 初始化时会检查 `torch.cuda.is_available()`；如果 CUDA 不可用，会打印 warning 并把 pyannote device 与 X-ASR provider 切换为 CPU 继续运行。

注意：pip 版 `sherpa-onnx` 可能只有 CPU provider。X-ASR 要真正跑 CUDA，需要安装或编译 GPU-enabled sherpa-onnx。
