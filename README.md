# m_asr

多说话人流式 ASR 级联系统第一版实现。

目标流程：

```text
Audio Stream
-> AudioBuffer
-> SpeechChunker
-> X-ASR chunk recognition
-> pyannote speaker embedding
-> SpeakerRegistry
-> [start - end] SPEAKER_N: text
```

## 当前能力

- 统一音频为 16k mono float32，并维护全局时间轴
- 基于 VAD/静音思想的动态 speech chunk 状态机
- chunk 级 X-ASR 适配层
- pyannote speaker embedding 适配层
- 增量 speaker centroid 匹配
- 事件流输出：`chunk_finalized`、`partial`、`speaker`、`final`
- 命令行演示：支持合成音频和 WAV 文件输入

## 快速运行

```bash
cd /root/shared-nvme/yuxinliu/m_asr
export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
uv sync --extra asr
bash scripts/run_dev.sh --audio /path/to/audio.wav
```

检查环境：

```bash
cd /root/shared-nvme/yuxinliu/m_asr
uv run --offline --extra asr python scripts/check_env.py
uv run --offline --extra asr python scripts/check_cuda.py
```

真实 X-ASR 依赖 `sherpa-onnx`。如果环境检查显示 `MISSING module sherpa_onnx`，需要先安装 ASR 依赖：

```bash
export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
uv sync --extra asr
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

## 真实模型

`configs/local.yaml` 默认强制使用真实后端：

```yaml
asr:
  mode: real
speaker:
  mode: real
```

真实依赖或模型路径不可用时，程序会直接报错。CUDA 不可用时会打印 warning 并自动切换到 CPU。

```bash
bash scripts/run_dev.sh --audio /path/to/audio.wav
```

默认配置优先请求 CUDA：

```yaml
runtime:
  device: cuda
  asr_provider: cuda
```

如果 `torch.cuda.is_available()` 为 False，程序会打印 warning，然后把 pyannote 和 X-ASR provider 切到 CPU 继续运行。当前 `sherpa-onnx` pip wheel 也可能只有 CPU provider；X-ASR 要真正跑 CUDA，需要安装或编译支持 GPU provider 的 sherpa-onnx。

## 文档

主要代码和关键函数说明见：

```text
docs/ARCHITECTURE.md
```
