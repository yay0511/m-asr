# patches

当前第一版没有修改外部目录，因此没有必须应用的 patch。

外部目录保持只读引用：

- `/root/shared-nvme/yuxinliu/X-ASR`
- `/root/shared-nvme/yuxinliu/pyannote-audio-4.0.7`
- `/root/shared-nvme/yuxinliu/pyannote-speaker-diarization-community-1`

如果后续必须修改外部依赖，请把 diff 保存到本目录，并记录：

- patch 文件名
- 作用
- 适用的外部仓库路径
- 适用的 commit/tag
- 应用方式
- 是否必须
