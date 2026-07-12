import os

import torch
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook

# Community-1 open-source speaker diarization pipeline
pipeline = Pipeline.from_pretrained(
    "/root/shared-nvme/yuxinliu/pyannote-speaker-diarization-community-1",
    token=os.environ.get("HF_TOKEN"))

# send pipeline to GPU (when available)
pipeline.to(torch.device("cuda"))

# apply pretrained pipeline (with optional progress hook)
with ProgressHook() as hook:
    output = pipeline("/root/shared-nvme/yuxinliu/Track03.mp3", hook=hook)  # runs locally

# print the result
for turn, speaker in output.speaker_diarization:
    print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
# start=0.2s stop=1.5s speaker_0
# start=1.8s stop=3.9s speaker_1
# start=4.2s stop=5.7s speaker_0
# ...
