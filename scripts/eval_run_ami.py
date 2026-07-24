#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from m_asr.config import load_config
from m_asr.main import read_wav
from m_asr.pipeline import StreamingCascadePipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Run m_asr on AMI Array1-01 wav files.")
    parser.add_argument("--manifest", default="eval/ami_array1_01/refs/manifest.jsonl")
    parser.add_argument("--output-dir", default="eval/ami_array1_01/pred")
    parser.add_argument(
        "--per-meeting-dir",
        default="",
        help="directory containing per-meeting *.pred.jsonl files; defaults to OUTPUT_DIR/meetings",
    )
    parser.add_argument("--config", default="configs/local.yaml")
    parser.add_argument("--meetings", default="", help="comma-separated meeting ids to run")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--frame-seconds", type=float, default=0.2)
    parser.add_argument("--start-seconds", type=float, default=0.0, help="start offset inside each wav")
    parser.add_argument("--max-seconds", type=float, default=0.0, help="process at most this many seconds per meeting")
    parser.add_argument(
        "--warmup-seconds",
        type=float,
        default=0.0,
        help=(
            "feed this many seconds before --start-seconds to warm up streaming state, "
            "but write predictions only inside the scoring window"
        ),
    )
    parser.add_argument(
        "--boundary-policy",
        choices=("drop", "clip"),
        default="drop",
        help=(
            "how to handle hypothesis turns crossing the scoring window boundary. "
            "'drop' avoids scoring text that cannot be trimmed at exact word boundaries; "
            "'clip' keeps the old time-clipping behavior."
        ),
    )
    parser.add_argument("--progress-seconds", type=float, default=30.0, help="print audio progress interval")
    parser.add_argument("--resume", action="store_true", help="skip meetings whose per-meeting jsonl already exists")
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="rebuild pred.jsonl/pred.rttm from existing per-meeting predictions and exit",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    per_meeting_dir = Path(args.per_meeting_dir) if args.per_meeting_dir else output_dir / "meetings"
    per_meeting_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_manifest(Path(args.manifest), args.meetings)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit("no manifest rows selected")

    all_rows: list[dict[str, object]] = []
    pred_jsonl = output_dir / "pred.jsonl"
    pred_rttm = output_dir / "pred.rttm"

    if args.collect_only:
        all_rows = _collect_existing_predictions(rows, per_meeting_dir)
        _write_outputs(all_rows, pred_jsonl, pred_rttm)
        print(f"[eval] collected meetings={len({row['audio_id'] for row in all_rows})} turns={len(all_rows)}")
        print(f"[eval] pred_jsonl={pred_jsonl}")
        print(f"[eval] pred_rttm={pred_rttm}")
        return 0

    for index, row in enumerate(rows, start=1):
        audio_id = str(row["audio_id"])
        meeting_jsonl = per_meeting_dir / f"{audio_id}.pred.jsonl"
        if args.resume and meeting_jsonl.exists():
            turns = _read_jsonl(meeting_jsonl)
            all_rows.extend(turns)
            print(f"[eval] {index}/{len(rows)} {audio_id}: resume {len(turns)} turns")
            _write_outputs(all_rows, pred_jsonl, pred_rttm)
            continue

        audio_path = Path(str(row["audio_path"]))
        print(f"[eval] {index}/{len(rows)} {audio_id}: {audio_path}", flush=True)
        turns = _run_one(
            audio_id,
            audio_path,
            args.config,
            args.frame_seconds,
            args.start_seconds,
            args.max_seconds,
            args.warmup_seconds,
            args.boundary_policy,
            args.progress_seconds,
        )
        with meeting_jsonl.open("w", encoding="utf-8") as f:
            for turn in turns:
                f.write(json.dumps(turn, ensure_ascii=False) + "\n")
        all_rows.extend(turns)
        print(f"[eval] {audio_id}: turns={len(turns)}", flush=True)
        _write_outputs(all_rows, pred_jsonl, pred_rttm)

    _write_outputs(all_rows, pred_jsonl, pred_rttm)

    print(f"[eval] pred_jsonl={pred_jsonl}")
    print(f"[eval] pred_rttm={pred_rttm}")
    return 0


def _collect_existing_predictions(
    manifest_rows: list[dict[str, object]],
    per_meeting_dir: Path,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    missing: list[str] = []
    for manifest_row in manifest_rows:
        audio_id = str(manifest_row["audio_id"])
        meeting_jsonl = per_meeting_dir / f"{audio_id}.pred.jsonl"
        if not meeting_jsonl.exists():
            missing.append(audio_id)
            continue
        rows.extend(_read_jsonl(meeting_jsonl))
    if missing:
        preview = ", ".join(missing[:10])
        suffix = "" if len(missing) <= 10 else f", ... (+{len(missing) - 10})"
        raise SystemExit(
            "missing per-meeting predictions for "
            f"{len(missing)} manifest meetings: {preview}{suffix}"
        )
    return rows


def _write_outputs(
    rows: list[dict[str, object]],
    pred_jsonl: Path,
    pred_rttm: Path,
) -> None:
    pred_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with pred_jsonl.open("w", encoding="utf-8") as f:
        for turn in rows:
            f.write(json.dumps(turn, ensure_ascii=False) + "\n")

    with pred_rttm.open("w", encoding="utf-8") as f:
        for turn in rows:
            speaker = str(turn.get("speaker", "UNKNOWN"))
            start = float(turn["start"])
            end = float(turn["end"])
            if end <= start:
                continue
            duration = end - start
            f.write(
                f"SPEAKER {turn['audio_id']} 1 {start:.3f} {duration:.3f} "
                f"<NA> <NA> {speaker} <NA> <NA>\n"
            )


def _load_manifest(path: Path, meetings_arg: str) -> list[dict[str, object]]:
    selected = {item.strip() for item in meetings_arg.split(",") if item.strip()}
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if not selected or row.get("audio_id") in selected:
                rows.append(row)
    return rows


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _run_one(
    audio_id: str,
    audio_path: Path,
    config_path: str,
    frame_seconds: float,
    start_seconds: float,
    max_seconds: float,
    warmup_seconds: float,
    boundary_policy: str,
    progress_seconds: float,
) -> list[dict[str, object]]:
    config = load_config(config_path)
    waveform, sample_rate = read_wav(audio_path)
    waveform, decode_start, score_start, score_end, duration = _slice_waveform(
        waveform,
        sample_rate,
        start_seconds,
        max_seconds,
        warmup_seconds,
    )
    print(
        f"[eval] {audio_id}: loaded {duration:.1f}s "
        f"from offset {decode_start:.1f}s sr={sample_rate}; "
        f"scoring {score_start:.1f}-{score_end:.1f}s warmup={max(0.0, score_start - decode_start):.1f}s",
        flush=True,
    )

    pipeline = StreamingCascadePipeline(config)
    _process_with_progress(
        pipeline,
        audio_id,
        waveform,
        sample_rate,
        frame_seconds=frame_seconds,
        progress_seconds=progress_seconds,
    )

    rows: list[dict[str, object]] = []
    for turn in pipeline.transcript:
        start = turn.start + decode_start
        end = turn.end + decode_start
        if end <= score_start or start >= score_end:
            continue
        if boundary_policy == "drop" and (start < score_start or end > score_end):
            continue
        if boundary_policy == "clip":
            start = max(start, score_start)
            end = min(end, score_end)
        if end <= start:
            continue
        rows.append(
            {
                "audio_id": audio_id,
                "start": round(start, 3),
                "end": round(end, 3),
                "speaker": turn.speaker_id,
                "text": turn.text,
                "confidence": round(float(turn.confidence), 6),
            }
        )
    return rows


def _slice_waveform(
    waveform: np.ndarray,
    sample_rate: int,
    start_seconds: float,
    max_seconds: float,
    warmup_seconds: float,
) -> tuple[np.ndarray, float, float, float, float]:
    start_seconds = max(0.0, float(start_seconds))
    warmup_seconds = max(0.0, float(warmup_seconds))
    score_start_sample = min(len(waveform), int(round(start_seconds * sample_rate)))
    if max_seconds > 0:
        score_end_sample = min(len(waveform), score_start_sample + int(round(max_seconds * sample_rate)))
    else:
        score_end_sample = len(waveform)
    warmup_samples = int(round(warmup_seconds * sample_rate))
    decode_start_sample = max(0, score_start_sample - warmup_samples)
    sliced = np.asarray(waveform[decode_start_sample:score_end_sample], dtype=np.float32)
    decode_start = decode_start_sample / sample_rate
    score_start = score_start_sample / sample_rate
    score_end = score_end_sample / sample_rate
    return sliced, decode_start, score_start, score_end, sliced.size / sample_rate


def _process_with_progress(
    pipeline: StreamingCascadePipeline,
    audio_id: str,
    waveform: np.ndarray,
    sample_rate: int,
    frame_seconds: float,
    progress_seconds: float,
) -> None:
    frame_samples = max(1, int(round(sample_rate * frame_seconds)))
    total_seconds = len(waveform) / sample_rate if sample_rate else 0.0
    next_progress = max(progress_seconds, frame_seconds)
    finalized_chunks = 0
    final_turns = 0

    with ThreadPoolExecutor(max_workers=pipeline.config.runtime.max_workers) as executor:
        for offset in range(0, len(waveform), frame_samples):
            frame = waveform[offset : offset + frame_samples]
            audio_time = min(total_seconds, (offset + len(frame)) / sample_rate)
            if progress_seconds > 0 and audio_time >= next_progress:
                print(
                    f"[eval] {audio_id}: audio {audio_time:.1f}/{total_seconds:.1f}s "
                    f"chunks={finalized_chunks} turns={final_turns}",
                    flush=True,
                )
                while next_progress <= audio_time:
                    next_progress += progress_seconds

            write = pipeline.audio_buffer.append(frame, sample_rate)
            for chunk in pipeline.chunker.accept(write.waveform):
                finalized_chunks += 1
                print(
                    f"[eval] {audio_id}: chunk #{chunk.chunk_id} "
                    f"{chunk.start:.2f}-{chunk.end:.2f}s",
                    flush=True,
                )
                for event in pipeline._process_chunk(chunk, executor):
                    if event.event_type == "final":
                        final_turns += 1
                    elif event.event_type == "error":
                        print(f"[warn] {audio_id} #{event.chunk_id}: {event.message}", file=sys.stderr)

        for chunk in pipeline.chunker.flush():
            finalized_chunks += 1
            print(
                f"[eval] {audio_id}: chunk #{chunk.chunk_id} "
                f"{chunk.start:.2f}-{chunk.end:.2f}s",
                flush=True,
            )
            for event in pipeline._process_chunk(chunk, executor):
                if event.event_type == "final":
                    final_turns += 1
                elif event.event_type == "error":
                    print(f"[warn] {audio_id} #{event.chunk_id}: {event.message}", file=sys.stderr)

    print(f"[eval] {audio_id}: done chunks={finalized_chunks} turns={final_turns}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
