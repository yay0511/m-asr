#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


NITE = "{http://nite.sourceforge.net/}"
ID_RE = re.compile(r"id\(([^)]+)\)")


@dataclass(slots=True)
class AmiTurn:
    audio_id: str
    start: float
    end: float
    speaker: str
    text: str

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert AMI manual XML annotations to RTTM/JSONL references.")
    parser.add_argument("--data-root", default="/root/shared-nvme/yuxinliu/data_a", help="AMI data root")
    parser.add_argument("--output-dir", default="eval/ami_array1_01/refs", help="output directory")
    parser.add_argument("--meetings", default="", help="comma-separated meeting ids to include")
    parser.add_argument("--meeting-list", default="", help="file containing one meeting id per line")
    parser.add_argument("--limit", type=int, default=0, help="limit number of meetings after filtering")
    parser.add_argument("--start-seconds", type=float, default=0.0, help="start offset to keep in each meeting")
    parser.add_argument("--max-seconds", type=float, default=0.0, help="keep at most this many seconds per meeting")
    parser.add_argument(
        "--der-text-only",
        action="store_true",
        help="write RTTM only for segments that contain lexical words",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    manual_root = data_root / "ami_manual_1.6.1"
    audio_root = _find_array_audio_root(data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected = _selected_meetings(args.meetings, args.meeting_list)
    meetings = _discover_meetings(audio_root, selected)
    if args.limit:
        meetings = meetings[: args.limit]
    if not meetings:
        raise SystemExit("no meetings found; check --data-root/--meetings")

    all_turns: list[AmiTurn] = []
    manifest_rows: list[dict[str, str]] = []
    ref_jsonl = output_dir / "ref.jsonl"
    ref_rttm = output_dir / "ref.rttm"
    manifest = output_dir / "manifest.jsonl"
    meetings_txt = output_dir / "meetings.txt"

    for meeting in meetings:
        audio_path = audio_root / meeting / "audio" / f"{meeting}.Array1-01.wav"
        turns = _crop_turns(
            list(_load_meeting_turns(manual_root, meeting)),
            start_seconds=args.start_seconds,
            max_seconds=args.max_seconds,
        )
        if not turns:
            print(f"[warn] no reference turns for {meeting}", file=sys.stderr)
            continue
        all_turns.extend(turns)
        manifest_rows.append(
            {
                "audio_id": meeting,
                "audio_path": str(audio_path),
                "ref_jsonl": str(ref_jsonl),
                "ref_rttm": str(ref_rttm),
            }
        )

    with ref_jsonl.open("w", encoding="utf-8") as f:
        for turn in all_turns:
            f.write(json.dumps(_turn_to_json(turn), ensure_ascii=False) + "\n")

    with ref_rttm.open("w", encoding="utf-8") as f:
        for turn in all_turns:
            if args.der_text_only and not turn.text.strip():
                continue
            if turn.duration <= 0:
                continue
            f.write(_turn_to_rttm(turn) + "\n")

    with manifest.open("w", encoding="utf-8") as f:
        for row in manifest_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with meetings_txt.open("w", encoding="utf-8") as f:
        for meeting in meetings:
            f.write(f"{meeting}\n")

    print(f"[ami] meetings={len(manifest_rows)} turns={len(all_turns)}")
    print(f"[ami] ref_jsonl={ref_jsonl}")
    print(f"[ami] ref_rttm={ref_rttm}")
    print(f"[ami] manifest={manifest}")
    return 0


def _find_array_audio_root(data_root: Path) -> Path:
    candidates = [data_root / "Arrat1-01", data_root / "Array1-01"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"cannot find Arrat1-01/Array1-01 under {data_root}")


def _selected_meetings(meetings_arg: str, meeting_list: str) -> set[str] | None:
    selected: set[str] = set()
    if meetings_arg.strip():
        selected.update(item.strip() for item in meetings_arg.split(",") if item.strip())
    if meeting_list:
        with Path(meeting_list).open("r", encoding="utf-8") as f:
            selected.update(line.strip() for line in f if line.strip() and not line.startswith("#"))
    return selected or None


def _discover_meetings(audio_root: Path, selected: set[str] | None) -> list[str]:
    meetings: list[str] = []
    for wav in sorted(audio_root.glob("*/audio/*.Array1-01.wav")):
        meeting = wav.name.replace(".Array1-01.wav", "")
        if selected is None or meeting in selected:
            meetings.append(meeting)
    return meetings


def _load_meeting_turns(manual_root: Path, meeting: str) -> Iterable[AmiTurn]:
    segment_dir = manual_root / "segments"
    word_dir = manual_root / "words"
    segment_files = sorted(segment_dir.glob(f"{meeting}.*.segments.xml"))
    for segment_file in segment_files:
        speaker_suffix = _speaker_suffix(segment_file.name, meeting, ".segments.xml")
        speaker = f"{meeting}.{speaker_suffix}"
        word_file = word_dir / f"{meeting}.{speaker_suffix}.words.xml"
        words = _load_words(word_file)
        for start, end, word_ids in _load_segments(segment_file):
            text = _words_for_segment(words, word_ids)
            yield AmiTurn(audio_id=meeting, start=start, end=end, speaker=speaker, text=text)


def _crop_turns(turns: list[AmiTurn], start_seconds: float, max_seconds: float) -> list[AmiTurn]:
    start = max(0.0, float(start_seconds))
    end = float("inf") if max_seconds <= 0 else start + float(max_seconds)
    cropped: list[AmiTurn] = []
    for turn in turns:
        if turn.end <= start or turn.start >= end:
            continue
        cropped.append(
            AmiTurn(
                audio_id=turn.audio_id,
                start=max(turn.start, start),
                end=min(turn.end, end),
                speaker=turn.speaker,
                text=turn.text,
            )
        )
    return cropped


def _speaker_suffix(filename: str, meeting: str, suffix: str) -> str:
    prefix = f"{meeting}."
    if not filename.startswith(prefix) or not filename.endswith(suffix):
        raise ValueError(f"unexpected AMI file name: {filename}")
    return filename[len(prefix) : -len(suffix)]


def _load_words(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    root = ET.parse(path).getroot()
    words: dict[str, str] = {}
    for elem in root:
        if _local_name(elem.tag) != "w":
            continue
        word_id = elem.attrib.get(f"{NITE}id") or elem.attrib.get("nite:id")
        if not word_id:
            continue
        words[word_id] = elem.text or ""
    return words


def _load_segments(path: Path) -> Iterable[tuple[float, float, list[str]]]:
    root = ET.parse(path).getroot()
    for elem in root:
        if _local_name(elem.tag) != "segment":
            continue
        start = _float_attr(elem, "transcriber_start", "starttime")
        end = _float_attr(elem, "transcriber_end", "endtime")
        word_ids: list[str] = []
        for child in elem:
            href = child.attrib.get("href", "")
            word_ids.extend(_ids_from_href(href))
        if start is not None and end is not None and end > start:
            yield start, end, word_ids


def _float_attr(elem: ET.Element, *names: str) -> float | None:
    for name in names:
        value = elem.attrib.get(name)
        if value is not None:
            return float(value)
    return None


def _ids_from_href(href: str) -> list[str]:
    ids = ID_RE.findall(href)
    if len(ids) != 2:
        return ids
    start, end = ids
    prefix, start_index = _split_word_id(start)
    end_prefix, end_index = _split_word_id(end)
    if prefix != end_prefix or start_index is None or end_index is None:
        return ids
    step = 1 if end_index >= start_index else -1
    return [f"{prefix}{index}" for index in range(start_index, end_index + step, step)]


def _split_word_id(word_id: str) -> tuple[str, int | None]:
    match = re.match(r"^(.*?)(\d+)$", word_id)
    if not match:
        return word_id, None
    return match.group(1), int(match.group(2))


def _words_for_segment(words: dict[str, str], word_ids: list[str]) -> str:
    tokens = [words[word_id] for word_id in word_ids if word_id in words and words[word_id]]
    text = ""
    for token in tokens:
        if not text:
            text = token
        elif token in {".", ",", "?", "!", ":", ";", "%"}:
            text += token
        else:
            text += " " + token
    return text


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _turn_to_json(turn: AmiTurn) -> dict[str, object]:
    return {
        "audio_id": turn.audio_id,
        "start": round(turn.start, 3),
        "end": round(turn.end, 3),
        "speaker": turn.speaker,
        "text": turn.text,
    }


def _turn_to_rttm(turn: AmiTurn) -> str:
    return (
        f"SPEAKER {turn.audio_id} 1 {turn.start:.3f} {turn.duration:.3f} "
        f"<NA> <NA> {turn.speaker} <NA> <NA>"
    )


if __name__ == "__main__":
    raise SystemExit(main())
