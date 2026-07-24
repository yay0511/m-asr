#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import math
import re
import string
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

UnitSequence = str | tuple[str, ...]


@dataclass(slots=True)
class Turn:
    audio_id: str
    start: float
    end: float
    speaker: str
    text: str = ""

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute DER, cpCER/tcpCER and cpWER/tcpWER for m_asr outputs.")
    parser.add_argument("--ref-jsonl", default="eval/ami_array1_01/refs/ref.jsonl")
    parser.add_argument("--hyp-jsonl", default="eval/ami_array1_01/pred/pred.jsonl")
    parser.add_argument("--ref-rttm", default="eval/ami_array1_01/refs/ref.rttm")
    parser.add_argument("--hyp-rttm", default="eval/ami_array1_01/pred/pred.rttm")
    parser.add_argument("--output-json", default="eval/ami_array1_01/metrics.json")
    parser.add_argument("--collar", type=float, default=0.25)
    parser.add_argument("--frame-step", type=float, default=0.01)
    parser.add_argument("--skip-overlap", action="store_true")
    parser.add_argument("--tcp-window", type=float, default=0.5, help="seconds of boundary tolerance for tcpCER/tcpWER")
    parser.add_argument("--remove-spaces", action="store_true", help="remove spaces before CER")
    parser.add_argument(
        "--metrics",
        default="all",
        help=(
            "comma-separated metrics to compute: all, summary, der, cpcer, tcpcer, cpwer, tcpwer. "
            "Use 'der,summary' for a fast first pass."
        ),
    )
    parser.add_argument(
        "--allow-missing-audios",
        action="store_true",
        help="allow ref/hyp audio-id mismatch; missing audios are scored as complete misses/false alarms",
    )
    parser.add_argument(
        "--audio-scope",
        choices=("strict", "union", "intersection"),
        default="strict",
        help=(
            "how to handle audio-id mismatch: strict fails, union scores missing audios, "
            "intersection scores only ids present in both ref and hyp"
        ),
    )
    args = parser.parse_args()

    ref_turns = read_jsonl_turns(Path(args.ref_jsonl))
    hyp_turns = read_jsonl_turns(Path(args.hyp_jsonl))
    ref_rttm = read_rttm(Path(args.ref_rttm))
    hyp_rttm = read_rttm(Path(args.hyp_rttm))
    audio_scope = "union" if args.allow_missing_audios else args.audio_scope
    ref_turns, hyp_turns, ref_rttm, hyp_rttm, scoped_audio_ids = apply_audio_scope(
        ref_turns,
        hyp_turns,
        ref_rttm,
        hyp_rttm,
        scope=audio_scope,
    )

    normalizer = TextNormalizer(remove_spaces=args.remove_spaces)
    selected_metrics = parse_metrics(args.metrics)

    metrics = {
        "settings": {
            "collar": args.collar,
            "frame_step": args.frame_step,
            "skip_overlap": args.skip_overlap,
            "tcp_window": args.tcp_window,
            "remove_spaces": args.remove_spaces,
            "audio_scope": audio_scope,
            "audio_ids": scoped_audio_ids,
            "metrics": sorted(selected_metrics),
        },
    }
    if "summary" in selected_metrics:
        metrics["summary"] = compute_summary(ref_turns, hyp_turns, normalizer)
    if "der" in selected_metrics:
        metrics["DER"] = compute_der(
            ref_rttm,
            hyp_rttm,
            collar=args.collar,
            frame_step=args.frame_step,
            skip_overlap=args.skip_overlap,
        )
    if "cpcer" in selected_metrics:
        metrics["cpCER"] = compute_cpcer(ref_turns, hyp_turns, normalizer)
    if "tcpcer" in selected_metrics:
        metrics["tcpCER_strict"] = compute_tcpcer(ref_turns, hyp_turns, normalizer, window=0.0)
        metrics[f"tcpCER_{args.tcp_window:.2f}s"] = compute_tcpcer(
            ref_turns,
            hyp_turns,
            normalizer,
            window=args.tcp_window,
        )
    if "cpwer" in selected_metrics:
        metrics["cpWER"] = compute_cpwer(ref_turns, hyp_turns, normalizer)
    if "tcpwer" in selected_metrics:
        metrics["tcpWER_strict"] = compute_tcpwer(ref_turns, hyp_turns, normalizer, window=0.0)
        metrics[f"tcpWER_{args.tcp_window:.2f}s"] = compute_tcpwer(
            ref_turns,
            hyp_turns,
            normalizer,
            window=args.tcp_window,
        )
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    for name in (
        "summary",
        "DER",
        "cpCER",
        "tcpCER_strict",
        f"tcpCER_{args.tcp_window:.2f}s",
        "cpWER",
        "tcpWER_strict",
        f"tcpWER_{args.tcp_window:.2f}s",
    ):
        if name == "summary" and name in metrics:
            print_summary(metrics[name])
        elif name in metrics:
            print_metric(name, metrics[name])
    print(f"[metrics] output={output_json}")
    return 0


def read_jsonl_turns(path: Path) -> dict[str, list[Turn]]:
    grouped: dict[str, list[Turn]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            turn = Turn(
                audio_id=str(row["audio_id"]),
                start=float(row["start"]),
                end=float(row["end"]),
                speaker=str(row.get("speaker", "UNKNOWN")),
                text=str(row.get("text", "")),
            )
            grouped[turn.audio_id].append(turn)
    for turns in grouped.values():
        turns.sort(key=lambda item: (item.start, item.end, item.speaker))
    return dict(grouped)


def read_rttm(path: Path) -> dict[str, list[Turn]]:
    grouped: dict[str, list[Turn]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 8 or fields[0] != "SPEAKER":
                continue
            audio_id = fields[1]
            start = float(fields[3])
            duration = float(fields[4])
            speaker = fields[7]
            grouped[audio_id].append(Turn(audio_id, start, start + duration, speaker))
    for turns in grouped.values():
        turns.sort(key=lambda item: (item.start, item.end, item.speaker))
    return dict(grouped)


def parse_metrics(metrics_arg: str) -> set[str]:
    aliases = {
        "all": {"summary", "der", "cpcer", "tcpcer", "cpwer", "tcpwer"},
        "fast": {"summary", "der"},
        "text": {"cpcer", "cpwer"},
        "tcp": {"tcpcer", "tcpwer"},
    }
    selected: set[str] = set()
    for item in metrics_arg.split(","):
        name = item.strip().lower()
        if not name:
            continue
        if name in aliases:
            selected.update(aliases[name])
        elif name in {"summary", "der", "cpcer", "tcpcer", "cpwer", "tcpwer"}:
            selected.add(name)
        else:
            raise SystemExit(f"unknown metric: {item!r}")
    return selected or aliases["all"]


def apply_audio_scope(
    ref_turns: dict[str, list[Turn]],
    hyp_turns: dict[str, list[Turn]],
    ref_rttm: dict[str, list[Turn]],
    hyp_rttm: dict[str, list[Turn]],
    scope: str,
) -> tuple[dict[str, list[Turn]], dict[str, list[Turn]], dict[str, list[Turn]], dict[str, list[Turn]], list[str]]:
    problems: list[str] = []
    problems.extend(_coverage_problems("JSONL", set(ref_turns), set(hyp_turns)))
    problems.extend(_coverage_problems("RTTM", set(ref_rttm), set(hyp_rttm)))
    if problems and scope == "strict":
        message = "\n".join(problems)
        raise SystemExit(
            "audio-id coverage mismatch between reference and prediction.\n"
            f"{message}\n"
            "This usually means the AMI run is incomplete or pred.jsonl/pred.rttm is stale. "
            "Finish the missing meetings, rebuild predictions with "
            "`python scripts/eval_run_ami.py --collect-only ...`, or explicitly use "
            "`--audio-scope intersection` for a partial sanity check."
        )

    if scope == "intersection":
        ids = sorted(set(ref_turns) & set(hyp_turns) & set(ref_rttm) & set(hyp_rttm))
    elif scope == "union":
        ids = sorted(set(ref_turns) | set(hyp_turns) | set(ref_rttm) | set(hyp_rttm))
    else:
        ids = sorted(set(ref_turns) | set(hyp_turns) | set(ref_rttm) | set(hyp_rttm))

    if problems:
        message = "\n".join(problems)
        print(f"[warn] audio-id coverage mismatch; scoring scope={scope}:\n{message}")
    if not ids:
        raise SystemExit("no shared audio ids to score")

    return (
        _filter_audio_ids(ref_turns, ids),
        _filter_audio_ids(hyp_turns, ids),
        _filter_audio_ids(ref_rttm, ids),
        _filter_audio_ids(hyp_rttm, ids),
        ids,
    )


def _filter_audio_ids(turns: dict[str, list[Turn]], audio_ids: list[str]) -> dict[str, list[Turn]]:
    return {audio_id: turns.get(audio_id, []) for audio_id in audio_ids}


def _coverage_problems(label: str, ref_ids: set[str], hyp_ids: set[str]) -> list[str]:
    problems: list[str] = []
    missing = sorted(ref_ids - hyp_ids)
    extra = sorted(hyp_ids - ref_ids)
    if missing:
        problems.append(f"{label}: missing predictions for {len(missing)} audio ids: {_preview_ids(missing)}")
    if extra:
        problems.append(f"{label}: predictions contain {len(extra)} extra audio ids: {_preview_ids(extra)}")
    return problems


def _preview_ids(ids: list[str], limit: int = 10) -> str:
    shown = ", ".join(ids[:limit])
    if len(ids) > limit:
        shown += f", ... (+{len(ids) - limit})"
    return shown


class TextNormalizer:
    def __init__(self, remove_spaces: bool = False):
        self.remove_spaces = remove_spaces
        self._punct = str.maketrans("", "", string.punctuation)

    def __call__(self, text: str) -> str:
        return self.normalize_text(text, remove_spaces=self.remove_spaces)

    def words(self, text: str) -> tuple[str, ...]:
        return tuple(self.normalize_text(text, remove_spaces=False).split())

    def normalize_text(self, text: str, remove_spaces: bool) -> str:
        text = text.lower()
        text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
        text = text.translate(self._punct)
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        if remove_spaces:
            text = text.replace(" ", "")
        return text


def compute_summary(
    ref_by_audio: dict[str, list[Turn]],
    hyp_by_audio: dict[str, list[Turn]],
    normalize: TextNormalizer,
) -> dict[str, object]:
    per_audio: dict[str, dict[str, object]] = {}
    totals = {
        "ref_turns": 0,
        "hyp_turns": 0,
        "ref_speech_seconds": 0.0,
        "hyp_speech_seconds": 0.0,
        "ref_words": 0,
        "hyp_words": 0,
        "unknown_turns": 0,
    }
    for audio_id in sorted(set(ref_by_audio) | set(hyp_by_audio)):
        ref_turns = ref_by_audio.get(audio_id, [])
        hyp_turns = hyp_by_audio.get(audio_id, [])
        ref_words = sum(len(normalize.words(turn.text)) for turn in ref_turns)
        hyp_words = sum(len(normalize.words(turn.text)) for turn in hyp_turns)
        ref_seconds = sum(turn.duration for turn in ref_turns)
        hyp_seconds = sum(turn.duration for turn in hyp_turns)
        unknown_turns = sum(1 for turn in hyp_turns if turn.speaker == "UNKNOWN")
        row = {
            "ref_turns": len(ref_turns),
            "hyp_turns": len(hyp_turns),
            "ref_speakers": len({turn.speaker for turn in ref_turns}),
            "hyp_speakers": len({turn.speaker for turn in hyp_turns}),
            "unknown_turns": unknown_turns,
            "ref_speech_seconds": ref_seconds,
            "hyp_speech_seconds": hyp_seconds,
            "speech_coverage": ratio(hyp_seconds, ref_seconds),
            "ref_words": ref_words,
            "hyp_words": hyp_words,
            "word_coverage": ratio(hyp_words, ref_words),
        }
        per_audio[audio_id] = row
        totals["ref_turns"] += len(ref_turns)
        totals["hyp_turns"] += len(hyp_turns)
        totals["ref_speech_seconds"] += ref_seconds
        totals["hyp_speech_seconds"] += hyp_seconds
        totals["ref_words"] += ref_words
        totals["hyp_words"] += hyp_words
        totals["unknown_turns"] += unknown_turns

    totals["speech_coverage"] = ratio(totals["hyp_speech_seconds"], totals["ref_speech_seconds"])
    totals["word_coverage"] = ratio(totals["hyp_words"], totals["ref_words"])
    return {"total": totals, "per_audio": per_audio}


def compute_cpcer(
    ref_by_audio: dict[str, list[Turn]],
    hyp_by_audio: dict[str, list[Turn]],
    normalize: TextNormalizer,
) -> dict[str, object]:
    return compute_cp_metric(ref_by_audio, hyp_by_audio, normalize, rate_key="CER", count_key="ref_chars")


def compute_cpwer(
    ref_by_audio: dict[str, list[Turn]],
    hyp_by_audio: dict[str, list[Turn]],
    normalize: TextNormalizer,
) -> dict[str, object]:
    return compute_cp_metric(ref_by_audio, hyp_by_audio, normalize.words, rate_key="WER", count_key="ref_words")


def compute_cp_metric(
    ref_by_audio: dict[str, list[Turn]],
    hyp_by_audio: dict[str, list[Turn]],
    to_units: Callable[[str], UnitSequence],
    rate_key: str,
    count_key: str,
) -> dict[str, object]:
    per_audio: dict[str, dict[str, float]] = {}
    total_edits = 0
    total_ref = 0
    for audio_id in sorted(set(ref_by_audio) | set(hyp_by_audio)):
        ref_texts = concat_by_speaker(ref_by_audio.get(audio_id, []), to_units)
        hyp_texts = concat_by_speaker(hyp_by_audio.get(audio_id, []), to_units)
        edits, ref_len, _mapping = min_permutation_cost(ref_texts, hyp_texts)
        per_audio[audio_id] = {rate_key: ratio(edits, ref_len), "edits": edits, count_key: ref_len}
        total_edits += edits
        total_ref += ref_len
    return {rate_key: ratio(total_edits, total_ref), "edits": total_edits, count_key: total_ref, "per_audio": per_audio}


def compute_tcpcer(
    ref_by_audio: dict[str, list[Turn]],
    hyp_by_audio: dict[str, list[Turn]],
    normalize: TextNormalizer,
    window: float,
) -> dict[str, object]:
    return compute_tcp_metric(ref_by_audio, hyp_by_audio, normalize, window, rate_key="CER", count_key="ref_chars")


def compute_tcpwer(
    ref_by_audio: dict[str, list[Turn]],
    hyp_by_audio: dict[str, list[Turn]],
    normalize: TextNormalizer,
    window: float,
) -> dict[str, object]:
    return compute_tcp_metric(ref_by_audio, hyp_by_audio, normalize.words, window, rate_key="WER", count_key="ref_words")


def compute_tcp_metric(
    ref_by_audio: dict[str, list[Turn]],
    hyp_by_audio: dict[str, list[Turn]],
    to_units: Callable[[str], UnitSequence],
    window: float,
    rate_key: str,
    count_key: str,
) -> dict[str, object]:
    per_audio: dict[str, dict[str, float]] = {}
    total_edits = 0
    total_ref = 0
    for audio_id in sorted(set(ref_by_audio) | set(hyp_by_audio)):
        ref_turns = ref_by_audio.get(audio_id, [])
        hyp_turns = hyp_by_audio.get(audio_id, [])
        ref_speakers = sorted({turn.speaker for turn in ref_turns})
        hyp_speakers = sorted({turn.speaker for turn in hyp_turns})
        best_edits, best_ref = min_tcp_cost(ref_turns, hyp_turns, ref_speakers, hyp_speakers, to_units, window)
        per_audio[audio_id] = {rate_key: ratio(best_edits, best_ref), "edits": best_edits, count_key: best_ref}
        total_edits += best_edits
        total_ref += best_ref
    return {rate_key: ratio(total_edits, total_ref), "edits": total_edits, count_key: total_ref, "per_audio": per_audio}


def concat_by_speaker(turns: list[Turn], to_units: Callable[[str], UnitSequence]) -> dict[str, UnitSequence]:
    grouped: dict[str, list[UnitSequence]] = defaultdict(list)
    for turn in sorted(turns, key=lambda item: (item.start, item.end)):
        units = to_units(turn.text)
        if units:
            grouped[turn.speaker].append(units)
    return {speaker: join_units(parts) for speaker, parts in grouped.items()}


def join_units(parts: list[UnitSequence]) -> UnitSequence:
    if not parts:
        return ""
    if isinstance(parts[0], str):
        return " ".join(str(part) for part in parts).strip()
    words: list[str] = []
    for part in parts:
        words.extend(part)
    return tuple(words)


def min_permutation_cost(
    ref_texts: dict[str, UnitSequence],
    hyp_texts: dict[str, UnitSequence],
) -> tuple[int, int, dict[str, str | None]]:
    ref_speakers = sorted(ref_texts)
    hyp_speakers = sorted(hyp_texts)
    ref_len = sum(len(text) for text in ref_texts.values())
    if not ref_speakers:
        return sum(len(text) for text in hyp_texts.values()), 0, {}
    if not hyp_speakers:
        return ref_len, ref_len, {speaker: None for speaker in ref_speakers}

    n = max(len(ref_speakers), len(hyp_speakers))
    if n > 8:
        return greedy_permutation_cost(ref_texts, hyp_texts)
    ref_slots = ref_speakers + [None] * (n - len(ref_speakers))
    hyp_slots = hyp_speakers + [None] * (n - len(hyp_speakers))
    best_cost = math.inf
    best_mapping: dict[str, str | None] = {}

    for perm in itertools.permutations(hyp_slots):
        cost = 0
        mapping: dict[str, str | None] = {}
        for ref_spk, hyp_spk in zip(ref_slots, perm):
            if ref_spk is None and hyp_spk is None:
                continue
            if ref_spk is None:
                cost += len(hyp_texts[str(hyp_spk)])
            elif hyp_spk is None:
                cost += len(ref_texts[ref_spk])
                mapping[ref_spk] = None
            else:
                cost += edit_distance(ref_texts[ref_spk], hyp_texts[str(hyp_spk)])
                mapping[ref_spk] = str(hyp_spk)
        if cost < best_cost:
            best_cost = cost
            best_mapping = mapping
    return int(best_cost), ref_len, best_mapping


def greedy_permutation_cost(
    ref_texts: dict[str, UnitSequence],
    hyp_texts: dict[str, UnitSequence],
) -> tuple[int, int, dict[str, str | None]]:
    ref_len = sum(len(text) for text in ref_texts.values())
    baseline = ref_len + sum(len(text) for text in hyp_texts.values())
    pairs: list[tuple[int, str, str, int]] = []
    for ref_spk, ref_text in ref_texts.items():
        for hyp_spk, hyp_text in hyp_texts.items():
            paired_cost = edit_distance(ref_text, hyp_text)
            improvement = len(ref_text) + len(hyp_text) - paired_cost
            pairs.append((improvement, ref_spk, hyp_spk, paired_cost))
    pairs.sort(reverse=True)
    used_ref: set[str] = set()
    used_hyp: set[str] = set()
    mapping: dict[str, str | None] = {speaker: None for speaker in ref_texts}
    cost = baseline
    for improvement, ref_spk, hyp_spk, _paired_cost in pairs:
        if improvement <= 0 or ref_spk in used_ref or hyp_spk in used_hyp:
            continue
        used_ref.add(ref_spk)
        used_hyp.add(hyp_spk)
        mapping[ref_spk] = hyp_spk
        cost -= improvement
    return int(cost), ref_len, mapping


def min_tcp_cost(
    ref_turns: list[Turn],
    hyp_turns: list[Turn],
    ref_speakers: list[str],
    hyp_speakers: list[str],
    to_units: Callable[[str], UnitSequence],
    window: float,
) -> tuple[int, int]:
    ref_len = sum(len(to_units(turn.text)) for turn in ref_turns)
    if not ref_speakers:
        return sum(len(to_units(turn.text)) for turn in hyp_turns), 0
    n = max(len(ref_speakers), len(hyp_speakers))
    if n > 8:
        return greedy_tcp_cost(ref_turns, hyp_turns, ref_speakers, hyp_speakers, to_units, window), ref_len
    ref_slots = ref_speakers + [None] * (n - len(ref_speakers))
    hyp_slots = hyp_speakers + [None] * (n - len(hyp_speakers))
    best = math.inf
    for perm in itertools.permutations(hyp_slots):
        mapping = {ref: hyp for ref, hyp in zip(ref_slots, perm) if ref is not None}
        cost = tcp_cost_for_mapping(ref_turns, hyp_turns, mapping, to_units, window)
        if cost < best:
            best = cost
    return int(best), ref_len


def greedy_tcp_cost(
    ref_turns: list[Turn],
    hyp_turns: list[Turn],
    ref_speakers: list[str],
    hyp_speakers: list[str],
    to_units: Callable[[str], UnitSequence],
    window: float,
) -> int:
    empty_mapping = {speaker: None for speaker in ref_speakers}
    baseline = tcp_cost_for_mapping(ref_turns, hyp_turns, empty_mapping, to_units, window)
    pairs: list[tuple[int, str, str]] = []
    for ref_spk in ref_speakers:
        for hyp_spk in hyp_speakers:
            mapping = {speaker: None for speaker in ref_speakers}
            mapping[ref_spk] = hyp_spk
            pair_cost = tcp_cost_for_mapping(
                [turn for turn in ref_turns if turn.speaker == ref_spk],
                [turn for turn in hyp_turns if turn.speaker == hyp_spk],
                mapping,
                to_units,
                window,
            )
            pair_baseline = sum(len(to_units(turn.text)) for turn in ref_turns if turn.speaker == ref_spk)
            pair_baseline += sum(len(to_units(turn.text)) for turn in hyp_turns if turn.speaker == hyp_spk)
            improvement = pair_baseline - pair_cost
            pairs.append((improvement, ref_spk, hyp_spk))
    pairs.sort(reverse=True)
    used_ref: set[str] = set()
    used_hyp: set[str] = set()
    mapping = {speaker: None for speaker in ref_speakers}
    for improvement, ref_spk, hyp_spk in pairs:
        if improvement <= 0 or ref_spk in used_ref or hyp_spk in used_hyp:
            continue
        used_ref.add(ref_spk)
        used_hyp.add(hyp_spk)
        mapping[ref_spk] = hyp_spk
    return tcp_cost_for_mapping(ref_turns, hyp_turns, mapping, to_units, window)


def tcp_cost_for_mapping(
    ref_turns: list[Turn],
    hyp_turns: list[Turn],
    mapping: dict[str, str | None],
    to_units: Callable[[str], UnitSequence],
    window: float,
) -> int:
    used_hyp: set[int] = set()
    cost = 0
    for ref in sorted(ref_turns, key=lambda item: (item.start, item.end)):
        ref_units = to_units(ref.text)
        hyp_speaker = mapping.get(ref.speaker)
        if hyp_speaker is None:
            cost += len(ref_units)
            continue
        candidates: list[tuple[float, float, int, Turn]] = []
        for index, hyp in enumerate(hyp_turns):
            if index in used_hyp or hyp.speaker != hyp_speaker:
                continue
            if time_match(ref, hyp, window):
                overlap = interval_overlap(ref.start, ref.end, hyp.start, hyp.end)
                boundary_distance = abs(ref.start - hyp.start) + abs(ref.end - hyp.end)
                candidates.append((-overlap, boundary_distance, index, hyp))
        candidates.sort()
        overlapping = [candidate for candidate in candidates if candidate[0] < 0.0]
        selected = overlapping or candidates[:1]
        hyp_parts: list[UnitSequence] = []
        for _score, _distance, index, hyp in selected:
            used_hyp.add(index)
            units = to_units(hyp.text)
            if units:
                hyp_parts.append(units)
        hyp_units = join_units(hyp_parts)
        cost += edit_distance(ref_units, hyp_units)

    for index, hyp in enumerate(hyp_turns):
        if index not in used_hyp:
            units = to_units(hyp.text)
            cost += len(units)
    return cost


def time_match(ref: Turn, hyp: Turn, window: float) -> bool:
    if interval_overlap(ref.start, ref.end, hyp.start, hyp.end) > 0:
        return True
    if window <= 0:
        return False
    return hyp.end >= ref.start - window and hyp.start <= ref.end + window


def compute_der(
    ref_by_audio: dict[str, list[Turn]],
    hyp_by_audio: dict[str, list[Turn]],
    collar: float,
    frame_step: float,
    skip_overlap: bool,
) -> dict[str, object]:
    per_audio: dict[str, dict[str, float]] = {}
    totals = {"miss": 0.0, "false_alarm": 0.0, "confusion": 0.0, "reference": 0.0}
    for audio_id in sorted(set(ref_by_audio) | set(hyp_by_audio)):
        stats = der_one(ref_by_audio.get(audio_id, []), hyp_by_audio.get(audio_id, []), collar, frame_step, skip_overlap)
        per_audio[audio_id] = der_stats_to_dict(stats)
        for key in totals:
            totals[key] += stats[key]
    result = der_stats_to_dict(totals)
    result["per_audio"] = per_audio
    return result


def der_one(ref_turns: list[Turn], hyp_turns: list[Turn], collar: float, frame_step: float, skip_overlap: bool) -> dict[str, float]:
    end_time = max([0.0] + [turn.end for turn in ref_turns] + [turn.end for turn in hyp_turns])
    ref_speakers = sorted({turn.speaker for turn in ref_turns})
    hyp_speakers = sorted({turn.speaker for turn in hyp_turns})
    mapping = best_der_mapping(ref_turns, hyp_turns, ref_speakers, hyp_speakers, end_time, collar, frame_step, skip_overlap)
    stats = {"miss": 0.0, "false_alarm": 0.0, "confusion": 0.0, "reference": 0.0}
    for t, duration in scored_regions(ref_turns, hyp_turns, end_time, collar):
        ref = active_speakers(ref_turns, t)
        if skip_overlap and len(ref) > 1:
            continue
        hyp_raw = active_speakers(hyp_turns, t)
        hyp = {mapping[speaker] for speaker in hyp_raw if speaker in mapping and mapping[speaker] is not None}
        hyp_count = len(hyp_raw)
        ref_count = len(ref)
        if ref_count == 0 and hyp_count == 0:
            continue
        correct = len(ref & hyp)
        stats["reference"] += ref_count * duration
        stats["miss"] += max(0, ref_count - hyp_count) * duration
        stats["false_alarm"] += max(0, hyp_count - ref_count) * duration
        stats["confusion"] += max(0, min(ref_count, hyp_count) - correct) * duration
    return stats


def best_der_mapping(
    ref_turns: list[Turn],
    hyp_turns: list[Turn],
    ref_speakers: list[str],
    hyp_speakers: list[str],
    end_time: float,
    collar: float,
    frame_step: float,
    skip_overlap: bool,
) -> dict[str, str | None]:
    if not ref_speakers:
        return {speaker: None for speaker in hyp_speakers}
    overlap: dict[tuple[str, str], float] = defaultdict(float)
    for t, duration in scored_regions(ref_turns, hyp_turns, end_time, collar):
        ref = active_speakers(ref_turns, t)
        if skip_overlap and len(ref) > 1:
            continue
        hyp = active_speakers(hyp_turns, t)
        for ref_spk in ref:
            for hyp_spk in hyp:
                overlap[(ref_spk, hyp_spk)] += duration
    n = max(len(ref_speakers), len(hyp_speakers))
    if n > 8:
        return greedy_der_mapping(overlap, ref_speakers, hyp_speakers)
    ref_slots = ref_speakers + [None] * (n - len(ref_speakers))
    hyp_slots = hyp_speakers + [None] * (n - len(hyp_speakers))
    best_score = -1.0
    best_mapping: dict[str, str | None] = {}
    for perm in itertools.permutations(ref_slots):
        score = 0.0
        mapping: dict[str, str | None] = {}
        for hyp_spk, ref_spk in zip(hyp_slots, perm):
            if hyp_spk is None:
                continue
            mapping[hyp_spk] = ref_spk
            if ref_spk is not None:
                score += overlap[(ref_spk, hyp_spk)]
        if score > best_score:
            best_score = score
            best_mapping = mapping
    return best_mapping


def scored_regions(
    ref_turns: list[Turn],
    hyp_turns: list[Turn],
    end_time: float,
    collar: float,
) -> list[tuple[float, float]]:
    boundaries = {0.0, end_time}
    for turn in itertools.chain(ref_turns, hyp_turns):
        boundaries.add(max(0.0, turn.start))
        boundaries.add(max(0.0, turn.end))
    if collar > 0:
        for turn in ref_turns:
            boundaries.add(max(0.0, turn.start - collar))
            boundaries.add(max(0.0, turn.start + collar))
            boundaries.add(max(0.0, turn.end - collar))
            boundaries.add(max(0.0, turn.end + collar))
    points = sorted(point for point in boundaries if 0.0 <= point <= end_time)
    regions: list[tuple[float, float]] = []
    for left, right in zip(points, points[1:]):
        if right <= left:
            continue
        mid = (left + right) / 2.0
        if in_collar(mid, ref_turns, collar):
            continue
        regions.append((mid, right - left))
    return regions


def greedy_der_mapping(
    overlap: dict[tuple[str, str], float],
    ref_speakers: list[str],
    hyp_speakers: list[str],
) -> dict[str, str | None]:
    pairs: list[tuple[float, str, str]] = []
    for ref_spk in ref_speakers:
        for hyp_spk in hyp_speakers:
            pairs.append((overlap[(ref_spk, hyp_spk)], ref_spk, hyp_spk))
    pairs.sort(reverse=True)
    used_ref: set[str] = set()
    used_hyp: set[str] = set()
    mapping: dict[str, str | None] = {speaker: None for speaker in hyp_speakers}
    for score, ref_spk, hyp_spk in pairs:
        if score <= 0 or ref_spk in used_ref or hyp_spk in used_hyp:
            continue
        used_ref.add(ref_spk)
        used_hyp.add(hyp_spk)
        mapping[hyp_spk] = ref_spk
    return mapping


def in_collar(time: float, ref_turns: list[Turn], collar: float) -> bool:
    if collar <= 0:
        return False
    for turn in ref_turns:
        if abs(time - turn.start) <= collar or abs(time - turn.end) <= collar:
            return True
    return False


def active_speakers(turns: list[Turn], time: float) -> set[str]:
    return {turn.speaker for turn in turns if turn.start <= time < turn.end}


def der_stats_to_dict(stats: dict[str, float]) -> dict[str, float]:
    ref = stats["reference"]
    total = stats["miss"] + stats["false_alarm"] + stats["confusion"]
    return {
        "DER": ratio(total, ref),
        "miss": ratio(stats["miss"], ref),
        "false_alarm": ratio(stats["false_alarm"], ref),
        "confusion": ratio(stats["confusion"], ref),
        "scored_reference_seconds": ref,
    }


def edit_distance(left: UnitSequence, right: UnitSequence) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    try:
        from rapidfuzz.distance import Levenshtein

        return int(Levenshtein.distance(left, right))
    except Exception:
        pass
    try:
        import Levenshtein

        return int(Levenshtein.distance(left, right))
    except Exception:
        pass
    if isinstance(left, str) and isinstance(right, str) and max(len(left), len(right)) >= 256:
        return myers_distance(left, right)
    return dp_edit_distance(left, right)


def myers_distance(pattern: str, text: str) -> int:
    if len(pattern) > len(text):
        pattern, text = text, pattern
    m = len(pattern)
    if m == 0:
        return len(text)

    peq: dict[str, int] = {}
    for i, char in enumerate(pattern):
        peq[char] = peq.get(char, 0) | (1 << i)

    score = m
    mask = 1 << (m - 1)
    all_bits = (1 << m) - 1
    positive = all_bits
    negative = 0

    for char in text:
        eq = peq.get(char, 0)
        x = eq | negative
        d0 = (((x & positive) + positive) ^ positive) | x
        hp = negative | ~(d0 | positive)
        hn = positive & d0
        if hp & mask:
            score += 1
        elif hn & mask:
            score -= 1
        hp = ((hp << 1) | 1) & all_bits
        hn = (hn << 1) & all_bits
        positive = (hn | ~(d0 | hp)) & all_bits
        negative = hp & d0
    return int(score)


def dp_edit_distance(left: UnitSequence, right: UnitSequence) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def interval_overlap(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))


def ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0 if numerator <= 0 else 1.0
    return float(numerator) / float(denominator)


def print_metric(name: str, value: dict[str, object]) -> None:
    key = "DER" if name == "DER" else "WER" if "WER" in name else "CER"
    count_key = "ref_words" if key == "WER" else "ref_chars"
    score = float(value[key])
    edits = value.get("edits")
    if edits is None:
        print(f"{name}: {score * 100:.2f}%")
    else:
        print(f"{name}: {score * 100:.2f}%  edits={edits} {count_key}={value.get(count_key)}")


def print_summary(value: object) -> None:
    if not isinstance(value, dict):
        return
    total = value.get("total", {})
    if not isinstance(total, dict):
        return
    print(
        "summary: "
        f"turns hyp/ref={total.get('hyp_turns')}/{total.get('ref_turns')} "
        f"speech_coverage={float(total.get('speech_coverage', 0.0)) * 100:.2f}% "
        f"words hyp/ref={total.get('hyp_words')}/{total.get('ref_words')} "
        f"word_coverage={float(total.get('word_coverage', 0.0)) * 100:.2f}% "
        f"unknown_turns={total.get('unknown_turns')}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
