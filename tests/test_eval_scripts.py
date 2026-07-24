from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_eval_metrics_strict_scope_rejects_missing_audio_ids():
    metrics = load_script("eval_metrics")
    ref_turns = {"a": [metrics.Turn("a", 0.0, 1.0, "ref")]}
    hyp_turns = {}

    with pytest.raises(SystemExit) as exc_info:
        metrics.apply_audio_scope(ref_turns, hyp_turns, ref_turns, hyp_turns, scope="strict")

    assert "audio-id coverage mismatch" in str(exc_info.value)


def test_eval_metrics_intersection_scope_keeps_only_shared_audio_ids():
    metrics = load_script("eval_metrics")
    ref_turns = {
        "a": [metrics.Turn("a", 0.0, 1.0, "ref")],
        "b": [metrics.Turn("b", 0.0, 1.0, "ref")],
    }
    hyp_turns = {"a": [metrics.Turn("a", 0.0, 1.0, "hyp")]}

    scoped = metrics.apply_audio_scope(ref_turns, hyp_turns, ref_turns, hyp_turns, scope="intersection")

    assert scoped[-1] == ["a"]
    assert set(scoped[0]) == {"a"}
    assert set(scoped[1]) == {"a"}


def test_eval_run_collect_existing_predictions_requires_all_manifest_meetings(tmp_path):
    runner = load_script("eval_run_ami")
    per_meeting_dir = tmp_path / "meetings"
    per_meeting_dir.mkdir()
    row = {"audio_id": "a", "start": 0.0, "end": 1.0, "speaker": "SPEAKER_01", "text": "hello"}
    with (per_meeting_dir / "a.pred.jsonl").open("w", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")

    with pytest.raises(SystemExit) as exc_info:
        runner._collect_existing_predictions(
            [{"audio_id": "a"}, {"audio_id": "b"}],
            per_meeting_dir,
        )

    assert "missing per-meeting predictions" in str(exc_info.value)


def test_eval_run_slice_waveform_keeps_warmup_before_scoring_window():
    runner = load_script("eval_run_ami")
    sample_rate = 10
    waveform = np.arange(1000, dtype=np.float32)

    sliced, decode_start, score_start, score_end, duration = runner._slice_waveform(
        waveform,
        sample_rate,
        start_seconds=60.0,
        max_seconds=20.0,
        warmup_seconds=30.0,
    )

    assert decode_start == 30.0
    assert score_start == 60.0
    assert score_end == 80.0
    assert duration == 50.0
    assert sliced[0] == 300
    assert sliced[-1] == 799


def test_eval_run_drop_boundary_policy_filters_crossing_turns(monkeypatch, tmp_path):
    runner = load_script("eval_run_ami")

    class FakePipeline:
        def __init__(self, _config):
            self.transcript = [
                type("Turn", (), {"start": 29.0, "end": 31.0, "speaker_id": "S1", "text": "before", "confidence": 1.0})(),
                type("Turn", (), {"start": 31.0, "end": 33.0, "speaker_id": "S1", "text": "inside", "confidence": 1.0})(),
                type("Turn", (), {"start": 39.0, "end": 41.0, "speaker_id": "S1", "text": "after", "confidence": 1.0})(),
            ]

    monkeypatch.setattr(runner, "load_config", lambda _path: object())
    monkeypatch.setattr(runner, "read_wav", lambda _path: (np.zeros(500, dtype=np.float32), 10))
    monkeypatch.setattr(runner, "StreamingCascadePipeline", FakePipeline)
    monkeypatch.setattr(runner, "_process_with_progress", lambda *args, **kwargs: None)

    rows = runner._run_one(
        "a",
        tmp_path / "a.wav",
        "config.yaml",
        frame_seconds=0.2,
        start_seconds=30.0,
        max_seconds=10.0,
        warmup_seconds=30.0,
        boundary_policy="drop",
        progress_seconds=30.0,
    )

    assert [row["text"] for row in rows] == ["inside"]


def test_eval_metrics_computes_cpwer_and_tcpwer():
    metrics = load_script("eval_metrics")
    normalizer = metrics.TextNormalizer()
    ref_turns = {
        "a": [
            metrics.Turn("a", 0.0, 1.0, "ref_a", "hello world"),
            metrics.Turn("a", 1.0, 2.0, "ref_b", "good morning"),
        ]
    }
    hyp_turns = {
        "a": [
            metrics.Turn("a", 0.0, 1.0, "hyp_x", "hello word"),
            metrics.Turn("a", 1.0, 2.0, "hyp_y", "good morning"),
        ]
    }

    cpwer = metrics.compute_cpwer(ref_turns, hyp_turns, normalizer)
    tcpwer = metrics.compute_tcpwer(ref_turns, hyp_turns, normalizer, window=0.0)

    assert cpwer["ref_words"] == 4
    assert cpwer["edits"] == 1
    assert cpwer["WER"] == 0.25
    assert tcpwer["ref_words"] == 4
    assert tcpwer["edits"] == 1
    assert tcpwer["WER"] == 0.25


def test_eval_metrics_parse_metrics_and_summary():
    metrics = load_script("eval_metrics")
    assert metrics.parse_metrics("fast") == {"summary", "der"}
    assert metrics.parse_metrics("cpwer,tcpwer") == {"cpwer", "tcpwer"}

    normalizer = metrics.TextNormalizer()
    ref_turns = {"a": [metrics.Turn("a", 0.0, 2.0, "ref", "hello world")]}
    hyp_turns = {"a": [metrics.Turn("a", 0.0, 1.0, "UNKNOWN", "hello")]}
    summary = metrics.compute_summary(ref_turns, hyp_turns, normalizer)

    assert summary["total"]["ref_words"] == 2
    assert summary["total"]["hyp_words"] == 1
    assert summary["total"]["word_coverage"] == 0.5
    assert summary["total"]["unknown_turns"] == 1
