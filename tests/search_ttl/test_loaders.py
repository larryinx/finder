"""Benchmark loader registry + the shared ``load_split`` dispatch."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from search_ttl.evaluator.loaders import (
    get_loader,
    load_bcp_split,
    load_browsecomp_split,
    load_hotpotqa_split,
    load_split,
)
from search_ttl.rollout.schemas import QuestionRecord


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def test_get_loader_registry():
    assert get_loader("browsecomp_plus") is load_bcp_split
    assert get_loader("browsecomp") is load_browsecomp_split
    assert get_loader("hotpotqa") is load_hotpotqa_split


def test_load_split_hotpotqa_keeps_golden_answers(tmp_path):
    p = _write_jsonl(tmp_path / "eval.jsonl", [
        {"query_id": "h1", "question": "Who?", "golden_answers": ["Ada", "Ada Lovelace"]},
    ])
    rows = load_split("hotpotqa", p)
    assert rows == [{
        "query_id": "h1", "question": "Who?", "target": "Ada",
        "golden_answers": ["Ada", "Ada Lovelace"], "context": "",
    }]
    q = QuestionRecord.from_bcp_dict(rows[0])
    assert (q.qid, q.target, q.golden_answers) == ("h1", "Ada", ["Ada", "Ada Lovelace"])


def test_load_split_browsecomp_single_target(tmp_path):
    p = _write_jsonl(tmp_path / "eval.jsonl", [{"query_id": "b1", "question": "Q?", "target": "42"}])
    rows = load_split("browsecomp", p, decrypted_jsonl="ignored-for-this-benchmark")
    assert rows[0]["target"] == "42"
    q = QuestionRecord.from_bcp_dict(rows[0])
    # A lone target is surfaced as the sole gold answer.
    assert q.golden_answers == ["42"]


def test_load_split_missing_file_is_explicit(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_split("hotpotqa", tmp_path / "nope.jsonl")
