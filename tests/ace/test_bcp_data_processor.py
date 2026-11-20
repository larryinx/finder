"""Unit tests for the BCP DataProcessor (``src/ace_react/bcp/data_processor.py``).

Covers the ``answer_is_correct`` branching: fast-path on identical
strings (no LLM call), LLM-as-judge on non-trivial pairs, and the
``parse_judge_response`` integration. The judge LLM is mocked — we
don't make network calls in unit tests.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from ace_react.bcp.data_processor import DataProcessor


@pytest.fixture
def proc(monkeypatch):
    # Ensure the env-based config has *something* so make_chat_model
    # constructs without complaining; we never actually call the LLM.
    monkeypatch.setenv("AGENT_LLM_MODEL", "test-model")
    monkeypatch.setenv("AGENT_LLM_BASE_URL", "http://localhost:1/v1")
    monkeypatch.setenv("AGENT_LLM_API_KEY", "EMPTY")
    return DataProcessor()


# ── fast-path: trivial matches don't call the judge ──────────────────────


def test_exact_match_skips_judge(proc):
    with patch("ace_react.bcp.data_processor.achat_once") as mock_call:
        assert proc.answer_is_correct("Paris", "Paris") is True
        assert proc.answer_is_correct("paris", "PARIS") is True
        assert proc.answer_is_correct("  Paris  ", "Paris") is True
        mock_call.assert_not_called()


def test_empty_inputs_short_circuit(proc):
    with patch("ace_react.bcp.data_processor.achat_once") as mock_call:
        assert proc.answer_is_correct("", "Paris") is False
        assert proc.answer_is_correct("Paris", "") is False
        assert proc.answer_is_correct("", "") is False
        mock_call.assert_not_called()


# ── LLM-as-judge path ────────────────────────────────────────────────────


def _judge_yes(*_args, **_kw):
    return (
        "extracted_final_answer: Paris, France\n"
        "reasoning: matches gold\n"
        "correct: yes\n"
        "confidence: 95"
    )


def _judge_no(*_args, **_kw):
    return (
        "extracted_final_answer: London\n"
        "reasoning: wrong city\n"
        "correct: no\n"
        "confidence: 80"
    )


def test_judge_yes(proc):
    """LLM judge says yes → True even when surface strings differ."""
    async def fake(*args, **kw):
        return _judge_yes()
    with patch("ace_react.bcp.data_processor.achat_once", side_effect=fake):
        assert proc.answer_is_correct("Paris, France", "Paris") is True


def test_judge_no(proc):
    async def fake(*args, **kw):
        return _judge_no()
    with patch("ace_react.bcp.data_processor.achat_once", side_effect=fake):
        assert proc.answer_is_correct("Tokyo", "Paris") is False


def test_unparseable_judge_response_treated_as_incorrect(proc):
    """parse_judge_response returns parse_error when no `correct: yes/no`
    is present — should map to False (don't pretend correctness)."""
    async def fake(*args, **kw):
        return "rambling response without the expected structure"
    with patch("ace_react.bcp.data_processor.achat_once", side_effect=fake):
        assert proc.answer_is_correct("Tokyo", "Paris") is False


# ── evaluate_accuracy ────────────────────────────────────────────────────


def test_evaluate_accuracy_aggregates(proc):
    async def fake(*args, **kw):
        return _judge_yes()  # always yes via judge
    with patch("ace_react.bcp.data_processor.achat_once", side_effect=fake):
        # 2 of 3 are exact (fast-path yes); 1 differs and goes through judge yes.
        preds = ["Paris", "Tokyo, JP", "London"]
        golds = ["Paris", "Tokyo",     "London"]
        acc = proc.evaluate_accuracy(preds, golds)
        assert acc == pytest.approx(1.0)


def test_evaluate_accuracy_length_mismatch_raises(proc):
    with pytest.raises(ValueError, match="≠"):
        proc.evaluate_accuracy(["a"], ["a", "b"])


def test_evaluate_accuracy_empty(proc):
    assert proc.evaluate_accuracy([], []) == 0.0


# ── load_bcp_split smoke test (live data) ────────────────────────────────


def test_load_bcp_split_sizes():
    """Verifies the data prep produced disjoint 200/500/100 splits.

    Skipped automatically if data/BCP/ hasn't been built yet.
    """
    from pathlib import Path

    from ace_react.bcp.data_processor import load_bcp_split

    project_root = Path(__file__).resolve().parents[2]
    bcp_root = project_root / "data" / "BCP"
    paths = {
        "eval200":  bcp_root / "topics-qrels-200"      / "queries.tsv",
        "train500": bcp_root / "topics-qrels-train500" / "queries.tsv",
        "val100":   bcp_root / "topics-qrels-val100"   / "queries.tsv",
    }
    if not all(p.exists() for p in paths.values()):
        pytest.skip("data/BCP not materialized; run scripts/data_prep/prepare.sh hiexp")

    e = {td["query_id"] for td in load_bcp_split(paths["eval200"])}
    t = {td["query_id"] for td in load_bcp_split(paths["train500"])}
    v = {td["query_id"] for td in load_bcp_split(paths["val100"])}
    assert len(e) == 200
    assert len(t) == 500
    assert len(v) == 100
    assert e.isdisjoint(t)
    assert e.isdisjoint(v)
    assert t.isdisjoint(v)
