"""Unit tests for ``hiexp.evaluation.benchmarks`` and ``judge``.

Covers what the agent doesn't: the benchmark-spec registry, scorer
construction, and judge-output parsing. The Scorer protocol is exercised
via the synchronous ``WordLevelScorer`` only; ``JudgeScorer`` requires a
live LLM and is covered by integration scripts, not unit tests.
"""

from __future__ import annotations

import asyncio

import pytest

from hiexp.evaluation import (
    BenchmarkSpec,
    JudgeScorer,
    Scorer,
    WordLevelScorer,
    extract_citations_from_response,
    compute_citation_metrics,
    get_spec,
    list_benchmarks,
    make_scorer,
    parse_judge_response,
    register,
    render_grader_prompt,
)


def _arun(coro):
    return asyncio.get_event_loop().run_until_complete(coro) if asyncio.get_event_loop().is_running() else asyncio.run(coro)


# ── Registry ─────────────────────────────────────────────────────────────


def test_registry_has_builtin_specs():
    names = list_benchmarks()
    assert "hotpotqa" in names
    assert "browsecomp_plus" in names

    hotpot = get_spec("hotpotqa")
    assert hotpot.scorer_kind == "word_level"
    assert hotpot.question_template == "hiexp_boxed"
    assert hotpot.extractor_field == "prediction_boxed"

    bcp = get_spec("browsecomp_plus")
    assert bcp.scorer_kind == "llm_judge"
    assert bcp.question_template == "browsecomp_plus"
    assert bcp.extractor_field == "prediction_exact"


def test_get_spec_unknown_raises():
    with pytest.raises(KeyError, match="unknown benchmark"):
        get_spec("not_a_real_benchmark")


def test_register_adds_new_spec():
    spec = BenchmarkSpec(
        name="dummy_test_spec",
        question_template="raw",
        extractor=lambda s: s.strip(),
        extractor_field="prediction_raw_x",
        scorer_kind="word_level",
        decisive_metric="em",
    )
    register(spec)
    assert get_spec("dummy_test_spec") is spec
    assert "dummy_test_spec" in list_benchmarks()


# ── WordLevelScorer ──────────────────────────────────────────────────────


def test_word_level_correct_match_cem():
    scorer = WordLevelScorer(decisive_metric="cem")
    out = asyncio.run(
        scorer.ascore(
            question="what is the capital of France?",
            golden_answers=["Paris"],
            prediction_raw="The answer is Paris.",
            prediction_extracted="Paris",
        )
    )
    assert out["correct"] is True
    assert out["em"] == 1
    assert out["cem"] == 1
    assert out["f1"] == 1.0


def test_word_level_substring_match_cem_only():
    """CEM rewards substring containment; EM still requires exact match."""
    scorer = WordLevelScorer(decisive_metric="cem")
    out = asyncio.run(
        scorer.ascore(
            question="...",
            golden_answers=["Paris"],
            prediction_raw="...",
            prediction_extracted="Paris, France",
        )
    )
    assert out["correct"] is True   # CEM = 1 (gold ⊂ prediction)
    assert out["cem"] == 1
    assert out["em"] == 0


def test_word_level_em_strict():
    scorer = WordLevelScorer(decisive_metric="em")
    out = asyncio.run(
        scorer.ascore(
            question="...",
            golden_answers=["Paris"],
            prediction_raw="...",
            prediction_extracted="Paris, France",
        )
    )
    assert out["correct"] is False   # EM = 0 → bucket as neg
    assert out["em"] == 0
    assert out["cem"] == 1


def test_word_level_wrong_answer():
    scorer = WordLevelScorer(decisive_metric="cem")
    out = asyncio.run(
        scorer.ascore(
            question="...",
            golden_answers=["Paris"],
            prediction_raw="...",
            prediction_extracted="London",
        )
    )
    assert out["correct"] is False


def test_word_level_invalid_metric():
    with pytest.raises(ValueError, match="decisive_metric"):
        WordLevelScorer(decisive_metric="bleurt")  # type: ignore[arg-type]


# ── make_scorer ──────────────────────────────────────────────────────────


def test_make_scorer_word_level():
    scorer = make_scorer(get_spec("hotpotqa"))
    assert isinstance(scorer, WordLevelScorer)
    # Protocol check
    assert isinstance(scorer, Scorer)


def test_make_scorer_llm_judge_requires_llm():
    with pytest.raises(ValueError, match="judge_llm"):
        make_scorer(get_spec("browsecomp_plus"))


def test_make_scorer_llm_judge_with_llm():
    class _StubLLM:
        async def ainvoke(self, msgs):  # noqa: D401 - test stub
            class _R:
                content = "irrelevant"
            return _R()

    scorer = make_scorer(get_spec("browsecomp_plus"), judge_llm=_StubLLM())
    assert isinstance(scorer, JudgeScorer)
    assert isinstance(scorer, Scorer)


# ── JudgeScorer (via stub LLM) ───────────────────────────────────────────


def test_judge_scorer_parses_yes():
    class _StubLLM:
        async def ainvoke(self, msgs):
            class _R:
                content = (
                    "extracted_final_answer: Paris\n"
                    "reasoning: matches gold\n"
                    "correct: yes\n"
                    "confidence: 95"
                )
            return _R()

    scorer = JudgeScorer(llm=_StubLLM())
    out = asyncio.run(
        scorer.ascore(
            question="capital of France?",
            golden_answers=["Paris"],
            prediction_raw="Paris is the capital.",
            prediction_extracted="Paris",
        )
    )
    assert out["correct"] is True
    assert out["judge_extracted"] == "Paris"
    assert out["judge_confidence"] == 95.0
    assert out["judge_parse_error"] is False


def test_judge_scorer_parses_no():
    class _StubLLM:
        async def ainvoke(self, msgs):
            class _R:
                content = (
                    "extracted_final_answer: London\n"
                    "reasoning: wrong city\n"
                    "correct: no\n"
                    "confidence: 80"
                )
            return _R()

    scorer = JudgeScorer(llm=_StubLLM())
    out = asyncio.run(
        scorer.ascore(
            question="...",
            golden_answers=["Paris"],
            prediction_raw="London",
            prediction_extracted="London",
        )
    )
    assert out["correct"] is False
    assert out["judge_confidence"] == 80.0


def test_judge_scorer_handles_llm_exception():
    class _BoomLLM:
        async def ainvoke(self, msgs):
            raise RuntimeError("oops")

    scorer = JudgeScorer(llm=_BoomLLM())
    out = asyncio.run(
        scorer.ascore(
            question="...",
            golden_answers=["Paris"],
            prediction_raw="Paris",
            prediction_extracted="Paris",
        )
    )
    assert out["correct"] is False
    assert out["judge_parse_error"] is True
    assert "oops" in out["judge_error"]


# ── Judge primitives ─────────────────────────────────────────────────────


def test_parse_judge_response_bold_form():
    text = (
        "**extracted_final_answer:** Paris\n"
        "**reasoning:** matches\n"
        "**correct:** yes\n"
        "**confidence:** 95"
    )
    r = parse_judge_response(text)
    assert r["correct"] is True
    assert r["extracted_final_answer"] == "Paris"
    assert r["confidence"] == 95.0
    assert r["parse_error"] is False


def test_parse_judge_response_missing_correct():
    text = "extracted_final_answer: Paris\nreasoning: ok\nconfidence: 50"
    r = parse_judge_response(text)
    assert r["parse_error"] is True
    assert r["correct"] is None


def test_parse_judge_response_clamps_confidence():
    text = "extracted_final_answer: x\ncorrect: yes\nconfidence: 250"
    r = parse_judge_response(text)
    assert r["confidence"] == 100.0


def test_extract_citations_dedupes_across_forms():
    cites = extract_citations_from_response("see [1] and [2, 3] also 【4】")
    assert sorted(cites) == ["1", "2", "3", "4"]


def test_compute_citation_metrics_precision_recall():
    m = compute_citation_metrics(["1", "2", "3"], ["2", "4"])
    assert m["precision"] == pytest.approx(1 / 3)
    assert m["recall"] == pytest.approx(0.5)


def test_render_grader_prompt_has_placeholders_filled():
    p = render_grader_prompt(
        question="Q?", response="R", correct_answer="A"
    )
    assert "Q?" in p
    assert "[response]: R" in p
    assert "[correct_answer]: A" in p
