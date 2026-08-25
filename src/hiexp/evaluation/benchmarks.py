"""Benchmark spec + scorer abstractions.

A ``BenchmarkSpec`` is the *only* thing that should change when we plug
in a new dataset. It bundles the four dimensions that are
benchmark-specific:

1. ``question_template`` — which user-message wrapper the agent uses.
2. ``extractor`` — how to pull the final answer out of the agent's
   raw final text.
3. ``extractor_field`` — what key to write the extracted answer under
   (e.g. ``prediction_boxed`` for HotpotQA, ``prediction_exact`` for BCP).
4. ``scorer_kind`` — ``word_level`` (EM/CEM/F1) or ``llm_judge``
   (BCP ``GRADER_TEMPLATE``). The script-side ``make_scorer`` factory
   builds the matching :class:`Scorer` instance.

What is *deliberately* NOT in the spec:

* dataset loaders — those live in ``scripts/hiexp_prompt/_shared.py``
  because they touch FS layout / HuggingFace and don't belong in
  ``src/``;
* the agent — the agent is a single, dataset-agnostic class
  (``hiexp.agent.HiExpAgent``);
* the LLM endpoint — judge config (model / base_url / api_key) is
  threaded in at scorer build time, not encoded in the spec.

This separation is the contract that lets ``run_eval.py`` and
``sample_rollouts.py`` share the same orchestration loop and only
differ in I/O (1 rollout + metrics summary vs K rollouts + pos/neg
buckets).

Plug-in checklist for a new benchmark:

* Add a ``QaExample`` loader to ``_shared.py`` (or wherever your
  dataset lives).
* Pick one of the existing ``scorer_kind`` values, or add a new
  scorer class implementing the :class:`Scorer` protocol.
* Register a ``BenchmarkSpec`` here. That's it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, Protocol, runtime_checkable

import weave

from hiexp.evaluation.answer_extract import (
    extract_boxed_answer,
    extract_exact_answer,
)
from hiexp.evaluation.judge import parse_judge_response, render_grader_prompt
from hiexp.evaluation.metrics import score_one


# ── Scorer protocol ──────────────────────────────────────────────────────


@runtime_checkable
class Scorer(Protocol):
    """Score a single rollout.

    Implementations MUST return a dict containing at least ``correct: bool``;
    other keys are merged into the persisted record (per-rollout for train,
    per-example for eval). All scorers are async to accommodate
    LLM-as-judge backends without forcing the orchestration loop to fork.
    """

    async def ascore(
        self,
        *,
        question: str,
        golden_answers: list[str],
        prediction_raw: str,
        prediction_extracted: str,
    ) -> dict: ...


# ── Concrete scorers ─────────────────────────────────────────────────────


class WordLevelScorer:
    """Token-overlap scorer: EM / CEM / F1 over normalized text.

    ``decisive_metric`` selects which bit drives the ``correct`` flag for
    pos/neg bucketing during HEK construction. Defaults to ``cem`` (Cover
    EM, the metric the HiExp paper §4.1 reports as primary).
    """

    def __init__(
        self,
        decisive_metric: Literal["em", "cem", "f1"] = "cem",
        f1_threshold: float = 0.5,
    ) -> None:
        if decisive_metric not in {"em", "cem", "f1"}:
            raise ValueError(
                f"decisive_metric must be em|cem|f1; got {decisive_metric!r}"
            )
        self.decisive_metric = decisive_metric
        self.f1_threshold = f1_threshold

    async def ascore(
        self,
        *,
        question: str,
        golden_answers: list[str],
        prediction_raw: str,
        prediction_extracted: str,
    ) -> dict:
        m = score_one(prediction_extracted, golden_answers)
        if self.decisive_metric == "em":
            correct = m.em == 1
        elif self.decisive_metric == "cem":
            correct = m.cem == 1
        else:
            correct = m.f1 >= self.f1_threshold
        return {
            "correct": bool(correct),
            "em": int(m.em),
            "cem": int(m.cem),
            "f1": float(m.f1),
        }


class JudgeScorer:
    """LLM-as-judge using BrowseComp-Plus's ``GRADER_TEMPLATE``.

    Wraps any object with an async ``ainvoke`` taking a single human
    message — i.e. ``langchain_openai.ChatOpenAI``. The prompt and parser
    come from :mod:`hiexp.evaluation.judge`, so this scorer's behavior
    is bit-identical to the standalone ``run_judge.py`` post-hoc grader.
    """

    def __init__(self, llm: Any) -> None:
        self.llm = llm

    @weave.op(name="JudgeScorer.ascore")
    async def ascore(
        self,
        *,
        question: str,
        golden_answers: list[str],
        prediction_raw: str,
        prediction_extracted: str,
    ) -> dict:
        # Lazy import to keep this module's import graph small for
        # callers that never use the judge path.
        from langchain_core.messages import HumanMessage

        gold = golden_answers[0] if golden_answers else ""
        prompt = render_grader_prompt(
            question=question, response=prediction_raw, correct_answer=gold
        )
        try:
            resp = await self.llm.ainvoke([HumanMessage(content=prompt)])
            text = resp.content if isinstance(resp.content, str) else str(resp.content)
        except Exception as exc:  # noqa: BLE001
            return {
                "correct": False,
                "judge_extracted": None,
                "judge_reasoning": None,
                "judge_confidence": None,
                "judge_response": None,
                "judge_parse_error": True,
                "judge_error": repr(exc),
            }

        parsed = parse_judge_response(text)
        return {
            "correct": bool(parsed.get("correct")) if parsed.get("correct") is not None else False,
            "judge_extracted": parsed.get("extracted_final_answer"),
            "judge_reasoning": parsed.get("reasoning"),
            "judge_confidence": parsed.get("confidence"),
            "judge_response": text,
            "judge_parse_error": bool(parsed.get("parse_error")),
        }


# ── Benchmark spec ───────────────────────────────────────────────────────


ScorerKind = Literal["word_level", "llm_judge"]


@dataclass(frozen=True)
class BenchmarkSpec:
    """The four dimensions a benchmark plug-in needs to declare."""

    name: str
    question_template: str
    extractor: Callable[[str], str]
    extractor_field: str
    scorer_kind: ScorerKind

    # ``WordLevelScorer`` knob; ignored by ``llm_judge``.
    decisive_metric: Literal["em", "cem", "f1"] = "cem"


# ── Built-in registry ────────────────────────────────────────────────────


_HOTPOTQA = BenchmarkSpec(
    name="hotpotqa",
    question_template="hiexp_boxed",
    extractor=extract_boxed_answer,
    extractor_field="prediction_boxed",
    scorer_kind="word_level",
    decisive_metric="cem",
)

_BROWSECOMP_PLUS = BenchmarkSpec(
    name="browsecomp_plus",
    question_template="browsecomp_plus",
    extractor=extract_exact_answer,
    extractor_field="prediction_exact",
    scorer_kind="llm_judge",
)


_REGISTRY: dict[str, BenchmarkSpec] = {
    _HOTPOTQA.name: _HOTPOTQA,
    _BROWSECOMP_PLUS.name: _BROWSECOMP_PLUS,
}


def get_spec(name: str) -> BenchmarkSpec:
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown benchmark {name!r}; available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]


def list_benchmarks() -> list[str]:
    return sorted(_REGISTRY)


def register(spec: BenchmarkSpec) -> None:
    """Add (or replace) a benchmark spec in the registry.

    Useful for tests and for downstream packages that want to ship a
    new benchmark without editing this file.
    """
    _REGISTRY[spec.name] = spec


# ── Scorer factory ───────────────────────────────────────────────────────


def make_scorer(
    spec: BenchmarkSpec,
    *,
    judge_llm: Any | None = None,
) -> Scorer:
    """Build the :class:`Scorer` matching ``spec.scorer_kind``.

    Args:
        spec: the benchmark spec.
        judge_llm: required when ``spec.scorer_kind == 'llm_judge'``.
            Any object exposing an async ``ainvoke`` (typically
            ``langchain_openai.ChatOpenAI``).

    Raises:
        ValueError: ``llm_judge`` requested but ``judge_llm`` is None.
    """
    if spec.scorer_kind == "word_level":
        return WordLevelScorer(decisive_metric=spec.decisive_metric)
    if spec.scorer_kind == "llm_judge":
        if judge_llm is None:
            raise ValueError(
                f"benchmark {spec.name!r} uses scorer_kind='llm_judge' but no "
                "judge_llm was provided. Configure EVAL_JUDGE_* (or "
                "AGENT_LLM_*) and pass the ChatOpenAI instance."
            )
        return JudgeScorer(llm=judge_llm)
    raise ValueError(f"unknown scorer_kind: {spec.scorer_kind!r}")


__all__ = [
    "Scorer",
    "WordLevelScorer",
    "JudgeScorer",
    "BenchmarkSpec",
    "ScorerKind",
    "get_spec",
    "list_benchmarks",
    "register",
    "make_scorer",
]
