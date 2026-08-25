"""Shared dataclass schemas for the rollout layer.

Designed to be JSON-serialisable with `dataclasses.asdict`, so any module in
the pipeline can persist intermediate state without bespoke serializers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class QuestionRecord:
    """One row from the BCP split — what the sampler hands the rollout."""

    qid: str
    question: str
    target: str
    context: str = ""  # BCP always empty; reserved for benchmarks that ship per-q context
    # Full list of accepted gold answers. BCP / BrowseComp have a single
    # canonical answer (``target`` == ``golden_answers[0]``); HotpotQA
    # and other QA benchmarks may have several aliases. The qa_em_f1
    # reward scores against this whole list (max EM/CEM/F1).
    golden_answers: list[str] = field(default_factory=list)

    @classmethod
    def from_bcp_dict(cls, d: dict[str, Any]) -> "QuestionRecord":
        golds = list(d.get("golden_answers") or [])
        target = str(d.get("target", "")) or (golds[0] if golds else "")
        # If a row only had `target`, surface it as the sole gold answer
        # so reward backends that look at golden_answers still work.
        if not golds and target:
            golds = [target]
        return cls(
            qid=str(d.get("query_id", d.get("qid", ""))),
            question=str(d["question"]),
            target=target,
            context=str(d.get("context", "")),
            golden_answers=golds,
        )


@dataclass
class QueryRecord:
    """A single search query the agent issued, masked of snippet content.

    The Reflector sees this list (not the raw trajectory) so it focuses on
    *query design* without being polluted by snippet text.
    """

    idx: int
    query: str
    preceding_thought: str  # truncated reasoning step that led to this query
    n_results: int
    top_doc_ids: list[str] = field(default_factory=list)


@dataclass
class Trajectory:
    """A single rollout — BCP-shape compatible with `runs/bcp/.../run_*.json`."""

    qid: str
    question: str
    target: str
    experience_snapshot: str   # full text of the experience used during this rollout
    experience_hash: str       # sha256:... fast lookup key
    # Full gold-answer list (populated by the rollout layer from the
    # ``QuestionRecord``). Reward backends that score multi-gold QA
    # (qa_em_f1) read from this; LLM-judge backends use ``target``.
    golden_answers: list[str] = field(default_factory=list)

    # ── BCP-compatible fields ────────────────────────────────────────────
    steps: list[dict] = field(default_factory=list)  # reasoning/tool_call/tool_result/output_text
    queries: list[QueryRecord] = field(default_factory=list)
    final_answer: str | None = None
    extracted_answer: str | None = None
    status: str = "completed"  # completed | max_turn | max_search | error
    retrieved_docids: list[str] = field(default_factory=list)
    tool_call_counts: dict[str, int] = field(default_factory=dict)
    raw_response: str = ""

    # ── pipeline metadata ────────────────────────────────────────────────
    metrics: dict[str, Any] = field(default_factory=dict)
    metadata_ttl: dict[str, Any] = field(default_factory=dict)
    r_idx: int = 0
    step: int = 0


@dataclass
class RolloutBatchSlice:
    """All G trajectories for a single question."""

    qid: str
    question: str
    target: str
    trajectories: list[Trajectory]
