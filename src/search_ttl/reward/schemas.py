"""Reward dataclasses — see docs/stage1/plan-claude/04_reward.md §4.3."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


IntentAlignment = Literal["aligned", "drifting", "off_topic", "unclear"]
QueryQuality = Literal["good", "ok", "bad", "wasteful"]
FailureMode = Literal[
    "ok",
    "bad_query",
    "max_search",
    "max_turn",
    "parse_error",
    "hallucination",
    "off_topic",
    "error",
]


@dataclass
class QueryAnalysis:
    idx: int
    query: str
    intent_alignment: IntentAlignment
    quality: QueryQuality
    rationale: str
    n_results: int
    novel_docids: int


@dataclass
class EfficiencyStats:
    n_search: int
    n_turn: int
    hit_max_search: bool
    hit_max_turn: bool
    redundant_query_count: int
    novel_doc_ratio: float


@dataclass
class JudgeResult:
    correct: bool
    extracted_answer: str | None
    reasoning: str | None
    confidence: float | None
    parse_error: bool
    raw_response: str


@dataclass
class RewardReport:
    qid: str
    r_idx: int
    correctness: bool
    judge: JudgeResult
    per_query_analysis: list[QueryAnalysis]
    efficiency: EfficiencyStats
    failure_mode: FailureMode
    text_for_reflector: str
    extra: dict[str, Any] = field(default_factory=dict)
