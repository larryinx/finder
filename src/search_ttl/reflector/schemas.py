"""Reflector dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


ReflectionStatus = Literal["ok", "no_useful_insight", "already_covered", "error"]


@dataclass
class ReflectionPayload:
    """The structured update a Reflector emits via `submit_reflection`.

    Sub-Agent contract: every ``status='ok'`` payload MUST cite a
    verified alternative or peer pattern in ``verification_evidence``.
    """

    status: ReflectionStatus
    scope: Literal["case", "strategy"] = "strategy"
    section_hint: str = "OTHERS"

    trigger: str = ""
    what_went_wrong_with_my_traj: str = ""
    what_peer_did_better: str = ""
    verified_alternative_query: str = ""
    verification_evidence: str = ""
    principle: str = ""
    anti_pattern: str = ""
    applies_to_future_questions_when: str = ""
    do_not_emit_if: str = ""

    confidence: float = 0.5
    notes: str = ""


@dataclass
class ReflectionOutput:
    """One Reflector's submission, ready to be queued for the Curator."""

    qid: str
    r_idx: int
    reflector_kind: str
    status: ReflectionStatus
    structured_payload: dict[str, Any]
    text_for_curator: str
    used_tool_calls: int = 0
    used_react_turns: int = 0
    transcript_path: Path | None = None
    extra: dict[str, Any] = field(default_factory=dict)
