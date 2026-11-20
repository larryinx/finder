"""Curator dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


OpKind = Literal["ADD", "MODIFY", "DELETE"]


@dataclass
class OpRecord:
    step: int
    op: OpKind
    section: str
    bullet_id_before: str | None
    bullet_id_after: str | None
    content_before: str | None
    content_after: str | None
    source_reflection_ids: list[str] = field(default_factory=list)
    justification: str | None = None
    react_turn_idx: int = -1


@dataclass
class CuratorStepResult:
    step: int
    ops: list[OpRecord]
    final_summary: str
    leftover_reflection_ids: list[str]
    react_turns_used: int
    rejected: bool = False
    rejection_reason: str | None = None
