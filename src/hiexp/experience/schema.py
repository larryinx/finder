"""HEK data model.

Maps 1:1 to the JSON schema in HiExp paper Table 11 (contrastive
distillation prompt) and Table 12 (clustering prompt).

Field semantics, per paper §3.1.1:
    > "the 'description' field corresponds to e_i and captures detailed
    > experiential knowledge, whereas the 'title' field corresponds to
    > d_i and functions as the retrieval key."
"""

from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, Field, model_validator

ExperienceLevel = Literal[1, 2, 3]


class ExperienceEntry(BaseModel):
    """A single HEK entry. Matches paper Table 11 schema, plus ``level``
    (E1 / E2 / E3) and ``id`` for reproducible referencing."""

    type: str = ""
    title: str
    """One-sentence summary; used as the retrieval key (paper §3.1.1, ``d_i``)."""

    tags: list[str] = Field(default_factory=list)
    description: str
    """Detailed analysis; injected as the experience (paper §3.1.1, ``e_i``)."""

    thinking: str = ""
    level: ExperienceLevel
    source_ids: list[str] = Field(default_factory=list)
    """For E2/E3: ids of the lower-level experiences this entry summarizes."""

    id: str = ""
    """Stable id; auto-derived from ``(level, title)`` if blank."""

    @model_validator(mode="after")
    def _default_id(self) -> "ExperienceEntry":
        if not self.id:
            # Hash on (level, title) — title is the retrieval key, so
            # this is stable across re-runs as long as the title is
            # unique within a level (parent-child invariant).
            digest = hashlib.sha1(
                f"{self.level}|{self.title}".encode("utf-8")
            ).hexdigest()[:12]
            self.id = f"E{self.level}-{digest}"
        return self

    def to_jsonl(self) -> str:
        return json.dumps(self.model_dump(), ensure_ascii=False)

    @classmethod
    def from_jsonl(cls, line: str) -> "ExperienceEntry":
        return cls.model_validate_json(line)


class ExperienceHit(BaseModel):
    """A retrieved entry plus its similarity score."""

    entry: ExperienceEntry
    score: float

    @property
    def title(self) -> str:
        return self.entry.title

    @property
    def description(self) -> str:
        return self.entry.description

    @property
    def level(self) -> int:
        return self.entry.level
