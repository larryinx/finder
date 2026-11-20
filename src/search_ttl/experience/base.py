"""Experience ABC. The Curator talks to this interface, never to a concrete backend.

Adding a new backend (skill_library, hybrid, ...) = subclass + register.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Bullet:
    bullet_id: str
    section: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SectionInfo:
    name: str
    bullet_count: int
    char_count: int


class Experience(ABC):
    """Abstract experience backend.

    Lifecycle:
        ``from_initial(cfg) → render() → ... → apply Curator ops ... → to_text()``

    The Experience is **immutable from the SearchAgent's point of view** during
    rollout (the agent only calls ``render``). The Curator mutates it via the
    ``add_bullet`` / ``modify_bullet`` / ``delete_bullet`` methods between steps.
    """

    # ── construction ──────────────────────────────────────────────────────
    @classmethod
    @abstractmethod
    def from_omegaconf(cls, cfg) -> "Experience":
        """Build from the experience config block. ``cfg.kind`` already verified."""

    @classmethod
    @abstractmethod
    def from_text(cls, text: str, *, cfg=None) -> "Experience":
        """Round-trip from a serialised string."""

    # ── read ──────────────────────────────────────────────────────────────
    @abstractmethod
    def render(self) -> str:
        """Return the string injected into the SearchAgent system prompt."""

    @abstractmethod
    def to_text(self) -> str:
        """Persisted form. May equal `render()` or be richer (metadata embedded)."""

    @abstractmethod
    def list_sections(self) -> list[SectionInfo]: ...

    @abstractmethod
    def read_section(self, section: str) -> list[Bullet]: ...

    @abstractmethod
    def read_bullet(self, bullet_id: str) -> Bullet | None: ...

    @abstractmethod
    def search_bullets(self, keyword: str, top_k: int = 5) -> list[Bullet]: ...

    @abstractmethod
    def char_count(self) -> int: ...

    # ── write (returns NEW instance — Experience is value-typed) ──────────
    @abstractmethod
    def add_bullet(
        self, section: str, content: str, *, metadata: dict[str, Any] | None = None
    ) -> tuple["Experience", Bullet]: ...

    @abstractmethod
    def modify_bullet(
        self,
        bullet_id: str,
        new_content: str,
        *,
        metadata_patch: dict[str, Any] | None = None,
    ) -> tuple["Experience", Bullet | None]: ...

    @abstractmethod
    def delete_bullet(self, bullet_id: str, *, reason: str = "") -> tuple["Experience", bool]:
        """Returns (new_experience, deleted). Deleting a missing ID is a no-op."""
