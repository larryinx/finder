"""Abstract CuratorManager — pluggable.

Takes a list of ``ReflectionOutput`` (across all questions in the
accumulation window) + the current ``Experience`` and returns a new
``Experience`` plus the audit trail.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from search_ttl.experience.base import Experience
from search_ttl.curator.schemas import CuratorStepResult
from search_ttl.reflector.schemas import ReflectionOutput


class CuratorManager(ABC):
    @classmethod
    @abstractmethod
    def from_omegaconf(cls, cfg) -> "CuratorManager": ...

    @abstractmethod
    async def update(
        self,
        experience: Experience,
        reflections: list[ReflectionOutput],
        *,
        step: int,
        save_dir=None,
    ) -> tuple[Experience, CuratorStepResult]: ...
