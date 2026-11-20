"""Abstract ReflectorManager — pluggable.

A reflector consumes ``(question, G trajectories, G reward reports, current
experience text)`` and emits a list of ``ReflectionOutput`` (one per
trajectory in v1; future reflectors may emit different cardinalities).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from search_ttl.reflector.schemas import ReflectionOutput
from search_ttl.reward.schemas import RewardReport
from search_ttl.rollout.schemas import QuestionRecord, Trajectory


class ReflectorManager(ABC):
    @classmethod
    @abstractmethod
    def from_omegaconf(cls, cfg) -> "ReflectorManager": ...

    @abstractmethod
    async def reflect(
        self,
        question_record: QuestionRecord,
        trajectories: list[Trajectory],
        rewards: list[RewardReport],
        experience_text: str,
        *,
        save_dir=None,
    ) -> list[ReflectionOutput]: ...
