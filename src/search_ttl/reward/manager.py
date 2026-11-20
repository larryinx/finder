"""Abstract RewardManager.

The return type is ``RewardReport`` (structured *text*) instead of a
token-level tensor. We never compute a scalar; the Reflector consumes prose.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from search_ttl.reward.schemas import RewardReport
from search_ttl.rollout.schemas import Trajectory


class RewardManager(ABC):
    @classmethod
    @abstractmethod
    def from_omegaconf(cls, cfg) -> "RewardManager": ...

    @abstractmethod
    async def score(self, trajectories: list[Trajectory]) -> list[RewardReport]:
        """Same length as input; one ``RewardReport`` per trajectory."""
