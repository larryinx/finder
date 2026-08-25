"""Reward: structured diagnostic *text* for the Reflector to consume.

Pluggable per agent type via ``@register_reward`` (registry.py).
"""
from search_ttl.reward.schemas import (
    QueryAnalysis,
    EfficiencyStats,
    JudgeResult,
    RewardReport,
)
from search_ttl.reward.manager import RewardManager

__all__ = [
    "QueryAnalysis",
    "EfficiencyStats",
    "JudgeResult",
    "RewardReport",
    "RewardManager",
]
