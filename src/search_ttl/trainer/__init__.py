"""Trainer: main loop + registries + checkpoint."""
from search_ttl.trainer.registry import (
    REWARDS,
    REFLECTORS,
    CURATORS,
    EXPERIENCES,
    register_reward,
    register_reflector,
    register_curator,
    register_experience,
    build_reward,
    build_reflector,
    build_curator,
    build_experience,
)

__all__ = [
    "REWARDS",
    "REFLECTORS",
    "CURATORS",
    "EXPERIENCES",
    "register_reward",
    "register_reflector",
    "register_curator",
    "register_experience",
    "build_reward",
    "build_reflector",
    "build_curator",
    "build_experience",
]
