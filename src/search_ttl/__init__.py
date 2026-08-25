"""TTL-Search: training-free GRPO over a search-agent's experience playbook.

The package layout mirrors verl's decoupling pattern (trainer / rollout /
reward / algorithm / config), but every "weight update" is replaced with an
LLM agent that reflects on rollouts and edits a markdown playbook.
"""

from __future__ import annotations

__version__ = "0.1.0"

from search_ttl.trainer.registry import (
    REWARDS,
    REFLECTORS,
    CURATORS,
    EXPERIENCES,
    register_reward,
    register_reflector,
    register_curator,
    register_experience,
)

__all__ = [
    "__version__",
    "REWARDS",
    "REFLECTORS",
    "CURATORS",
    "EXPERIENCES",
    "register_reward",
    "register_reflector",
    "register_curator",
    "register_experience",
]
