"""HiExp: prompt-based reproduction of Hierarchical Experience for agentic search.

Top-level public API:
    HiExpAgent: the agent — runs the loop, returns raw final text + trajectory.
    HiExpConfig: typed configuration loaded from env / kwargs.
    HiExpTrajectory: structured trajectory returned when ``return_trajectory=True``.

Answer extraction and metrics live in ``hiexp.evaluation`` (deliberately
not re-exported here, to keep the agent surface clean).
"""

from hiexp.agent import (
    HiExpAgent,
    HiExpTrajectory,
    RetrievedDoc,
    TrajectoryItem,
)
from hiexp.config import HiExpConfig
from hiexp.config_compose import compose, list_options
from hiexp.experience.schema import ExperienceHit
from hiexp.tracing import (
    init_weave,
    is_enabled as is_tracing_enabled,
    log_hek_artifacts,
    log_playbook_checkpoint,
)

__all__ = [
    "HiExpAgent",
    "HiExpConfig",
    "HiExpTrajectory",
    "TrajectoryItem",
    "ExperienceHit",
    "RetrievedDoc",
    "compose",
    "list_options",
    "init_weave",
    "is_tracing_enabled",
    "log_hek_artifacts",
    "log_playbook_checkpoint",
]
