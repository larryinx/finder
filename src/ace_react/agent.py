"""ReActAgent — adapter that lets ACE drive HiExp's ReAct loop.

The wrapper does three things:

1. forces ``enable_hek=False`` (ACE provides its own context layer; HEK is
   the HiExp project's own context layer; running both at once would
   muddle the experiment design);
2. renders ACE's evolving playbook as a system-prompt suffix via
   :attr:`HiExpConfig.system_prompt_extra`, so the agent sees the
   playbook on every turn without any change to LangGraph;
3. exposes a single ``arun(question)`` returning the (raw_text,
   :class:`HiExpTrajectory`) pair downstream code already knows how to
   serialize / score.

There is no caching of agents across calls — when the playbook changes,
build a new ``ReActAgent``. The cost is one ``HiExpAgent.__init__`` plus
a LangGraph compile, both cheap; HiExp's MCP loader and HEK store stay
None on this code path.
"""

from __future__ import annotations

import dataclasses
from typing import Tuple

from hiexp import HiExpAgent, HiExpConfig
from hiexp.agent import HiExpTrajectory


_PLAYBOOK_PREAMBLE_HEADER = (
    "## ACE PLAYBOOK (curated insights from prior attempts)\n"
    "These insights were distilled from previous successful and failed\n"
    "attempts on similar problems. Apply the strategies that fit the\n"
    "current question; pay attention to entries marked as common\n"
    "failure modes. You may cite a bullet by its id (e.g. [search-00012])\n"
    "when it influenced your reasoning."
)


def render_playbook_preamble(playbook: str) -> str:
    """Wrap a raw playbook string with the ACE preamble; '' if empty.

    The empty branch matters: an empty preamble prevents ``"\\n\\n"``
    from polluting the system prompt when the controller has not yet
    produced any bullets.
    """
    text = (playbook or "").strip()
    if not text:
        return ""
    return f"{_PLAYBOOK_PREAMBLE_HEADER}\n\n{text}\n"


class ReActAgent:
    """One-shot ReAct executor with optional playbook injection.

    Args:
        base_config: a :class:`HiExpConfig`. Field overrides applied:
            ``enable_hek=False``, ``hek_path=None``,
            ``system_prompt_extra=render_playbook_preamble(playbook)``.
            All other fields (LLM model, MCP url, max_turn, etc.) flow
            through unchanged.
        playbook: ACE's playbook text. Empty/whitespace → no preamble.

    Example:
        >>> from hiexp import HiExpConfig, compose
        >>> cfg = compose(family="ace", benchmark="bcp", retriever="qwen3-8b")
        >>> base = HiExpConfig.from_omegaconf(cfg, enable_hek=False)
        >>> agent = ReActAgent(base, playbook="...")
        >>> raw, traj = await agent.arun("What is ...?")
    """

    def __init__(self, base_config: HiExpConfig, playbook: str = "") -> None:
        cfg = dataclasses.replace(
            base_config,
            enable_hek=False,
            hek_path=None,
            system_prompt_extra=render_playbook_preamble(playbook),
        )
        cfg.validate()
        self._agent = HiExpAgent(cfg)
        self._playbook_chars = len(playbook or "")

    async def arun(self, question: str) -> Tuple[str, HiExpTrajectory]:
        return await self._agent.arun(question, return_trajectory=True)

    @property
    def playbook_chars(self) -> int:
        """Length of the playbook text the agent was constructed with.

        Useful for log/dashboard sanity ("did the playbook actually
        grow turn over turn?"). Doesn't count the preamble.
        """
        return self._playbook_chars
