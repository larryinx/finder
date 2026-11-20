"""Typed state for the HiExp LangGraph agent.

Maps to ``00b_implementation_plan.md`` §4.4. Replaces Search-o1's
``active_sequences`` dict (``run_search_o1.py:377-385``) with a
``TypedDict`` so each node has a documented contract.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class HiExpState(TypedDict, total=False):
    """LangGraph state for a single agent run.

    Reducers:
        ``messages`` uses :func:`langgraph.graph.message.add_messages`
        (append + dedup-by-id).
        ``executed_queries`` and ``trajectory_items`` use
        :func:`operator.add` (list concatenation), so each node returns
        only the *new* items it produces.
        Scalars (``search_count``, ``turn``) and singletons (``question``)
        use the default reducer (overwrite).

    Fields:
        messages: rolling chat history.
        question: original user query, cached for trajectory headers.
        executed_queries: query strings already searched (dedup).
        search_count: how many distinct searches have run.
        turn: total agent loop iterations.
        trajectory_items: structured per-step records, used to build
            :class:`hiexp.agent.HiExpTrajectory` after the loop ends.
    """

    messages: Annotated[list[BaseMessage], add_messages]
    question: str
    executed_queries: Annotated[list[str], operator.add]
    search_count: int
    turn: int
    trajectory_items: Annotated[list[Any], operator.add]
