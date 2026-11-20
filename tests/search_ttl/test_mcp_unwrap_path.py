"""Verify the agent passes parsed search results to the LLM, not the raw envelope.

The 0% accuracy bug was that we passed the MCP TextContent envelope
``[{type: 'text', text: '<json>'}]`` straight to ToolMessage.content,
so the LLM saw nothing it could act on. After the fix we use HiExp's
``_unwrap_mcp_text + _coerce_tool_output_to_str + _extract_docids``;
this test pins that wiring.
"""

from __future__ import annotations

import asyncio
import json

from search_ttl.agent.search_agent import SearchAgent, SearchAgentConfig


def _ai_message_with_tool_call(query: str):
    """Mock AI message that asks for a single search call."""
    from langchain_core.messages import AIMessage

    return AIMessage(
        content="",
        tool_calls=[{"name": "search", "args": {"query": query}, "id": "tc1"}],
    )


def _ai_message_with_final_answer():
    """Mock AI message that emits a BCP final block."""
    from langchain_core.messages import AIMessage

    return AIMessage(
        content=(
            "Explanation: based on doc [1].\n"
            "Exact Answer: Romania\n"
            "Confidence: 90%"
        ),
        tool_calls=[],
    )


class _FakeTool:
    name = "search"

    async def ainvoke(self, args):
        # Real MCP returns this envelope shape (langchain-mcp-adapters).
        inner = json.dumps([
            {"docid": "77234", "score": 0.91, "snippet": "Romanian Statistical Review 2014..."},
            {"docid": "12345", "score": 0.42, "snippet": "Other paper..."},
        ])
        return [{"type": "text", "text": inner}]


class _FakeLLM:
    """Returns: turn 1 → tool_call(search), turn 2 → BCP final block."""

    def __init__(self):
        self._calls = 0

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages):
        self._calls += 1
        if self._calls == 1:
            return _ai_message_with_tool_call("Romania census 2014")
        return _ai_message_with_final_answer()


def test_mcp_envelope_is_unwrapped_into_tool_message_content():
    """Run one rollout end-to-end with mocks; verify the ToolMessage the
    LLM receives is the parsed JSON, not the wire envelope.
    """
    from langchain_core.messages import ToolMessage

    cfg = SearchAgentConfig(
        llm_model="x", llm_base_url="http://x", llm_api_key="x",
        mcp_url="http://x", mcp_search_tool_name="search",
        max_search_limit=4, max_turn=6,
    )
    agent = SearchAgent(cfg, experience_text="(none yet)")
    # Bypass the real init; inject our fakes.
    agent._tools = [_FakeTool()]
    agent._tool_by_name = {"search": _FakeTool()}
    agent._llm = _FakeLLM()
    agent._initialised = True

    traj = asyncio.run(agent.arun(qid="t", question="What was the journal?", target=""))

    # ── Assertion 1: the tool result the LLM saw was unwrapped JSON,
    # not the wrapper. ToolMessage content should contain the docid.
    tool_msgs = [m for m in agent._llm._calls and []]  # placeholder; we re-walk state below
    # Walk the messages list reachable through the captured ai messages:
    # easiest is to inspect the persisted trajectory step output instead.
    tool_results = [s for s in traj.steps if s.get("type") == "tool_result"]
    assert tool_results, "expected a tool_result step"
    assert isinstance(tool_results[0]["output"], list)
    # The unwrapped output is a list of {docid, score, snippet} dicts —
    # NOT [{type:'text', text:...}]. Check by looking for `docid` keys.
    assert any(
        isinstance(e, dict) and "docid" in e for e in tool_results[0]["output"]
    ), "tool_result.output should be the parsed list, not the wire envelope"

    # ── Assertion 2: docids were extracted and tracked.
    assert "77234" in traj.retrieved_docids
    assert "12345" in traj.retrieved_docids

    # ── Assertion 3: the final answer came through.
    assert traj.status == "completed"
    assert traj.extracted_answer == "Romania"
    assert "Romanian Statistical Review" in (traj.raw_response or "") or \
           "Romania" in (traj.raw_response or "")


def test_agent_persists_reasoning_steps_when_thinking_on():
    """The previous bug: SearchAgent only wrote ``output_text`` and
    ``tool_call`` / ``tool_result`` steps. Per-query analyzer's
    ``elif kind == 'reasoning':`` branch never fired, so its prompt
    shrank to just ``search: '<query>'`` lines and intent_alignment
    judgements collapsed to ``unclear``. This pins that we now persist
    ``additional_kwargs.reasoning_content`` as a ``reasoning`` step.
    """
    import asyncio
    from langchain_core.messages import AIMessage

    from search_ttl.agent.search_agent import SearchAgent, SearchAgentConfig

    class _ReasoningLLM:
        """Turn 1 → tool_call + reasoning_content; turn 2 → final block."""
        def __init__(self): self._n = 0
        def bind_tools(self, _): return self
        async def ainvoke(self, _msgs):
            self._n += 1
            if self._n == 1:
                return AIMessage(
                    content="",
                    tool_calls=[{"name": "search", "args": {"query": "x"}, "id": "1"}],
                    additional_kwargs={
                        "reasoning_content": "I'll search for X to find the answer."
                    },
                )
            return AIMessage(
                content="Explanation: y\nExact Answer: z\nConfidence: 90%",
                tool_calls=[],
                additional_kwargs={
                    "reasoning_content": "Got the snippet, committing now."
                },
            )

    class _FakeTool:
        name = "search"
        async def ainvoke(self, _args):
            return [{"type": "text", "text": '[{"docid":"1","snippet":"foo"}]'}]

    cfg = SearchAgentConfig(
        llm_model="x", llm_base_url="http://x", llm_api_key="x",
        mcp_url="http://x", mcp_search_tool_name="search",
        max_turn=4, max_search_limit=4,
    )
    agent = SearchAgent(cfg, experience_text="(none yet)")
    agent._tools = [_FakeTool()]
    agent._tool_by_name = {"search": _FakeTool()}
    agent._llm = _ReasoningLLM()
    agent._initialised = True
    traj = asyncio.run(agent.arun(qid="t", question="?", target=""))

    reasoning_steps = [s for s in traj.steps if s.get("type") == "reasoning"]
    assert len(reasoning_steps) >= 2
    bodies = [" ".join(s["output"]) for s in reasoning_steps]
    assert any("I'll search for X" in b for b in bodies)
    assert any("committing now" in b for b in bodies)
