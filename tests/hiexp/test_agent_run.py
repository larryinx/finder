"""End-to-end LangGraph wiring tests with a scripted LLM and stub tools.

Avoids the network entirely:
- ``_ScriptedLLM`` returns a fixed list of ``AIMessage``s in order.
- ``_StubSearchTool`` returns canned passages.
- ``_DeterministicEmbedder`` (re-used pattern from ``test_hek_store``)
  drives a real :class:`HEKStore` without downloading the e5 model.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool

from hiexp import HiExpAgent, HiExpConfig
from hiexp.experience import ExperienceEntry, HEKStore


# ── Fake LLM ─────────────────────────────────────────────────────────────


class _ScriptedLLM:
    """Minimal chat-model substitute for graph wiring tests.

    Implements the surface ``HiExpAgent`` needs:
      - ``invoke(messages) -> AIMessage``
      - ``bind_tools(...)`` (returns self; we ignore the schema since
        the scripted messages already encode their tool calls).
    """

    def __init__(self, scripted: list[AIMessage]) -> None:
        self._scripted = list(scripted)
        self.invocations: list[list] = []

    def bind_tools(self, tools, **_):  # noqa: D401
        return self

    def invoke(self, messages, **_):
        self.invocations.append(list(messages))
        if not self._scripted:
            return AIMessage(content="(no more scripted replies)")
        return self._scripted.pop(0)


# ── Stub MCP tool ────────────────────────────────────────────────────────


class _StubSearchTool(BaseTool):
    name: str = "search"
    description: str = "stub search tool"
    canned: list[dict] = [
        {"docid": "doc-42", "score": 0.91, "snippet": "Skeleton Crew premiered in 2016."},
        {"docid": "doc-43", "score": 0.85, "snippet": "Dominique Morisseau is a 2018 MacArthur Fellow."},
    ]

    def _run(self, *args, **kwargs):  # type: ignore[override]
        return self.canned

    async def _arun(self, *args, **kwargs):  # type: ignore[override]
        return self.canned


# ── Deterministic embedder for HEK ───────────────────────────────────────


class _DetEmbedder:
    def __init__(self, dim: int = 32) -> None:
        self.dim = dim
        self._lex: dict[str, int] = {}

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        toks = (
            text.lower()
            .replace("query: ", "")
            .replace("passage: ", "")
            .split()
        )
        for w in toks:
            v[self._lex.setdefault(w, len(self._lex) % self.dim)] += 1.0
        norm = float(np.linalg.norm(v))
        if norm == 0:
            v[0] = 1.0
            norm = 1.0
        return v / norm

    def encode_passages(self, texts):
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self._vec(f"passage: {t}") for t in texts]).astype(np.float32)

    def encode_query(self, text):
        return self._vec(f"query: {text}").astype(np.float32)


def _make_hek(tmp_path) -> HEKStore:
    store = HEKStore(tmp_path, embedder=_DetEmbedder())
    store.add_entries(
        [
            ExperienceEntry(
                level=2,
                title="Multi-hop decomposition",
                description="Break the question into time then attribute.",
            ),
            ExperienceEntry(
                level=1,
                title="Avoid premiere date confusion",
                description="Premiere date is not the composition month.",
            ),
            ExperienceEntry(
                level=1,
                title="MacArthur Fellow",
                description="Cross-check the year of the fellowship.",
            ),
        ]
    )
    return store


def _make_cfg(**overrides) -> HiExpConfig:
    base = dict(
        mcp_url="http://localhost:8080/mcp",
        enable_hek=False,
        max_search_limit=3,
        max_turn=4,
    )
    base.update(overrides)
    cfg = HiExpConfig(**base)
    cfg.validate()
    return cfg


# ── Tests ────────────────────────────────────────────────────────────────


def test_run_with_no_tool_calls_returns_raw_response():
    cfg = _make_cfg()
    llm = _ScriptedLLM([AIMessage(content=r"<think>Easy.</think> The answer is \boxed{42}.")])
    agent = HiExpAgent(cfg)
    agent.setup(llm=llm, tools=[_StubSearchTool()], hek=None)

    out = agent.run("What is 6 times 7?")
    # Agent returns the *raw* final text — boxed extraction lives in
    # hiexp.evaluation, not here.
    assert isinstance(out, str)
    assert r"\boxed{42}" in out
    assert "<think>" in out


def test_run_with_tool_call_then_final_answer():
    cfg = _make_cfg()
    tool_call = {"name": "search", "args": {"query": "play written in May 2016"}, "id": "tc-1"}
    llm = _ScriptedLLM(
        [
            AIMessage(content="<think>Need to look this up.</think>", tool_calls=[tool_call]),
            AIMessage(content=r"<think>Got it.</think> The answer is \boxed{Skeleton Crew}."),
        ]
    )
    agent = HiExpAgent(cfg)
    agent.setup(llm=llm, tools=[_StubSearchTool()], hek=None)

    raw, traj = agent.run("Who wrote that play?", return_trajectory=True)

    assert r"\boxed{Skeleton Crew}" in raw
    assert traj.query == "Who wrote that play?"
    assert traj.final_response == raw
    assert traj.tool_call_counts == {"search": 1}
    assert "doc-42" in traj.retrieved_docids
    assert "doc-43" in traj.retrieved_docids
    types = [it.type for it in traj.items]
    # Order: e2_inject (none, since enable_hek=False) → reasoning → tool_call → reasoning → output_text
    assert "tool_call" in types
    assert "reasoning" in types
    assert "output_text" in types
    # Search-o1-style dedup tracker
    assert llm.invocations[0][0].content.startswith("In this environment")  # SystemMessage


def test_e2_injection_appears_in_system_prompt(tmp_path):
    cfg = _make_cfg(enable_hek=True, hek_path=str(tmp_path), e2_top_k=5)
    hek = _make_hek(tmp_path)
    llm = _ScriptedLLM([AIMessage(content=r"\boxed{ok}")])
    agent = HiExpAgent(cfg)
    agent.setup(llm=llm, tools=[_StubSearchTool()], hek=hek)

    raw, traj = agent.run("multi-hop decomposition question", return_trajectory=True)

    sys_msg_text = llm.invocations[0][0].content
    # E2 description must show up in the system message.
    assert "Break the question into time then attribute." in sys_msg_text
    # And the trajectory records the E2 injection separately.
    e2_items = [it for it in traj.items if it.type == "e2_inject"]
    assert len(e2_items) == 1
    assert isinstance(e2_items[0].output, list)
    assert len(e2_items[0].output) >= 1


def test_e1_injection_prefixes_tool_response(tmp_path):
    cfg = _make_cfg(
        enable_hek=True,
        hek_path=str(tmp_path),
        e1_threshold=0.0,  # disable threshold so any hit makes it through
        e1_top_k=2,
    )
    hek = _make_hek(tmp_path)
    tool_call = {
        "name": "search",
        "args": {"query": "Avoid premiere date confusion"},
        "id": "tc-1",
    }
    llm = _ScriptedLLM(
        [
            AIMessage(content="<think>hmm</think>", tool_calls=[tool_call]),
            AIMessage(content=r"\boxed{done}"),
        ]
    )
    agent = HiExpAgent(cfg)
    agent.setup(llm=llm, tools=[_StubSearchTool()], hek=hek)

    raw, traj = agent.run("ask something", return_trajectory=True)

    # Second LLM invocation receives the ToolMessage with E1 prefix
    second_call_messages = llm.invocations[1]
    tool_msg = next(m for m in second_call_messages if m.__class__.__name__ == "ToolMessage")
    assert "[Trigger E_1 Experience:" in tool_msg.content
    assert "Premiere date is not the composition month." in tool_msg.content
    # And the trajectory has e1_inject items.
    e1_items = [it for it in traj.items if it.type == "e1_inject"]
    assert len(e1_items) == 1


def test_dedup_short_circuits_repeat_query():
    cfg = _make_cfg(max_search_limit=5)
    repeat_call = {"name": "search", "args": {"query": "same query"}, "id": "tc-1"}
    repeat_call_2 = {"name": "search", "args": {"query": "same query"}, "id": "tc-2"}
    llm = _ScriptedLLM(
        [
            AIMessage(content="", tool_calls=[repeat_call]),
            AIMessage(content="", tool_calls=[repeat_call_2]),
            AIMessage(content=r"\boxed{stop}"),
        ]
    )
    tool = _StubSearchTool()
    agent = HiExpAgent(cfg)
    agent.setup(llm=llm, tools=[tool], hek=None)

    raw, traj = agent.run("dedup test", return_trajectory=True)

    # The second tool_call message should be the dedup stub, not the
    # canned snippets.
    tool_messages = [
        m
        for invocation in llm.invocations
        for m in invocation
        if m.__class__.__name__ == "ToolMessage"
    ]
    assert any("Please refer to previous results" in m.content for m in tool_messages)


def test_max_search_limit_short_circuits():
    cfg = _make_cfg(max_search_limit=1, max_turn=10)
    calls = [
        {"name": "search", "args": {"query": f"q{i}"}, "id": f"tc-{i}"} for i in range(4)
    ]
    llm = _ScriptedLLM(
        [
            AIMessage(content="", tool_calls=[calls[0]]),
            AIMessage(content="", tool_calls=[calls[1]]),
            AIMessage(content=r"\boxed{cap}"),
        ]
    )
    agent = HiExpAgent(cfg)
    agent.setup(llm=llm, tools=[_StubSearchTool()], hek=None)
    raw, traj = agent.run("many searches", return_trajectory=True)
    tool_messages = [
        m
        for invocation in llm.invocations
        for m in invocation
        if m.__class__.__name__ == "ToolMessage"
    ]
    assert any("maximum search limit is exceeded" in m.content for m in tool_messages)


def test_max_turn_terminates_loop():
    cfg = _make_cfg(max_turn=2, max_search_limit=10)
    # If max_turn weren't enforced this would loop forever with tool_calls.
    forever_call = {"name": "search", "args": {"query": "loop"}, "id": "tc-x"}
    llm = _ScriptedLLM(
        [AIMessage(content="", tool_calls=[forever_call]) for _ in range(20)]
    )
    agent = HiExpAgent(cfg)
    agent.setup(llm=llm, tools=[_StubSearchTool()], hek=None)
    raw, traj = agent.run("infinite loop check", return_trajectory=True)
    assert traj.status in {"max_turn", "incomplete"}


def test_unknown_tool_returns_stub_message():
    cfg = _make_cfg()
    bad_call = {"name": "nonexistent", "args": {}, "id": "tc-1"}
    llm = _ScriptedLLM(
        [
            AIMessage(content="", tool_calls=[bad_call]),
            AIMessage(content=r"\boxed{recovered}"),
        ]
    )
    agent = HiExpAgent(cfg)
    agent.setup(llm=llm, tools=[_StubSearchTool()], hek=None)
    raw, traj = agent.run("unknown tool", return_trajectory=True)
    tool_calls = [it for it in traj.items if it.type == "tool_call"]
    assert any("not available" in str(it.output) for it in tool_calls)
