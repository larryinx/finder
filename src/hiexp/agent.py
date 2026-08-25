"""Top-level HiExp agent.

The agent loop is intentionally narrow:
  * It runs the LangGraph state machine until the LLM emits a message
    with no tool calls, or until ``max_turn`` / ``max_search_limit`` is
    hit (Search-o1 reference defaults).
  * It emits the **raw** final response text plus a structured
    trajectory.
  * It does NOT extract a final answer (no ``\\boxed{}`` parsing, no
    ``Exact Answer:`` parsing) and does NOT compute any metric. Those
    live in ``hiexp.evaluation``.

The search tool comes from an MCP server (``AGENT_MCP_URL``); the agent
does not own a retriever. HEK retrieval (E1/E2) is HiExp-internal and
not exposed as an MCP tool.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, overload

import weave
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool

from hiexp.config import HiExpConfig

if TYPE_CHECKING:
    from hiexp.experience.schema import ExperienceHit
    from hiexp.experience.store import HEKStore

# ── Trajectory data classes ─────────────────────────────────────────────────


# Trajectory items mirror BrowseComp-Plus ``_persist_response`` output
# (``qwen_client.py:73-136``): each step is one of "reasoning",
# "tool_call", or "output_text". HiExp adds two extra item types so the
# experience-injection points are auditable separately.
TrajectoryItemType = Literal[
    "reasoning",
    "tool_call",
    "output_text",
    "e2_inject",  # HiExp-specific: experiences woven into system prompt
    "e1_inject",  # HiExp-specific: experiences appended to a tool response
]


@dataclass
class TrajectoryItem:
    type: TrajectoryItemType
    tool_name: str | None = None
    arguments: dict | None = None
    output: Any = None  # str | list[str] | list[dict]


@dataclass
class RetrievedDoc:
    """A passage returned by the MCP search tool.

    Field names mirror BrowseComp-Plus ``searcher/tools.py`` (``docid``,
    ``score``, ``snippet``) so trajectories are directly compatible with
    its evaluation runner.
    """

    docid: str
    snippet: str
    score: float | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class HiExpTrajectory:
    """Full structured trajectory of a single ``run`` call.

    Layout matches BrowseComp-Plus ``run_*.json`` so a recorded
    trajectory can be re-graded by
    ``third_party/BrowseComp-Plus/scripts_evaluation/evaluate_run.py``
    once we serialize via ``hiexp.evaluation.persist`` (M6).
    """

    query: str
    final_response: str  # raw text of the last AIMessage; NOT an extracted answer
    items: list[TrajectoryItem]
    tool_call_counts: dict[str, int]
    status: Literal["completed", "incomplete", "max_turn", "max_search"]
    retrieved_docids: list[str]


# ── Internal helpers ────────────────────────────────────────────────────────


def _last_ai_message_text(messages: list) -> str:
    """Return the visible text of the last AIMessage.

    Responses-API content is a list of blocks; only ``{"type": "text"}``
    blocks contribute to the user-facing answer. Chat-Completions content
    is a plain string.
    """
    for msg in reversed(messages):
        if not isinstance(msg, AIMessage):
            continue
        content = msg.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if not isinstance(block, dict):
                    parts.append(str(block))
                    continue
                # Only the "text" blocks are part of the visible answer.
                # Skip "reasoning", "function_call", etc.
                if block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
            return "\n".join(p for p in parts if p)
        return str(content)
    return ""


def _summarize_tool_calls(items: list[TrajectoryItem]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for it in items:
        if it.type == "tool_call" and it.tool_name:
            counts[it.tool_name] = counts.get(it.tool_name, 0) + 1
    return counts


def _collect_docids(items: list[TrajectoryItem]) -> list[str]:
    from hiexp.graph import _extract_docids

    seen: set[str] = set()
    out: list[str] = []
    for it in items:
        if it.type == "tool_call":
            for did in _extract_docids(it.output):
                if did not in seen:
                    seen.add(did)
                    out.append(did)
    return out


def _determine_status(
    cfg: HiExpConfig,
    final_state: dict,
    final_response: str,
) -> Literal["completed", "incomplete", "max_turn", "max_search"]:
    if final_state.get("turn", 0) >= cfg.max_turn:
        return "max_turn"
    if final_state.get("search_count", 0) >= cfg.max_search_limit and not final_response.strip():
        return "max_search"
    if final_response.strip():
        return "completed"
    return "incomplete"


# ── Agent ───────────────────────────────────────────────────────────────────


class HiExpAgent:
    """HiExp prompt-based agent.

    Construction is cheap; the LangGraph and HEK store are built lazily
    on the first :meth:`run` call so unit tests can patch the
    constructor's collaborators (LLM, tools, HEK) without paying any
    network cost.
    """

    def __init__(self, config: HiExpConfig | None = None) -> None:
        if config is None:
            # No-config construction = dataclass defaults for algorithmic
            # fields + endpoint env vars (AGENT_LLM_*, AGENT_MCP_URL). This
            # matches the path tests use; production code should compose
            # YAML and pass the resulting HiExpConfig explicitly.
            from omegaconf import OmegaConf

            config = HiExpConfig.from_omegaconf(OmegaConf.create({}))
        self.config = config
        self.config.validate()
        self._compiled: Any = None
        self._llm: BaseChatModel | None = None
        self._tools: list[BaseTool] | None = None
        self._hek: "HEKStore | None" = None
        # Lazily-created (so a fresh event loop after asyncio.run can
        # still bind it cleanly). When set, gates ``_async_init`` so 16
        # concurrent first-calls don't trigger 16 e5-base loads + 16
        # MCP connection setups.
        self._init_lock: asyncio.Lock | None = None

    # ── Test seam ──────────────────────────────────────────────────────────

    def setup(
        self,
        *,
        llm: BaseChatModel,
        tools: list[BaseTool],
        hek: "HEKStore | None" = None,
    ) -> None:
        """Inject the LLM, tool list, and HEK store directly.

        Used by tests to bypass MCP discovery and embedding loads. In
        production, :meth:`_lazy_init` performs the real wiring.
        """
        from hiexp.graph import build_graph

        self._llm = llm
        self._tools = tools
        self._hek = hek
        self._compiled = build_graph(self.config, llm, tools, hek)

    async def _async_init(self) -> None:
        """Build LLM + MCP tools + HEK + graph. Idempotent.

        Awaits MCP tool discovery so this is safe to call from inside an
        existing event loop (e.g. the rollout sampler's gather loop).
        Concurrency-safe: a per-instance lock guards the first init so
        N parallel ``arun`` calls don't all duplicate the e5 / MCP
        setup.
        """
        if self._compiled is not None:
            return
        if self._init_lock is None:
            self._init_lock = asyncio.Lock()
        async with self._init_lock:
            if self._compiled is not None:
                return
            await self._do_async_init()

    async def _do_async_init(self) -> None:
        from langchain_openai import ChatOpenAI

        from hiexp.experience.store import HEKStore
        from hiexp.graph import build_graph
        from hiexp.retrieval.mcp import MCPToolLoader

        tools = await MCPToolLoader(self.config).aload()

        hek: HEKStore | None = None
        if self.config.enable_hek and self.config.hek_path:
            hek = HEKStore.load(self.config.hek_path)

        # The agent supports two OpenAI-compatible endpoints, selected
        # by ``HiExpConfig.llm_api_flavor``:
        #
        # * ``responses`` (default, paper-faithful) — hits ``/v1/responses``.
        #   Reasoning text from reasoning models appears as a
        #   ``{"type": "reasoning", ...}`` block in ``AIMessage.content``,
        #   captured by our trajectory layer. Requires vLLM 0.19+ (or
        #   SGLang ≥0.4 with Responses API enabled).
        # * ``chat_completions`` — hits ``/v1/chat/completions`` for
        #   servers that don't expose Responses (older vLLM, current
        #   SGLang for some models). Reasoning lands in
        #   ``additional_kwargs.reasoning_content``; ``graph.py`` already
        #   has the fallback path for that.
        #
        # ``extra_body.chat_template_kwargs.enable_thinking`` is honored
        # by vLLM/SGLang chat templates on either endpoint, so the
        # thinking toggle works in both flavors.
        if self.config.llm_provider == "openrouter":
            # OpenRouter: the ``langchain-openrouter`` package handles
            # reasoning round-trip natively (response side fills
            # ``additional_kwargs.reasoning_content`` /
            # ``reasoning_details``; next-turn request rewrites them as
            # ``reasoning`` / ``reasoning_details`` fields on the
            # assistant message dict — see
            # ``langchain_openrouter/chat_models.py:1303-1308``).
            # ``preserve_thinking`` and our request-side injection are
            # SGLang chat-template specifics that have no analog here.
            from langchain_openrouter import ChatOpenRouter

            or_kwargs: dict[str, Any] = dict(
                model=self.config.llm_model,
                api_key=self.config.llm_api_key,
                base_url=self.config.llm_base_url,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                max_tokens=self.config.max_response_tokens,
            )
            # Map our ``enable_thinking`` knob onto OpenRouter's
            # ``reasoning`` constructor arg. ``summary=auto`` asks for a
            # reasoning summary in the response (some upstream models
            # return reasoning unconditionally; this is a hint only).
            if self.config.enable_thinking:
                or_kwargs["reasoning"] = {"summary": "auto"}
            llm = ChatOpenRouter(**or_kwargs)
        else:
            chat_template_kwargs: dict[str, Any] = {
                "enable_thinking": self.config.enable_thinking,
            }
            # Only forward ``preserve_thinking`` when both flags are on. The
            # chat-template flag has no effect (and clutters the request) if
            # the model isn't producing ``<think>`` blocks at all.
            if self.config.enable_thinking and self.config.preserve_thinking:
                chat_template_kwargs["preserve_thinking"] = True

            common_kwargs = dict(
                model=self.config.llm_model,
                base_url=self.config.llm_base_url,
                api_key=self.config.llm_api_key,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                max_tokens=self.config.max_response_tokens,
                extra_body={"chat_template_kwargs": chat_template_kwargs},
            )
            if self.config.llm_api_flavor == "responses":
                # vLLM's strict Responses-API validator rejects assistant
                # ``message`` items that LangChain round-trips without the
                # required ``status`` field (400 with a giant union dump
                # that then recursively bloats the ACE reflector prompt).
                # Install the same idempotent compat patch search_ttl uses
                # before any request is constructed.
                from hiexp.utils.responses_api_compat import (  # noqa: F401
                    _install_responses_api_status_patch,
                )

                _install_responses_api_status_patch()
                llm = ChatOpenAI(
                    **common_kwargs,
                    reasoning={"summary": "auto"},
                    output_version="responses/v1",
                )
            else:
                # Chat-completions servers (SGLang, vLLM with reasoning parser)
                # return ``reasoning_content`` as a non-standard message field.
                # ``langchain_openai`` drops it because it's not in OpenAI's
                # schema; ``ReasoningPreservingChatOpenAI`` puts it back into
                # ``additional_kwargs.reasoning_content`` (response side) and
                # — when ``preserve_thinking`` is on — re-attaches it to
                # outgoing assistant messages (request side) so SGLang's chat
                # template can rehydrate prior-turn ``<think>`` blocks.
                from hiexp.utils.reasoning_chat_openai import (
                    ReasoningPreservingChatOpenAI,
                )

                llm = ReasoningPreservingChatOpenAI(
                    **common_kwargs,
                    inject_reasoning_on_request=(
                        self.config.enable_thinking and self.config.preserve_thinking
                    ),
                )
        llm_with_tools = llm.bind_tools(tools)

        self._llm = llm_with_tools  # type: ignore[assignment]
        self._tools = tools
        self._hek = hek
        self._compiled = build_graph(self.config, llm_with_tools, tools, hek)

    def _lazy_init(self) -> None:
        if self._compiled is not None:
            return
        asyncio.run(self._async_init())

    # ── Public API ─────────────────────────────────────────────────────────

    @overload
    async def arun(self, query: str, return_trajectory: Literal[False] = False) -> str: ...

    @overload
    async def arun(self, query: str, return_trajectory: Literal[True]) -> tuple[str, HiExpTrajectory]: ...

    @weave.op(name="HiExpAgent.arun")
    async def arun(self, query: str, return_trajectory: bool = False):
        """Async equivalent of :meth:`run`.

        Use this when batching: ``asyncio.gather(*[agent.arun(q) for q in qs])``
        fires many requests at once so vLLM can batch them server-side.

        Traced under the op name ``HiExpAgent.arun`` when Weave is
        enabled — the LangChain / OpenAI SDK auto-patch then nests the
        per-LLM-call traces underneath.
        """
        await self._async_init()
        assert self._compiled is not None  # noqa: S101

        initial_state: dict[str, Any] = {
            "messages": [],
            "question": query,
            "executed_queries": [],
            "search_count": 0,
            "turn": 0,
            "trajectory_items": [],
        }
        final_state: dict = await self._compiled.ainvoke(initial_state)

        final_response = _last_ai_message_text(final_state.get("messages", []))
        if not return_trajectory:
            return final_response

        items: list[TrajectoryItem] = list(final_state.get("trajectory_items", []))
        traj = HiExpTrajectory(
            query=query,
            final_response=final_response,
            items=items,
            tool_call_counts=_summarize_tool_calls(items),
            status=_determine_status(self.config, final_state, final_response),
            retrieved_docids=_collect_docids(items),
        )
        return final_response, traj

    @overload
    def run(self, query: str, return_trajectory: Literal[False] = False) -> str: ...

    @overload
    def run(self, query: str, return_trajectory: Literal[True]) -> tuple[str, HiExpTrajectory]: ...

    def run(self, query: str, return_trajectory: bool = False):
        """Sync wrapper around :meth:`arun`.

        Returns:
            ``raw_final_response: str`` (or ``(raw, trajectory)`` when
            ``return_trajectory`` is True). Don't call from inside an
            existing event loop — use :meth:`arun` there.
        """
        return asyncio.run(self.arun(query, return_trajectory))
