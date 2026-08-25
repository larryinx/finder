"""SearchAgent — minimal LangGraph ReAct loop.

We borrow the three-node skeleton (init → agent → tool → agent → END) from
``src/hiexp/graph.py`` but strip the HEK injection branches because TTL
treats experience as a single opaque string. The reasoning-content
preservation logic comes from ``src/hiexp/utils/reasoning_chat_openai.py``
because reproducing it would just duplicate hard-won bug fixes.

Public surface:

    cfg = SearchAgentConfig(...)
    agent = SearchAgent(cfg, experience_text="...")
    traj  = await agent.arun(qid="q1", question="...", target="...")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from search_ttl.agent.extract import attach_query_records, extract_answer_for_template
from search_ttl.agent.prompts import render_system_prompt, render_user_prompt
from search_ttl.rollout.schemas import Trajectory
from search_ttl.utils.persistence import hash_text
from search_ttl.utils.tracing import traceable

LOG = logging.getLogger(__name__)


@dataclass
class SearchAgentConfig:
    """Knobs the SearchAgent needs. Kept independent from HiExpConfig.

    `from_omegaconf` accepts the ``agent`` block from `config/ttl/base.yaml`.
    """

    llm_model: str
    llm_base_url: str
    llm_api_key: str
    # ``openai_compat`` (default, any OpenAI-compatible server) or
    # ``openrouter`` (langchain-openrouter against openrouter.ai).
    # Mirrors HiExp's AGENT_LLM_PROVIDER knob.
    llm_provider: str = "openai_compat"
    llm_api_flavor: str = "chat_completions"  # chat_completions | responses
    temperature: float = 1.0
    top_p: float = 0.95
    max_response_tokens: int = 8192
    enable_thinking: bool = True
    preserve_thinking: bool = True
    # Which tool backend the agent talks to. ``mcp`` is the BCP path
    # (MCPToolBuilder); ``serper`` swaps in GoogleSerperAPIWrapper and
    # ``ddg`` DuckDuckGoSearchAPIWrapper — both for open-web benchmarks
    # like BrowseComp that don't ship a vetted corpus.
    tool_provider: str = "mcp"
    mcp_url: str = ""
    mcp_search_tool_name: str = "search"
    mcp_transport: str = "streamable_http"   # streamable_http | streamable-http | sse
    # Serper knobs (only read when ``tool_provider == 'serper'``).
    # Stored as a free-form dict so the builder can pick what it needs;
    # the OmegaConf side is ``agent.serper.{api_key,num_results,gl,hl,search_type}``.
    serper: dict = field(default_factory=dict)
    # DuckDuckGo knobs (only read when ``tool_provider == 'ddg'``).
    # OmegaConf side: ``agent.ddg.{max_results,region,safesearch,backend,...}``.
    ddg: dict = field(default_factory=dict)
    # Which user-prompt template to render: ``bcp`` (docid-citation
    # version, default) or ``browsecomp`` (no-docid version, matching
    # the official BrowseComp QUERY_TEMPLATE).
    user_prompt_template: str = "bcp"
    max_search_limit: int = 30
    max_turn: int = 30
    timeout_s: float = 600.0
    # Hard cap on a single tool call (default 90s). On timeout or
    # transient connection error the MCP builder reconnects once and
    # retries; the Serper builder just propagates the error.
    tool_call_timeout_s: float = 90.0
    # AgentIR-style reasoning-aware retrieval. When true the query sent
    # to the search tool is ``Reasoning: ...\n\nQuery: ...``; the
    # LLM context is unchanged. See ``_run_tools``.
    reasoning_aware_retrieval: bool = False

    @classmethod
    def from_omegaconf(cls, cfg) -> "SearchAgentConfig":
        def _subdict(key: str) -> dict:
            node = cfg.get(key, None)
            if node is None:
                return {}
            try:
                from omegaconf import OmegaConf

                return OmegaConf.to_container(node, resolve=True) or {}
            except Exception:  # noqa: BLE001
                return dict(node)

        serper_dict = _subdict("serper")
        ddg_dict = _subdict("ddg")
        return cls(
            llm_model=cfg.llm_model,
            llm_base_url=cfg.llm_base_url,
            llm_api_key=cfg.llm_api_key,
            llm_provider=cfg.get("llm_provider", "openai_compat"),
            llm_api_flavor=cfg.get("llm_api_flavor", "chat_completions"),
            temperature=float(cfg.get("temperature", 1.0)),
            top_p=float(cfg.get("top_p", 0.95)),
            max_response_tokens=int(cfg.get("max_response_tokens", 8192)),
            enable_thinking=bool(cfg.get("enable_thinking", True)),
            preserve_thinking=bool(cfg.get("preserve_thinking", True)),
            tool_provider=str(cfg.get("tool_provider", "mcp")),
            mcp_url=cfg.get("mcp_url", ""),
            mcp_search_tool_name=cfg.get("mcp_search_tool_name", "search"),
            mcp_transport=str(cfg.get("mcp_transport", "streamable_http")),
            serper=serper_dict,
            ddg=ddg_dict,
            user_prompt_template=str(cfg.get("user_prompt_template", "bcp")),
            max_search_limit=int(cfg.get("max_search_limit", 30)),
            max_turn=int(cfg.get("max_turn", 30)),
            timeout_s=float(cfg.get("timeout_s", 600.0)),
            tool_call_timeout_s=float(cfg.get("tool_call_timeout_s", 90.0)),
            reasoning_aware_retrieval=bool(
                cfg.get("reasoning_aware_retrieval", False)
            ),
        )


@dataclass
class _AgentState:
    """In-memory state for one rollout. Not persisted directly."""

    qid: str
    question: str
    target: str
    messages: list = field(default_factory=list)
    steps: list[dict] = field(default_factory=list)
    retrieved_docids: list[str] = field(default_factory=list)
    search_count: int = 0
    turn: int = 0
    status: str = "running"
    raw_response: str = ""
    # Thinking emitted on the CURRENT turn (before this turn's tool
    # call). Refreshed every turn; consumed by reasoning-aware
    # retrieval. "" when the model produced no reasoning this turn.
    last_reasoning: str = ""


class SearchAgent:
    """One LangGraph instance per (config, experience) pair.

    Stateful across calls in the sense that the LLM client + MCP tools are
    cached, but each ``arun`` call returns its own fresh trajectory.
    """

    def __init__(self, config: SearchAgentConfig, experience_text: str = "") -> None:
        self.config = config
        self.experience_text = experience_text or ""
        self._llm = None  # bound on first arun
        self._tools: list = []
        self._tool_by_name: dict = {}
        self._initialised = False
        # Lazily built on first connect. Holds provider-specific
        # (re)load + postprocess hooks; see ``tool_builders.py``.
        from search_ttl.agent.tool_builders import build_tool_builder

        self._tool_builder = build_tool_builder(config)

    # ── lazy init ─────────────────────────────────────────────────────────
    async def _async_init(self) -> None:
        if self._initialised:
            return
        await self._connect_tools()
        self._llm = self._build_llm().bind_tools(self._tools)
        self._initialised = True

    async def _connect_tools(self) -> None:
        """(Re)build the tool table. Used at init AND when a transient
        SSE failure forces a reconnect mid-run.
        """
        self._tools = await self._tool_builder.reconnect()
        self._tool_by_name = {t.name: t for t in self._tools}
        wanted = self.config.mcp_search_tool_name
        if wanted not in self._tool_by_name:
            raise RuntimeError(
                f"search tool {wanted!r} not found via "
                f"{self.config.tool_provider!r} provider; "
                f"got {list(self._tool_by_name)}"
            )

    def _build_llm(self):
        # Delegate to the shared builder so all ReAct loops (agent,
        # reflector, curator) handle ``llm_api_flavor`` /
        # ``preserve_thinking`` / ``enable_thinking`` identically.
        from search_ttl.utils.llm_builder import build_chat_llm

        return build_chat_llm(
            model=self.config.llm_model,
            base_url=self.config.llm_base_url,
            api_key=self.config.llm_api_key,
            llm_provider=self.config.llm_provider,
            llm_api_flavor=self.config.llm_api_flavor,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=self.config.max_response_tokens,
            enable_thinking=self.config.enable_thinking,
            preserve_thinking=self.config.preserve_thinking,
            timeout_s=self.config.timeout_s,
        )

    # ── public ────────────────────────────────────────────────────────────
    @traceable(name="SearchAgent.arun", run_type="chain")
    async def arun(self, qid: str, question: str, target: str = "") -> Trajectory:
        await self._async_init()

        from langchain_core.messages import HumanMessage, SystemMessage

        state = _AgentState(
            qid=qid,
            question=question,
            target=target,
            messages=[
                SystemMessage(content=render_system_prompt(self.experience_text)),
                HumanMessage(
                    content=render_user_prompt(
                        question, template=self.config.user_prompt_template
                    )
                ),
            ],
        )

        while state.status == "running":
            state.turn += 1
            if state.turn > self.config.max_turn:
                state.status = "max_turn"
                break

            try:
                ai = await self._llm.ainvoke(state.messages)  # type: ignore[union-attr]
            except Exception as exc:
                LOG.warning("LLM error on qid=%s turn=%s: %s", qid, state.turn, exc)
                state.status = "error"
                state.raw_response = f"[llm_error] {type(exc).__name__}: {exc}"
                break

            # ① Promote ``mcp_call`` content blocks into ``ai.tool_calls``
            #    if LangChain didn't already (vLLM gpt-oss may emit
            #    ONLY ``mcp_call`` with no parallel ``function_call``,
            #    in which case LangChain leaves ``ai.tool_calls`` empty
            #    and our loop would silently see "no tool call").
            # ② Strip the now-redundant ``mcp_call`` etc. blocks so
            #    vLLM's harmony serializer doesn't 400 on the next
            #    turn's ``input`` echo.
            from search_ttl.utils.tool_call_recovery import (
                recover_mcp_call_tool_calls,
                sanitize_responses_message_for_round_trip,
            )
            mcp_recovered = recover_mcp_call_tool_calls(ai)
            if mcp_recovered:
                LOG.info(
                    "agent qid=%s turn=%s: promoted %d mcp_call block(s) "
                    "into ai.tool_calls",
                    qid, state.turn, len(mcp_recovered),
                )
            n_stripped = sanitize_responses_message_for_round_trip(ai)
            if n_stripped:
                LOG.info(
                    "agent qid=%s turn=%s: stripped %d vLLM-rejected "
                    "content block(s) before round-trip",
                    qid, state.turn, n_stripped,
                )
            state.messages.append(ai)
            tool_calls = getattr(ai, "tool_calls", None) or []
            text_content = self._extract_text(ai)

            # ── Persist reasoning_content as a `reasoning` step (parity
            # with hiexp/graph.py:agent_node:369-377). Without this, the
            # trajectory has only `tool_call` / `tool_result` / `output_text`
            # steps, and downstream prompts that show "reasoning before
            # each query" (per-query analyzer, reflector preceding_thought)
            # see empty strings. The analyzer in particular hard-codes
            # ``elif kind == 'reasoning':`` so without this branch its
            # whole input shrinks to "search: 'query'  results: 5 docs ...".
            ak = getattr(ai, "additional_kwargs", {}) or {}
            ak_reasoning = ak.get("reasoning_content") or ak.get("reasoning") or ""
            if isinstance(ak_reasoning, str):
                ak_reasoning = ak_reasoning.strip()
            else:
                ak_reasoning = ""
            inline_reasoning = _extract_inline_thinking(text_content)
            # Responses API (``output_version='responses/v1'``, e.g.
            # gpt-oss): reasoning is NOT in ``additional_kwargs`` and is
            # NOT an inline ``<think>`` block. It arrives as a typed
            # ``{"type": "reasoning"}`` content block inside the list
            # ``ai.content``. Without this extraction the trajectory has
            # zero ``reasoning`` steps under the Responses API while
            # chat_completions (which populates ``reasoning_content``)
            # works — exactly the asymmetry observed.
            responses_reasoning = _extract_responses_reasoning(ai)
            reasoning_str = ak_reasoning or inline_reasoning or responses_reasoning
            # Refresh every turn (even when empty) so reasoning-aware
            # retrieval uses THIS turn's thinking, never a stale prior
            # turn's. Matches AgentIR's per-call ``current_thinking``.
            state.last_reasoning = reasoning_str
            if reasoning_str:
                state.steps.append(
                    {"type": "reasoning", "output": [reasoning_str]}
                )

            # When the LLM endpoint isn't configured with a tool-call
            # parser, different model families leak their tool-call
            # shape into ``content`` as raw text and ``ai.tool_calls``
            # stays empty:
            #   - Qwen3 / Hermes: <tool_call>{...}</tool_call>
            #   - gpt-oss (Harmony): <|start|>assistant<|channel|>...
            #     to=functions.X<|message|>{...}<|call|>
            # Without recovery our loop sees "no tool_calls" and
            # completes with an empty answer on turn 1.
            if not tool_calls and isinstance(ai.content, str):
                from search_ttl.utils.tool_call_recovery import recover_text_tool_calls

                recovered = recover_text_tool_calls(ai.content)
                if recovered:
                    LOG.info(
                        "agent qid=%s turn=%s: recovered %d tool_call(s) from text",
                        qid, state.turn, len(recovered),
                    )
                    ai.tool_calls = recovered
                    tool_calls = recovered

            # Last-resort: the model emitted a tool call as ``{"query": ...}``
            # JSON inside the analysis channel only (``reasoning_content``).
            # Happens with gpt-oss-120b on OpenRouter/DeepInfra — see
            # `tool_call_recovery._NAKED_QUERY_JSON_RE`. Without this, the
            # loop hits the "no tool_calls + no visible content" branch
            # below, nudges once, and quits with an empty answer.
            if not tool_calls:
                ak_for_recovery = getattr(ai, "additional_kwargs", {}) or {}
                rc_for_recovery = (
                    ak_for_recovery.get("reasoning_content")
                    or ak_for_recovery.get("reasoning")
                    or ""
                )
                if isinstance(rc_for_recovery, str) and rc_for_recovery:
                    from search_ttl.utils.tool_call_recovery import (
                        recover_naked_query_json,
                        recover_text_tool_calls,
                    )
                    recovered = (
                        recover_text_tool_calls(rc_for_recovery)
                        or recover_naked_query_json(
                            rc_for_recovery,
                            tool_name=self.config.mcp_search_tool_name,
                        )
                    )
                    if recovered:
                        LOG.info(
                            "agent qid=%s turn=%s: recovered %d tool_call(s) from reasoning_content",
                            qid, state.turn, len(recovered),
                        )
                        ai.tool_calls = recovered
                        tool_calls = recovered

            # Pull reasoning_content if present (SGLang/vLLM with the
            # reasoning parser put <think> blocks here; visible answers
            # land in `content`). We only use it as a fallback for
            # `prediction_raw` when `content` is empty AND the reasoning
            # already contains the BCP final block.
            ak = getattr(ai, "additional_kwargs", {}) or {}
            reasoning = ak.get("reasoning_content") or ak.get("reasoning") or ""
            if isinstance(reasoning, str):
                reasoning = reasoning.strip()
            else:
                reasoning = ""
            # Same Responses-API fallback as the reasoning-step capture
            # above: when the model emitted the final answer only inside
            # a responses/v1 ``reasoning`` block, recover it here too.
            reasoning = reasoning or responses_reasoning

            if text_content:
                state.steps.append({"type": "output_text", "output": [text_content]})
                state.raw_response = text_content
            elif reasoning and _looks_like_final_answer(reasoning):
                # Last-resort: model put the answer in `<think>` only.
                state.steps.append({"type": "output_text", "output": [reasoning]})
                state.raw_response = reasoning

            # Treat "produced a BCP-format final block" as a stop signal
            # even if the LLM also emitted a tool call alongside it. Some
            # Qwen3 checkpoints return both; without this check we'd run
            # the extra tool call and overwrite raw_response next turn.
            if state.raw_response and _looks_like_final_answer(state.raw_response):
                state.status = "completed"
                break

            if not tool_calls:
                # Empty content + no tool calls. Two sub-cases:
                #   (a) genuine model giveup — accept as completed.
                #   (b) model is mid-thinking but emitted nothing visible;
                #       happens with thinking-on. Nudge once before bailing.
                if not text_content and not state.raw_response:
                    if not getattr(state, "_nudged", False):
                        from langchain_core.messages import HumanMessage

                        state.messages.append(
                            HumanMessage(
                                content=(
                                    "You produced no visible output and made no tool call. "
                                    "Either call `search` with a new query, or commit to "
                                    "your final answer in this exact format as your visible "
                                    "message text:\n\n"
                                    "Explanation: ...\nExact Answer: ...\nConfidence: 0–100%"
                                )
                            )
                        )
                        # store on the state object even though dataclass is frozen-ish
                        state._nudged = True  # type: ignore[attr-defined]
                        continue
                state.status = "completed"
                break

            await self._run_tools(state, tool_calls)
            if state.status != "running":
                break

        traj = self._finalise(state)
        attach_query_records(traj)
        return traj

    # ── helpers ───────────────────────────────────────────────────────────
    def _extract_text(self, ai_message) -> str:
        content = getattr(ai_message, "content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for c in content:
                if isinstance(c, dict) and c.get("type") in {"text", "output_text"}:
                    parts.append(str(c.get("text") or c.get("content") or ""))
                elif isinstance(c, str):
                    parts.append(c)
            return "\n".join(p for p in parts if p)
        return ""

    async def _run_tools(self, state: _AgentState, tool_calls) -> None:
        from langchain_core.messages import ToolMessage

        # Per-provider unwrap / docid extraction lives on the
        # ToolBuilder so the agent stays provider-agnostic. The MCP
        # builder reuses HiExp's envelope unwrapper to keep the BCP
        # path byte-identical to the pre-refactor behaviour.
        for tc in tool_calls:
            name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "")
            args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {}) or {}
            tool_call_id = (
                tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", "") or ""
            )

            if state.search_count >= self.config.max_search_limit:
                state.status = "max_search"
                state.messages.append(
                    ToolMessage(
                        content="(search budget exhausted; please answer with what you have.)",
                        tool_call_id=tool_call_id,
                    )
                )
                break

            tool = self._tool_by_name.get(name)
            if tool is None:
                state.messages.append(
                    ToolMessage(
                        content=f"unknown tool {name!r}",
                        tool_call_id=tool_call_id,
                    )
                )
                continue

            # Defensive shape-check before invoking the MCP tool: if the
            # LLM emits a `search` call with NO `query` field, the FastMCP
            # server returns a long pydantic ValidationError that bubbles
            # all the way back. Catch the obvious case here and return a
            # short hint so the LLM corrects on next turn.
            if (
                name == self.config.mcp_search_tool_name
                and isinstance(args, dict)
                and "query" not in args
            ):
                LOG.info(
                    "agent qid=%s turn=%s: search called without 'query' kwarg "
                    "(args keys=%s) — returning hint",
                    state.qid, state.turn, list(args.keys()),
                )
                state.messages.append(
                    ToolMessage(
                        content=(
                            "ERROR: `search` requires a single `query` argument, "
                            "e.g. search(query='your search string'). The args you "
                            f"passed were {list(args.keys())} — none of those are "
                            "valid. Try again with `query=...`."
                        ),
                        tool_call_id=tool_call_id,
                    )
                )
                continue

            # Record the ORIGINAL args (clean query). The trajectory,
            # `attach_query_records`, reflector and reward all read this
            # — they must see the model's real query, not the augmented
            # one. The LLM's own context is the unmodified ``ai`` message
            # appended earlier, so it likewise never sees the fusion.
            state.steps.append({"type": "tool_call", "tool_name": name, "arguments": args})

            # AgentIR-style reasoning-aware retrieval: mutate ONLY the
            # value handed to ``ainvoke`` (i.e. what the MCP server's
            # encoder sees). The BCP search tool returns just
            # [{docid,score,snippet}] and never echoes the query, so the
            # fused string cannot leak back into the conversation —
            # context stays byte-identical to the disabled case.
            invoke_args = args
            if (
                self.config.reasoning_aware_retrieval
                and name == self.config.mcp_search_tool_name
                and isinstance(args, dict)
                and isinstance(args.get("query"), str)
            ):
                invoke_args = {
                    **args,
                    "query": _format_reasoning_query(
                        args["query"], state.last_reasoning
                    ),
                }
                LOG.info(
                    "agent qid=%s turn=%s: reasoning-aware retrieval ON "
                    "(orig query=%r, reasoning_chars=%d)",
                    state.qid, state.turn, args["query"],
                    len(state.last_reasoning or ""),
                )

            try:
                from search_ttl.agent.mcp_call import (
                    call_with_timeout_and_reconnect,
                )

                # We re-resolve the tool from the table inside the closure
                # so a reconnect (which rebuilds ``self._tool_by_name``)
                # picks up the freshly-rebound StructuredTool, not the
                # dead one we captured before. ``invoke_args`` is the
                # (possibly reasoning-fused) payload; it is consumed
                # before the loop advances, so the closure capture is
                # safe (same pattern the original ``args`` capture used).
                result = await call_with_timeout_and_reconnect(
                    invoke_fn=lambda: self._tool_by_name[name].ainvoke(invoke_args),
                    reconnect_fn=self._connect_tools,
                    timeout_s=self.config.tool_call_timeout_s,
                    label=f"agent.{name}",
                )
            except Exception as exc:
                LOG.warning("tool %s failed on qid=%s: %s", name, state.qid, exc)
                state.messages.append(
                    ToolMessage(
                        content=f"[tool_error] {type(exc).__name__}: {exc}",
                        tool_call_id=tool_call_id,
                    )
                )
                continue

            pp = self._tool_builder.postprocess(result)
            for did in pp.docids:
                if did not in state.retrieved_docids:
                    state.retrieved_docids.append(did)

            state.steps.append(
                {"type": "tool_result", "tool_name": name, "output": pp.persisted_output}
            )
            state.search_count += 1
            state.messages.append(
                ToolMessage(content=pp.tool_str, tool_call_id=tool_call_id)
            )

    def _finalise(self, state: _AgentState) -> Trajectory:
        # Tool-call counts (only one tool in v1, but keep dict-shaped for BCP compat).
        tool_counts: dict[str, int] = {}
        for s in state.steps:
            if s.get("type") == "tool_call":
                n = s.get("tool_name", "search")
                tool_counts[n] = tool_counts.get(n, 0) + 1

        extracted = (
            extract_answer_for_template(
                state.raw_response, self.config.user_prompt_template
            )
            if state.raw_response
            else None
        )

        return Trajectory(
            qid=state.qid,
            question=state.question,
            target=state.target,
            experience_snapshot=self.experience_text,
            experience_hash=hash_text(self.experience_text),
            steps=state.steps,
            queries=[],  # filled by attach_query_records
            final_answer=state.raw_response,
            extracted_answer=extracted,
            status=state.status,
            retrieved_docids=state.retrieved_docids,
            tool_call_counts=tool_counts,
            raw_response=state.raw_response,
            metrics={"search_count": state.search_count, "turns": state.turn},
        )


def _format_reasoning_query(query: str, reasoning: str) -> str:
    """Fuse the agent's current reasoning with its query for the retriever.

    Byte-for-byte mirror of AgentIR's
    ``related/AgentIR/searcher/prompts.py: format_reasoning_query`` —
    empty reasoning becomes the literal ``"Empty"`` (not dropped), so
    the retriever input distribution matches AgentIR exactly.
    """
    r = reasoning if reasoning else "Empty"
    return f"Reasoning: {r}\n\nQuery: {query}"


def _looks_like_final_answer(text: str) -> bool:
    """Heuristic: does this text contain a recognised final-answer marker?

    Two markers, matching the two user-prompt templates we ship:

    * BCP / BrowseComp: ``Exact Answer: ...\\nConfidence: ...``
    * HiExp-boxed:      ``\\boxed{...}``

    Used to stop the agent loop early when the model emitted both a
    final answer and a tool call on the same turn — otherwise we'd
    overwrite ``raw_response`` on the next turn.
    """
    if not text:
        return False
    lower = text.lower()
    if "exact answer:" in lower and "confidence:" in lower:
        return True
    if "\\boxed{" in text:
        return True
    return False


def _extract_responses_reasoning(ai_message) -> str:
    """Pull reasoning text out of Responses-API content blocks.

    Under ``output_version='responses/v1'`` (vLLM/SGLang gpt-oss,
    OpenAI Responses) ``ai.content`` is a list and reasoning is a
    ``{"type": "reasoning"}`` block. Two sub-shapes occur depending on
    whether the server emits full reasoning or only a summary:

      {"type": "reasoning",
       "content": [{"type": "reasoning_text", "text": "..."}],
       "summary": [{"type": "summary_text", "text": "..."}]}

    Either (or both) of ``content`` / ``summary`` may carry the text;
    older shapes put it directly in ``block["text"]``. We concatenate
    every text fragment we find across all reasoning blocks. Returns ""
    when there is no reasoning block (e.g. chat_completions, where the
    text lives in ``additional_kwargs`` instead).
    """
    content = getattr(ai_message, "content", None)
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "reasoning":
            continue
        for key in ("content", "summary"):
            seq = block.get(key)
            if isinstance(seq, list):
                for item in seq:
                    if isinstance(item, dict):
                        t = item.get("text")
                        if isinstance(t, str) and t.strip():
                            parts.append(t.strip())
                    elif isinstance(item, str) and item.strip():
                        parts.append(item.strip())
        direct = block.get("text")
        if isinstance(direct, str) and direct.strip():
            parts.append(direct.strip())
    return "\n".join(parts).strip()


def _extract_inline_thinking(content: str) -> str:
    """Pull a single ``<think>...</think>`` block out of plain content,
    if the server didn't already split reasoning into ``reasoning_content``.
    Mirrors ``hiexp.graph._split_thinking`` but returns just the body.
    """
    if not content or "<think>" not in content:
        return ""
    try:
        from hiexp.graph import _split_thinking  # type: ignore[import-not-found]

        thinking, _visible = _split_thinking(content)
        return (thinking or "").strip()
    except Exception:  # noqa: BLE001
        return ""
