"""LangGraph wiring for the HiExp agent.

Three nodes (initial → loop):

    init_node    Build SystemMessage (with E2 hits if HEK enabled) and
                 HumanMessage (rendered via the configured question
                 template). Records an ``e2_inject`` trajectory item.

    agent_node   Invoke the LLM (with MCP tools bound) on the current
                 message history. Appends the AIMessage and emits a
                 ``reasoning`` and/or ``output_text`` trajectory item.

    tool_node    For each tool call in the last AIMessage, dispatch to
                 the corresponding MCP tool. For the search tool, also
                 retrieve E1 hits using the search query and prepend
                 them to the tool response. Records ``tool_call`` (and
                 optionally ``e1_inject``) trajectory items.

Routing:

    init → agent → tools_condition → {tool, end}
                                      tool → agent (loop)

Guards:

    - ``cfg.max_turn`` bounds total iterations.
    - ``cfg.max_search_limit`` bounds *new* search queries; the agent
      keeps a dedup set and short-circuits repeats with a stub
      ``ToolMessage`` instead of re-querying the MCP server (mirrors
      Search-o1 ``run_search_o1.py:603-608``).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph

from hiexp.agent import TrajectoryItem
from hiexp.config import HiExpConfig
from hiexp.experience.store import HEKStore
from hiexp.prompts import (
    render_e1_prefix,
    render_e2_block,
    render_question,
    render_system_prompt,
)
from hiexp.state import HiExpState

logger = logging.getLogger(__name__)


# ── Helpers ─────────────────────────────────────────────────────────────────


_THINK_BLOCK = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_TOOL_CALL_BLOCK = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


def _split_thinking(content: str) -> tuple[str | None, str]:
    """Strip ``<think>...</think>`` blocks. Return ``(thinking, rest)``."""
    if not content:
        return None, content or ""
    matches = _THINK_BLOCK.findall(content)
    if not matches:
        return None, content
    rest = _THINK_BLOCK.sub("", content).strip()
    return "\n\n".join(m.strip() for m in matches), rest


def _split_responses_api_content(content: list) -> tuple[str, str]:
    """Pull (reasoning, visible_text) from a Responses-API content list.

    The OpenAI Responses API (and vLLM 0.19+'s implementation of it)
    returns ``AIMessage.content`` as a list of typed blocks::

        [
          {"type": "reasoning", "content": [{"text": "..."}], "summary": [...]},
          {"type": "text", "text": "..."},
          {"type": "function_call", ...},
        ]

    We concatenate all reasoning text and all visible text separately.
    Reasoning text may live under ``content[*].text`` (vLLM) or under
    ``summary[*].text`` (the documented OpenAI shape); we accept both.
    """
    reasoning_parts: list[str] = []
    text_parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "reasoning":
            for key in ("content", "summary"):
                inner = block.get(key) or []
                if not isinstance(inner, list):
                    continue
                for piece in inner:
                    if isinstance(piece, dict):
                        txt = piece.get("text") or ""
                        if isinstance(txt, str) and txt.strip():
                            reasoning_parts.append(txt.strip())
        elif btype == "text":
            txt = block.get("text") or ""
            if isinstance(txt, str) and txt.strip():
                text_parts.append(txt.strip())
        # function_call / tool_use blocks are surfaced via
        # ``AIMessage.tool_calls`` already; nothing to do here.
    return "\n\n".join(reasoning_parts), "\n\n".join(text_parts)


_NAME_KV = re.compile(r'"name"\s*:\s*"([^"]+)"')
_PARAM_BLOCK = re.compile(r"<parameter[=\s]+([A-Za-z_][\w.-]*)\s*>(.*?)</parameter>", re.DOTALL)
# Standalone ``<function=NAME>...</function>`` block — emitted by some
# SGLang chat templates / Qwen3 finetunes when the tool-call envelope
# collapses to an empty ``<tool_call></tool_call>`` followed by a free
# function block (observed empirically with Qwen3.6-35B-A3B on SGLang).
_FUNCTION_BLOCK = re.compile(
    r"<function[=\s]+([A-Za-z_][\w.-]*)\s*>(.*?)</function>", re.DOTALL,
)


def _parse_one_tool_call_block(raw: str) -> dict[str, Any] | None:
    """Try several increasingly lenient parses of one ``<tool_call>`` body.

    Order:
      1. Clean JSON ``{"name": ..., "arguments": ...}``.
      2. Fenced ```json {...} ``` block.
      3. Greedy ``{...}`` substring.
      4. Hybrid: ``"name": "X"`` plus ``<parameter=k>v</parameter>`` blocks
         (some Qwen-family models emit this when the chat-template tool
         renderer half-flips between OpenAI and Anthropic styles).
    """
    raw = raw.strip()
    if not raw:
        return None

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if fenced:
        try:
            obj = json.loads(fenced.group(1))
            if isinstance(obj, dict) and "name" in obj:
                return obj
        except json.JSONDecodeError:
            pass

    try:
        obj = json.loads(raw)
        if isinstance(obj, dict) and "name" in obj:
            return obj
    except json.JSONDecodeError:
        pass

    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict) and "name" in obj:
                return obj
        except json.JSONDecodeError:
            pass

    # Last resort: name from a JSON-ish key/value, args from XML params.
    name_match = _NAME_KV.search(raw)
    if not name_match:
        name_match = re.search(r"<function[=\s]+([A-Za-z_][\w.-]*)\s*>", raw)
    if not name_match:
        return None
    args: dict[str, Any] = {}
    for pm in _PARAM_BLOCK.finditer(raw):
        args[pm.group(1)] = pm.group(2).strip()
    if not args:
        return None
    return {"name": name_match.group(1), "arguments": args}


def _parse_text_tool_calls(content: str) -> list[dict[str, Any]]:
    """Parse ``<tool_call>...</tool_call>`` blocks into LangChain tool_call dicts.

    HiExp paper Table 7 specifies the standard protocol verbatim:
        ``<tool_call>\\n{"name": <function-name>, "arguments": <args>}\\n</tool_call>``

    When vLLM is launched without ``--tool-call-parser``, the model's
    raw ``<tool_call>...`` text lands in ``AIMessage.content`` and
    ``AIMessage.tool_calls`` is empty. We recover the structured form
    so the loop works without relaunching the server, and tolerate
    several common malformations of the body (fenced JSON, hybrid
    JSON/XML, etc.).
    """
    if not content:
        return []
    out: list[dict[str, Any]] = []

    # 1) Standard ``<tool_call>{json}</tool_call>`` blocks (paper Table 7).
    if "<tool_call>" in content:
        for i, body in enumerate(_TOOL_CALL_BLOCK.findall(content)):
            obj = _parse_one_tool_call_block(body)
            if obj is None:
                continue
            out.append(
                {
                    "name": str(obj["name"]),
                    "args": obj.get("arguments") or obj.get("args") or {},
                    "id": f"call_text_{i}",
                }
            )

    # 2) Fallback: standalone ``<function=NAME>...<parameter=K>V</parameter>...</function>``
    #    blocks. Triggered when a model wraps the call in <function> tags
    #    instead of <tool_call> (some SGLang chat templates / Qwen3
    #    fine-tunes do this). Only fires when (1) didn't already match.
    if not out and "<function" in content:
        for j, fn in enumerate(_FUNCTION_BLOCK.finditer(content)):
            name = fn.group(1)
            body = fn.group(2)
            args: dict[str, Any] = {}
            for pm in _PARAM_BLOCK.finditer(body):
                args[pm.group(1)] = pm.group(2).strip()
            if not args:
                continue
            out.append(
                {
                    "name": str(name),
                    "args": args,
                    "id": f"call_fn_{j}",
                }
            )
    return out


def _unwrap_mcp_text(raw: Any) -> Any:
    """Strip MCP's standard TextContent envelope.

    ``langchain-mcp-adapters`` returns raw MCP ``CallToolResult.content``
    which is a list of ``{"type": "text", "text": "<json or plain>"}``
    blocks. Concatenate the ``text`` payloads, parse JSON if possible,
    and return the inner value. If ``raw`` doesn't look like the MCP
    wrapper, returns it unchanged.
    """
    if (
        isinstance(raw, list)
        and raw
        and all(isinstance(b, dict) and b.get("type") == "text" for b in raw)
    ):
        joined = "".join(str(b.get("text", "")) for b in raw)
        try:
            return json.loads(joined)
        except json.JSONDecodeError:
            return joined
    return raw


def _coerce_tool_output_to_str(raw: Any) -> str:
    """Best-effort string form of an MCP tool's return value.

    First strips the MCP TextContent envelope so the model sees the
    actual payload (a JSON list of ``{docid, score, snippet}`` for
    BCP's ``search`` tool) rather than the wire-format wrapper.
    """
    raw = _unwrap_mcp_text(raw)
    if isinstance(raw, str):
        return raw
    try:
        return json.dumps(raw, ensure_ascii=False, default=str)
    except Exception:
        return str(raw)


def _extract_docids(raw: Any) -> list[str]:
    """Pull a list of docids out of a search-tool result.

    Handles raw list-of-dicts (the MCP server's native return type in
    ``third_party/BrowseComp-Plus/searcher/tools.py:51-62``),
    MCP-wrapped TextContent blocks, and JSON-stringified versions.
    """
    raw = _unwrap_mcp_text(raw)
    candidates: list[Any] = []
    if isinstance(raw, list):
        candidates = raw
    elif isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                candidates = parsed
        except json.JSONDecodeError:
            return []
    docids: list[str] = []
    for item in candidates:
        if isinstance(item, dict) and "docid" in item:
            docids.append(str(item["docid"]))
    return docids


# ── Node factories ──────────────────────────────────────────────────────────


def make_init_node(
    cfg: HiExpConfig,
    hek: HEKStore | None,
) -> Callable[[HiExpState], dict[str, Any]]:
    """Return the init node, closed over config + HEK."""

    def init_node(state: HiExpState) -> dict[str, Any]:
        question = state["question"]
        items: list[TrajectoryItem] = []

        if cfg.enable_hek and hek is not None:
            e2_hits = hek.retrieve_strategic(
                question,
                top_k=cfg.e2_top_k,
                use_e3=cfg.use_e3_instead_of_e2,
            )
            experience_block = render_e2_block(e2_hits)
            items.append(
                TrajectoryItem(
                    type="e2_inject",
                    output=[h.model_dump() for h in e2_hits],
                )
            )
        else:
            e2_hits = []
            experience_block = "None."

        system_text = render_system_prompt(
            experience_block,
            include_tool_call_template=cfg.include_tool_call_template,
        )
        # Optional extra block appended after the standard HiExp/BCP framing.
        # Used by ``src/ace_react`` to inject ACE's evolving playbook without
        # subclassing the prompt pipeline. Empty by default → no behavior change.
        if cfg.system_prompt_extra:
            system_text = system_text + "\n\n" + cfg.system_prompt_extra
        user_text = render_question(question, cfg.question_template)
        return {
            "messages": [SystemMessage(content=system_text), HumanMessage(content=user_text)],
            "executed_queries": [],
            "search_count": 0,
            "turn": 0,
            "trajectory_items": items,
        }

    return init_node


def make_agent_node(
    llm: BaseChatModel, cfg: HiExpConfig
) -> Callable[[HiExpState], dict[str, Any]]:
    """Return the agent node that calls the (tool-bound) LLM."""

    def agent_node(state: HiExpState) -> dict[str, Any]:
        ai_msg: AIMessage = llm.invoke(state["messages"])  # type: ignore[assignment]

        # vLLM serving gpt-oss over the Responses API emits the tool
        # call as an ``mcp_call`` content block with no parallel
        # ``function_call``; LangChain then leaves ``ai_msg.tool_calls``
        # empty and the router ends the graph after one search (every
        # rollout ``incomplete``, empty prediction). Promote ``mcp_call``
        # blocks into ``tool_calls``, then strip the blocks vLLM rejects
        # when they're echoed back in next turn's ``input``. Recover
        # MUST run before sanitize. Mirrors the working search_ttl react
        # loop (src/search_ttl/agent/search_agent.py:195-220).
        from hiexp.utils.tool_call_recovery import (
            recover_mcp_call_tool_calls,
            sanitize_responses_message_for_round_trip,
        )

        recover_mcp_call_tool_calls(ai_msg)
        sanitize_responses_message_for_round_trip(ai_msg)

        items: list[TrajectoryItem] = []

        # ``content`` shape depends on the LLM endpoint:
        #   - Chat Completions: a plain string (may carry inline
        #     ``<think>...</think>`` if the model emits it).
        #   - Responses API: a list of typed blocks
        #     (``reasoning`` / ``text`` / ``function_call`` / ...).
        if isinstance(ai_msg.content, list):
            api_reasoning, api_text = _split_responses_api_content(ai_msg.content)
        else:
            api_reasoning, api_text = "", ""

        content_str = ai_msg.content if isinstance(ai_msg.content, str) else api_text

        # Prefer the reasoning surfaced by the server's reasoning parser:
        #   - Responses API ``reasoning`` block (extracted above).
        #   - ``reasoning_content`` / ``reasoning`` in ``additional_kwargs``
        #     (Search-R1 / R1-family / DeepSeek-R1 chat-completions parsers).
        #   - Inline ``<think>...</think>`` in plain content.
        ak = getattr(ai_msg, "additional_kwargs", {}) or {}
        ak_reasoning = ak.get("reasoning_content") or ak.get("reasoning") or ""
        ak_reasoning = ak_reasoning.strip() if isinstance(ak_reasoning, str) else ""
        inline_reasoning, _ = _split_thinking(content_str if not api_text else "")
        thinking = api_reasoning or ak_reasoning or inline_reasoning or ""
        if thinking:
            items.append(TrajectoryItem(type="reasoning", output=[thinking]))

        # If vLLM was launched without --tool-call-parser, the model's
        # paper-Table-7 ``<tool_call>{json}</tool_call>`` blocks land in
        # ``content`` (string-valued) with empty ``tool_calls``. Recover
        # them so the rest of the loop works without relaunching the
        # server. The Responses-API path already populates ``tool_calls``
        # via structured ``function_call`` blocks, so this is a no-op
        # there.
        existing_tcs = getattr(ai_msg, "tool_calls", None) or []
        if not existing_tcs and not api_text:
            text_tcs = _parse_text_tool_calls(content_str)
            if text_tcs:
                ai_msg.tool_calls = text_tcs

        # Last-resort: gpt-oss sometimes fails to use the commentary
        # tool-call channel and instead emits its next search as raw
        # text — either a Harmony / <tool_call> block, or a bare
        # ``Search again.\n{"query": "..."}``. This shows up in the
        # *visible* content channel (Responses-API ``api_text``) just as
        # often as in the analysis/reasoning channel, so probe both —
        # content first, since that's where the observed gpt-oss-over-
        # Responses failures land. Parity with search_agent.py:255-294.
        if not getattr(ai_msg, "tool_calls", None):
            from hiexp.utils.tool_call_recovery import (
                recover_naked_query_json,
                recover_text_tool_calls,
            )

            for src_text in (content_str, thinking):
                if not src_text:
                    continue
                recovered = recover_text_tool_calls(
                    src_text
                ) or recover_naked_query_json(
                    src_text, tool_name=cfg.mcp_search_tool_name
                )
                if recovered:
                    ai_msg.tool_calls = recovered
                    break

        if not getattr(ai_msg, "tool_calls", None):
            visible = (api_text or content_str).strip()
            if visible:
                items.append(TrajectoryItem(type="output_text", output=visible))

        return {
            "messages": [ai_msg],
            "turn": state.get("turn", 0) + 1,
            "trajectory_items": items,
        }

    return agent_node


def make_tool_node(
    cfg: HiExpConfig,
    tools: list[BaseTool],
    hek: HEKStore | None,
) -> Callable[[HiExpState], dict[str, Any]]:
    """Return the tool node (async).

    Custom-built (vs ``langgraph.prebuilt.ToolNode``) so we can:
      - intercept the search tool's response and prefix E1 hits;
      - record structured trajectory items per call;
      - short-circuit duplicate queries without re-hitting the MCP
        server (Search-o1 dedup parity).

    ``langchain-mcp-adapters`` exposes its tools as async-only
    ``StructuredTool``s, so the node must invoke them via ``ainvoke``.
    LangGraph's ``ainvoke`` happily runs mixed sync/async nodes.
    """
    tools_by_name: dict[str, BaseTool] = {t.name: t for t in tools}
    search_tool_name = cfg.mcp_search_tool_name

    async def tool_node(state: HiExpState) -> dict[str, Any]:
        last_msg = state["messages"][-1]
        tool_calls = getattr(last_msg, "tool_calls", []) or []
        out_messages: list[ToolMessage] = []
        items: list[TrajectoryItem] = []
        new_executed: list[str] = []
        new_search_count = state.get("search_count", 0)
        executed = set(state.get("executed_queries", []))

        for tc in tool_calls:
            name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
            args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {})
            tc_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", "")

            tool = tools_by_name.get(name)
            if tool is None:
                content = f"Tool {name!r} is not available. Available: {sorted(tools_by_name)}"
                out_messages.append(ToolMessage(content=content, tool_call_id=tc_id, name=name or ""))
                items.append(
                    TrajectoryItem(type="tool_call", tool_name=name, arguments=args, output=content)
                )
                continue

            is_search = name == search_tool_name

            # ─── Search-o1-style guards on the search tool ────────────
            if is_search:
                query_str = (args or {}).get("query", "") if isinstance(args, dict) else ""
                if new_search_count >= cfg.max_search_limit:
                    note = (
                        "The maximum search limit is exceeded. You are not allowed to search."
                    )
                    out_messages.append(ToolMessage(content=note, tool_call_id=tc_id, name=name))
                    items.append(
                        TrajectoryItem(type="tool_call", tool_name=name, arguments=args, output=note)
                    )
                    continue
                if query_str and query_str in executed:
                    note = "You have searched this query. Please refer to previous results."
                    out_messages.append(ToolMessage(content=note, tool_call_id=tc_id, name=name))
                    items.append(
                        TrajectoryItem(type="tool_call", tool_name=name, arguments=args, output=note)
                    )
                    continue
                # Track the query and bump count *before* invocation so a
                # tool error still consumes the budget (mirrors Search-o1).
                executed.add(query_str)
                new_executed.append(query_str)
                new_search_count += 1

            # ─── Invoke the tool ──────────────────────────────────────
            try:
                raw_result: Any = await tool.ainvoke(args or {})
            except Exception as exc:  # noqa: BLE001
                logger.exception("Tool %r raised", name)
                raw_result = f"Error invoking {name}: {exc}"

            tool_output_str = _coerce_tool_output_to_str(raw_result)

            # ─── E1 injection for the search tool only ────────────────
            e1_hits = []
            content_to_send = tool_output_str
            if is_search and cfg.enable_hek and hek is not None and query_str:
                e1_hits = hek.retrieve_e1(
                    query_str,
                    threshold=cfg.e1_threshold,
                    top_k=cfg.e1_top_k,
                )
                if e1_hits:
                    content_to_send = render_e1_prefix(e1_hits) + tool_output_str

            out_messages.append(
                ToolMessage(content=content_to_send, tool_call_id=tc_id, name=name)
            )
            # Trajectory: ``tool_call`` records the *raw* output (without
            # E1 prefix) so eval can replay docs alone; ``e1_inject``
            # records the experience block separately.
            items.append(
                TrajectoryItem(
                    type="tool_call",
                    tool_name=name,
                    arguments=args,
                    output=tool_output_str,
                )
            )
            if e1_hits:
                items.append(
                    TrajectoryItem(
                        type="e1_inject",
                        tool_name=name,
                        arguments={"query": query_str},
                        output=[h.model_dump() for h in e1_hits],
                    )
                )

        return {
            "messages": out_messages,
            "executed_queries": new_executed,
            "search_count": new_search_count,
            "trajectory_items": items,
        }

    return tool_node


# ── Routing ─────────────────────────────────────────────────────────────────


def make_router(cfg: HiExpConfig) -> Callable[[HiExpState], str]:
    def route(state: HiExpState) -> str:
        if state.get("turn", 0) >= cfg.max_turn:
            return "end"
        last = state["messages"][-1] if state.get("messages") else None
        if last is None:
            return "end"
        tool_calls = getattr(last, "tool_calls", None)
        if tool_calls:
            return "tools"
        return "end"

    return route


# ── Public builder ──────────────────────────────────────────────────────────


def build_graph(
    cfg: HiExpConfig,
    llm_with_tools: BaseChatModel,
    tools: list[BaseTool],
    hek: HEKStore | None,
):
    """Compile the HiExp LangGraph for the given runtime objects."""
    graph = StateGraph(HiExpState)
    graph.add_node("init", make_init_node(cfg, hek))
    graph.add_node("agent", make_agent_node(llm_with_tools, cfg))
    graph.add_node("tools", make_tool_node(cfg, tools, hek))
    graph.set_entry_point("init")
    graph.add_edge("init", "agent")
    graph.add_conditional_edges(
        "agent",
        make_router(cfg),
        {"tools": "tools", "end": END},
    )
    graph.add_edge("tools", "agent")
    return graph.compile()


__all__ = ["build_graph", "_extract_docids"]
