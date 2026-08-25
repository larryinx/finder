"""Text-form tool-call recovery shared across agent / reflector / optimizer.

When the inference server isn't configured with a tool-call parser,
different model families leak their tool-call shape into ``content``
as raw text and ``ai.tool_calls`` stays empty. Without recovery, every
ReAct loop in this codebase sees "no tool call → break" and the
trajectory dies on turn 1.

Two shapes are supported:

- **Qwen3 / Hermes style** — ``<tool_call>{json}</tool_call>``.
  Delegates to ``hiexp.graph._parse_text_tool_calls``.

- **OpenAI Harmony style** — gpt-oss-20b/120b emit:

      <|start|>assistant
      <|channel|>analysis to=functions.search code
      <|message|>{"query": "..."}
      <|call|>

  Tolerant: any channel name (analysis / commentary / final), with or
  without the ``code`` keyword. Also tolerates malformed JSON in the
  message body — common when the model writes a phrase query like
  ``"foo" "bar"`` inside a JSON string and produces invalid escaping;
  in that case a regex fallback captures the inner query text.

Lives in ``hiexp.utils`` so the HiExp agent (``hiexp.graph``) and the
search_ttl stack can both use it without inverting the package layering
(search_ttl depends on hiexp, never the reverse). The legacy import
path ``search_ttl.utils.tool_call_recovery`` re-exports from here.
"""

from __future__ import annotations

import json
import re
import uuid

# OpenAI Harmony tool-call block. Captures the function name and the
# raw JSON body. Channel name + optional ``code`` keyword are tolerated.
_HARMONY_TOOL_CALL_RE = re.compile(
    r"<\|start\|>assistant"
    r"<\|channel\|>[^<]*?to=functions\.(?P<name>[A-Za-z_][A-Za-z0-9_]*)"
    r"[^<]*?<\|message\|>"
    r"(?P<body>.*?)"
    r"<\|call\|>",
    re.DOTALL,
)


def has_harmony_tool_calls(content: str) -> bool:
    return (
        isinstance(content, str)
        and "<|channel|>" in content
        and "to=functions." in content
    )


def parse_harmony_tool_calls(content: str) -> list[dict]:
    """Return ``[{name, args, id}, ...]`` for each harmony tool call
    found in ``content``. Returns ``[]`` when nothing matches."""
    if not has_harmony_tool_calls(content):
        return []
    out: list[dict] = []
    for m in _HARMONY_TOOL_CALL_RE.finditer(content):
        name = m.group("name")
        body = (m.group("body") or "").strip()
        args = _parse_args_tolerant(body)
        out.append(
            {
                "name": name,
                "args": args,
                "id": f"harmony_{uuid.uuid4().hex[:8]}",
            }
        )
    return out


def _parse_args_tolerant(body: str) -> dict:
    """Best-effort JSON parse with one repair attempt.

    gpt-oss sometimes emits malformed JSON when a query string itself
    contains double quotes (e.g. ``{"query": ""foo" "bar""}``). When
    strict json.loads fails, we capture the inner string between
    ``"query": "`` and ``"}`` via regex and use that as the query
    argument — which is what the model intended.
    """
    if not body:
        return {}
    if body.startswith("```"):
        body = re.sub(r"^```[a-zA-Z]*\n?", "", body)
        body = re.sub(r"\n?```$", "", body)
    try:
        v = json.loads(body)
        if isinstance(v, dict):
            return v
    except Exception:  # noqa: BLE001
        pass
    m = re.search(r'"query"\s*:\s*"(.*)"\s*\}\s*$', body, re.DOTALL)
    if m:
        return {"query": m.group(1)}
    # No structured args recoverable. Returning ``{"_raw": ...}`` lets
    # the caller surface a useful "missing query" error rather than
    # silently dropping the tool call.
    return {"_raw": body}


# ── Responses-API round-trip sanitizer ──────────────────────────────────


# Content-block ``type`` values that vLLM's Responses-API harmony
# serializer emits but rejects when they're echoed back in the next
# request's ``input`` array (``response_input_to_harmony`` raises
# ``ValueError: Unknown input type: ...``). Stripping them on the
# client side preserves the actionable information (which lives in
# ``ai.tool_calls`` / our trajectory layer) while letting multi-turn
# ReAct continue.
_VLLM_RESPONSES_REJECTED_BLOCK_TYPES: frozenset[str] = frozenset(
    {"mcp_call", "mcp_call_output", "mcp_list_tools", "mcp_approval_request"}
)


def recover_mcp_call_tool_calls(ai_message) -> list[dict]:
    """Promote ``mcp_call`` content blocks into ``ai.tool_calls`` entries.

    LangChain's ``_construct_lc_result_from_responses_api`` only
    populates ``ai.tool_calls`` from ``function_call`` blocks; it
    treats ``mcp_call`` as a content-only block. When vLLM gpt-oss
    serves the Responses API it sometimes emits ONLY ``mcp_call``
    (no parallel ``function_call``), in which case ``ai.tool_calls``
    is empty and our agent sees "no tool call → nudge" even though
    the model actually issued one.

    This helper walks ``ai.content``, extracts each ``mcp_call`` into
    the same dict shape ``function_call`` would have produced (``name``
    + parsed ``args`` + ``id``), appends them to ``ai.tool_calls``
    in place, and returns the recovered list. Callers should invoke
    this BEFORE ``sanitize_responses_message_for_round_trip`` strips
    the blocks — otherwise the tool-call evidence is lost.

    Returns the recovered tool_call dicts (empty list if nothing).
    """
    content = getattr(ai_message, "content", None)
    if not isinstance(content, list):
        return []
    existing_ids = {tc.get("id") for tc in (ai_message.tool_calls or [])}
    recovered: list[dict] = []
    for block in content:
        if not (isinstance(block, dict) and block.get("type") == "mcp_call"):
            continue
        name = block.get("name") or ""
        if not name:
            continue
        # Arguments are JSON-encoded string; parse if possible.
        raw_args = block.get("arguments") or "{}"
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            if not isinstance(args, dict):
                args = {"_raw": raw_args}
        except Exception:  # noqa: BLE001
            args = {"_raw": raw_args}
        # Prefer the mcp_call's id (no call_/fc_ prefix in vLLM's output);
        # the downstream ToolMessage.tool_call_id must match this.
        call_id = block.get("id") or f"mcpc_{uuid.uuid4().hex[:8]}"
        if call_id in existing_ids:
            continue  # avoid double-counting if function_call sibling already present
        recovered.append({
            "type": "tool_call",
            "name": name,
            "args": args,
            "id": call_id,
        })
    if recovered:
        # Append rather than replace so a sibling function_call (if any)
        # is preserved.
        ai_message.tool_calls = list(ai_message.tool_calls or []) + recovered
    return recovered


def sanitize_responses_message_for_round_trip(ai_message) -> int:
    """In-place strip Responses-API content blocks that vLLM emits but
    can't accept back. Returns the number of blocks removed.

    Only meaningful when ``ai.content`` is a list of typed dicts
    (``output_version='responses/v1'``); a string content is left
    untouched. Safe to call unconditionally.

    NOTE: callers must run ``recover_mcp_call_tool_calls`` BEFORE this
    function, otherwise tool-call evidence carried only in ``mcp_call``
    blocks is lost (LangChain populates ``ai.tool_calls`` from
    ``function_call`` blocks only; vLLM gpt-oss may emit ``mcp_call``
    without a parallel ``function_call``).
    """
    content = getattr(ai_message, "content", None)
    if not isinstance(content, list):
        return 0
    cleaned: list = []
    removed = 0
    for block in content:
        if (
            isinstance(block, dict)
            and block.get("type") in _VLLM_RESPONSES_REJECTED_BLOCK_TYPES
        ):
            removed += 1
            continue
        cleaned.append(block)
    if removed:
        ai_message.content = cleaned
    return removed


def recover_text_tool_calls(content: str | None) -> list[dict]:
    """Try every known text-form tool-call shape on ``content`` and
    return the union. Empty list if nothing matches or content is not
    a string."""
    if not isinstance(content, str) or not content:
        return []
    out: list[dict] = []
    if "<tool_call>" in content:
        # Qwen3 / Hermes form — reuse the helper HiExp already vets.
        from hiexp.graph import _parse_text_tool_calls  # type: ignore[import-not-found]

        out.extend(_parse_text_tool_calls(content) or [])
    if has_harmony_tool_calls(content):
        out.extend(parse_harmony_tool_calls(content))
    return out


# ── "Naked JSON in reasoning" recovery ─────────────────────────────────
#
# Matches ``{"query": "..."}`` (possibly with escaped quotes inside) as a
# substring. Used as a LAST-RESORT fallback on ``additional_kwargs.
# reasoning_content`` when neither structured ``tool_calls`` nor the
# Harmony/<tool_call> recovery turned up anything.
#
# Symptom this fixes: gpt-oss-120b served via OpenRouter/DeepInfra
# (and possibly other harmony backends) sometimes "thinks aloud" about
# its next search in the **analysis** channel — the channel content
# ends with a literal ``{"query": "..."}`` blob instead of being emitted
# as a proper ``<|channel|>commentary to=functions.search<|message|>...
# <|call|>`` item. The provider's harmony parser only routes the
# commentary channel into ``tool_calls``, so the analysis-channel JSON
# stays in ``message.reasoning`` and ``tool_calls`` is empty. Without
# recovery our agent loop sees "no tool call + no visible content",
# nudges once, then gives up — accounting for ~98% of failed
# ``baseline_200_openrouter`` trajectories before this fix.
_NAKED_QUERY_JSON_RE = re.compile(
    r'\{\s*"query"\s*:\s*"((?:[^"\\]|\\.)*)"\s*\}',
    re.DOTALL,
)


def recover_naked_query_json(
    content: str | None, tool_name: str = "search"
) -> list[dict]:
    """Promote the LAST ``{"query": "..."}`` JSON object in ``content``
    into a synthetic ``search`` tool call.

    Returns ``[]`` when no match. When matches exist, the last one is
    used (typically the model's most recent intended action). Callers
    should only invoke this AFTER the structured ``tool_calls`` field
    and ``recover_text_tool_calls(content)`` both came up empty, to
    minimise false positives.
    """
    if not isinstance(content, str) or not content:
        return []
    matches = list(_NAKED_QUERY_JSON_RE.finditer(content))
    if not matches:
        return []
    raw = matches[-1].group(1)
    # JSON-decode the captured string body to undo escaping
    # (e.g. ``\"foo\"`` → ``"foo"``). Fall back to the raw capture if
    # decoding fails — the search engine will still get *some* string.
    try:
        query = json.loads(f'"{raw}"')
    except Exception:  # noqa: BLE001
        query = raw
    if not query:
        return []
    return [{
        "name": tool_name,
        "args": {"query": query},
        "id": f"naked_{uuid.uuid4().hex[:8]}",
        "type": "tool_call",
    }]
