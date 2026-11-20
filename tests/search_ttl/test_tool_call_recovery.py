"""Tool-call text-form recovery: both Qwen3 and gpt-oss Harmony shapes.

The four ReAct loops (agent, both reflectors, curator) all share the
same recovery helper. These tests pin:

  - parse_harmony_tool_calls extracts ``name`` + ``args`` from gpt-oss
    Harmony blocks across the channel name variants.
  - the malformed-JSON repair path catches the actual production
    failure the user reported (nested double-quotes inside a phrase
    query produced invalid JSON).
  - recover_text_tool_calls dispatches to both Qwen3 and Harmony
    parsers without one breaking the other.
  - every ReAct entry point imports/uses the shared helper so a
    future regression that drops Harmony recovery from one site is
    caught by grep-style assertion.
"""

from __future__ import annotations

from pathlib import Path

from search_ttl.utils.tool_call_recovery import (
    has_harmony_tool_calls,
    parse_harmony_tool_calls,
    recover_text_tool_calls,
)


# ── direct parser tests ──────────────────────────────────────────────────


_HARMONY_ANALYSIS_CHANNEL = (
    "<|start|>assistant<|channel|>analysis to=functions.search code"
    '<|message|>{"query": "Bleomycin pulmonary fibrosis"}<|call|>'
)

_HARMONY_COMMENTARY_CHANNEL = (
    "<|start|>assistant<|channel|>commentary to=functions.search"
    '<|message|>{"query": "alt route"}<|call|>'
)


def test_has_harmony_tool_calls_detects_token_pair():
    assert has_harmony_tool_calls(_HARMONY_ANALYSIS_CHANNEL)
    assert has_harmony_tool_calls(_HARMONY_COMMENTARY_CHANNEL)
    assert not has_harmony_tool_calls("plain text")
    assert not has_harmony_tool_calls("<tool_call>{}</tool_call>")
    assert not has_harmony_tool_calls(None)  # type: ignore[arg-type]


def test_parse_harmony_extracts_name_and_query_analysis_channel():
    out = parse_harmony_tool_calls(_HARMONY_ANALYSIS_CHANNEL)
    assert len(out) == 1
    assert out[0]["name"] == "search"
    assert out[0]["args"] == {"query": "Bleomycin pulmonary fibrosis"}
    assert out[0]["id"].startswith("harmony_")


def test_parse_harmony_extracts_commentary_channel_without_code_keyword():
    out = parse_harmony_tool_calls(_HARMONY_COMMENTARY_CHANNEL)
    assert out[0]["name"] == "search"
    assert out[0]["args"]["query"] == "alt route"


def test_parse_harmony_repairs_malformed_json_with_nested_double_quotes():
    """The exact production failure: gpt-oss writes a phrase query
    like ``"Bleomycin" "sulfasalazine"`` inside the JSON ``query``
    string, producing invalid JSON. Strict json.loads fails. The
    repair regex should recover the inner string."""
    content = (
        "<|start|>assistant<|channel|>analysis to=functions.search code"
        '<|message|>{"query": ""Bleomycin" "sulfasalazine" "MRC-5" "Redox Biology""}'
        "<|call|>"
    )
    out = parse_harmony_tool_calls(content)
    assert len(out) == 1
    assert out[0]["name"] == "search"
    # The captured query is the intended phrase concatenation —
    # everything between the first inner `"` and the last `"`.
    q = out[0]["args"]["query"]
    assert "Bleomycin" in q
    assert "sulfasalazine" in q
    assert "MRC-5" in q
    assert "Redox Biology" in q


def test_parse_harmony_handles_multiple_blocks_in_one_message():
    content = _HARMONY_ANALYSIS_CHANNEL + "\n" + _HARMONY_COMMENTARY_CHANNEL
    out = parse_harmony_tool_calls(content)
    assert len(out) == 2
    assert out[0]["args"]["query"] == "Bleomycin pulmonary fibrosis"
    assert out[1]["args"]["query"] == "alt route"


def test_parse_harmony_returns_empty_on_no_match():
    assert parse_harmony_tool_calls("plain text") == []
    assert parse_harmony_tool_calls("<tool_call>{}</tool_call>") == []


def test_parse_harmony_strips_code_fences_around_body():
    """When the model wraps the JSON body in markdown fences."""
    content = (
        "<|start|>assistant<|channel|>analysis to=functions.search code"
        '<|message|>```json\n{"query": "x"}\n```<|call|>'
    )
    out = parse_harmony_tool_calls(content)
    assert out[0]["args"] == {"query": "x"}


def test_parse_harmony_unrecoverable_args_keeps_raw_so_caller_can_error():
    content = (
        "<|start|>assistant<|channel|>analysis to=functions.search code"
        "<|message|>totally not json<|call|>"
    )
    out = parse_harmony_tool_calls(content)
    assert out[0]["name"] == "search"
    # Surface the raw body so the MCP dispatch returns a useful
    # "missing query" hint instead of silently dropping the call.
    assert out[0]["args"] == {"_raw": "totally not json"}


# ── union recovery dispatch ──────────────────────────────────────────────


def test_recover_text_dispatches_to_qwen3_form():
    content = '<tool_call>\n{"name": "search", "arguments": {"query": "x"}}\n</tool_call>'
    out = recover_text_tool_calls(content)
    assert len(out) == 1
    assert out[0]["name"] == "search"
    assert out[0]["args"] == {"query": "x"}


def test_recover_text_dispatches_to_harmony_form():
    out = recover_text_tool_calls(_HARMONY_ANALYSIS_CHANNEL)
    assert len(out) == 1
    assert out[0]["name"] == "search"


def test_recover_text_returns_empty_on_non_string():
    assert recover_text_tool_calls(None) == []
    assert recover_text_tool_calls("") == []


def test_recover_text_returns_union_when_both_forms_appear():
    """Defensive: if a single message contained both shapes, both
    should surface (the recovery helper unions them; downstream
    deduplication is the caller's responsibility)."""
    qwen = '<tool_call>{"name": "search", "arguments": {"query": "q1"}}</tool_call>'
    content = qwen + "\n" + _HARMONY_COMMENTARY_CHANNEL
    out = recover_text_tool_calls(content)
    assert len(out) == 2


# ── all four ReAct entry points wire the shared helper ───────────────────


def test_sanitize_strips_mcp_call_block_keeps_others():
    """The exact production failure: vLLM's Responses API emits an
    ``mcp_call`` content block alongside reasoning + text. Subsequent
    turns echo it back and vLLM 400s on its own type. The sanitizer
    removes ``mcp_call`` (and related MCP-server-side types) while
    preserving everything actionable."""
    from langchain_core.messages import AIMessage

    from search_ttl.utils.tool_call_recovery import (
        sanitize_responses_message_for_round_trip,
    )

    ai = AIMessage(
        content=[
            {"type": "reasoning", "summary": [{"type": "summary_text",
                                                "text": "Let me search..."}]},
            {"type": "text", "text": "Calling search."},
            {"type": "mcp_call", "name": "search",
             "arguments": '{"query": "x"}'},
            {"type": "function_call", "name": "search",
             "arguments": '{"query": "x"}', "id": "fc_1"},
            {"type": "mcp_call_output", "output": "..."},
        ],
    )
    n = sanitize_responses_message_for_round_trip(ai)
    assert n == 2  # mcp_call + mcp_call_output
    types_left = [b.get("type") for b in ai.content if isinstance(b, dict)]
    assert "mcp_call" not in types_left
    assert "mcp_call_output" not in types_left
    # Actionable content preserved.
    assert "reasoning" in types_left
    assert "text" in types_left
    assert "function_call" in types_left


def test_sanitize_is_noop_on_string_content():
    """``chat_completions`` flavor produces string content; sanitizer
    must not touch it."""
    from langchain_core.messages import AIMessage

    from search_ttl.utils.tool_call_recovery import (
        sanitize_responses_message_for_round_trip,
    )

    ai = AIMessage(content="plain string content")
    n = sanitize_responses_message_for_round_trip(ai)
    assert n == 0
    assert ai.content == "plain string content"


def test_sanitize_returns_zero_when_no_rejected_blocks():
    """No-op fast path: typical Responses output with just reasoning +
    text + function_call should be untouched."""
    from langchain_core.messages import AIMessage

    from search_ttl.utils.tool_call_recovery import (
        sanitize_responses_message_for_round_trip,
    )

    blocks = [
        {"type": "reasoning", "summary": [{"type": "summary_text", "text": "..."}]},
        {"type": "function_call", "name": "search", "arguments": "{}", "id": "fc_1"},
    ]
    ai = AIMessage(content=list(blocks))
    n = sanitize_responses_message_for_round_trip(ai)
    assert n == 0
    assert ai.content == blocks


def test_sanitize_handles_unknown_mcp_variants():
    """vLLM may add more mcp_* block types in future versions; we
    pre-strip the ones the harmony serializer is known to reject so
    one new type doesn't break the entire run."""
    from langchain_core.messages import AIMessage

    from search_ttl.utils.tool_call_recovery import (
        sanitize_responses_message_for_round_trip,
    )

    ai = AIMessage(content=[
        {"type": "mcp_list_tools"},
        {"type": "mcp_approval_request"},
        {"type": "text", "text": "keep me"},
    ])
    n = sanitize_responses_message_for_round_trip(ai)
    assert n == 2
    assert ai.content == [{"type": "text", "text": "keep me"}]


def test_sanitize_e2e_preserves_call_id_linkage_when_both_blocks_present():
    """End-to-end through LangChain's actual ``_construct_responses_api_input``:
    when vLLM emits BOTH ``mcp_call`` and ``function_call``, stripping
    leaves a clean wire payload — ``function_call`` reaches vLLM and
    pairs with the downstream ``function_call_output`` by ``call_id``.
    Reasoning and text blocks are unchanged. This is the assertion
    that proves the strip "doesn't change what the model receives"
    (the user's actual concern)."""
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    from langchain_openai.chat_models.base import _construct_responses_api_input

    from search_ttl.utils.tool_call_recovery import (
        sanitize_responses_message_for_round_trip,
    )

    ai = AIMessage(
        content=[
            {"type": "reasoning", "id": "rs_1",
             "summary": [{"type": "summary_text", "text": "..."}]},
            {"type": "text", "id": "msg_1", "text": "Searching."},
            {"type": "mcp_call", "id": "mcpc_xxx", "name": "search",
             "server_label": "ttl_mcp", "arguments": '{"query": "x"}'},
            {"type": "function_call", "id": "fc_xxx", "call_id": "call_xxx",
             "name": "search", "arguments": '{"query": "x"}'},
        ],
        tool_calls=[{"name": "search", "args": {"query": "x"},
                     "id": "call_xxx", "type": "tool_call"}],
    )
    sanitize_responses_message_for_round_trip(ai)
    payload = _construct_responses_api_input([
        HumanMessage(content="q"),
        ai,
        ToolMessage(content='[]', tool_call_id="call_xxx"),
    ])
    types_on_wire = [it.get("type") for it in payload]
    # vLLM's harmony serializer rejects mcp_call — must not be present.
    assert "mcp_call" not in types_on_wire
    # function_call reaches the server so it has a record of the tool call.
    assert "function_call" in types_on_wire
    fc = next(it for it in payload if it.get("type") == "function_call")
    fco = next(it for it in payload if it.get("type") == "function_call_output")
    # call_id linkage intact so vLLM can pair tool result with call.
    assert fc["call_id"] == fco["call_id"] == "call_xxx"
    # Reasoning preserved → thinking-on round-trip works.
    assert "reasoning" in types_on_wire


def test_sanitize_e2e_synthesizes_function_call_when_only_mcp_call_emitted():
    """Worst case: vLLM emits ONLY ``mcp_call`` (no parallel
    ``function_call``). After strip, the content list has zero
    tool-call blocks. LangChain's serializer must re-synthesize a
    ``function_call`` from ``ai.tool_calls`` so the call_id linkage
    survives on the wire. If this assertion ever fails (e.g.,
    langchain-openai changes its merge logic), our strip would
    silently break call_id pairing and tool results would orphan."""
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    from langchain_openai.chat_models.base import _construct_responses_api_input

    from search_ttl.utils.tool_call_recovery import (
        sanitize_responses_message_for_round_trip,
    )

    ai = AIMessage(
        content=[
            {"type": "reasoning", "id": "rs_1",
             "summary": [{"type": "summary_text", "text": "..."}]},
            {"type": "mcp_call", "id": "mcpc_yyy", "name": "search",
             "server_label": "ttl_mcp", "arguments": '{"query": "y"}'},
        ],
        # Populated by LangChain from the mcp_call when it parses
        # vLLM's response into an AIMessage.
        tool_calls=[{"name": "search", "args": {"query": "y"},
                     "id": "call_yyy", "type": "tool_call"}],
    )
    sanitize_responses_message_for_round_trip(ai)
    payload = _construct_responses_api_input([
        HumanMessage(content="q"),
        ai,
        ToolMessage(content='[]', tool_call_id="call_yyy"),
    ])
    types_on_wire = [it.get("type") for it in payload]
    assert "mcp_call" not in types_on_wire
    assert "function_call" in types_on_wire, (
        "function_call NOT re-synthesized from ai.tool_calls — strip "
        "would silently break call_id linkage."
    )
    fc = next(it for it in payload if it.get("type") == "function_call")
    fco = next(it for it in payload if it.get("type") == "function_call_output")
    assert fc["call_id"] == fco["call_id"] == "call_yyy"


def test_recover_mcp_call_tool_calls_extracts_name_args_id():
    """The user-reported failure case: vLLM gpt-oss emits ONLY mcp_call
    (no parallel function_call). LangChain leaves ai.tool_calls empty.
    Our helper must promote the mcp_call into a proper tool_call so the
    agent loop can dispatch it."""
    from langchain_core.messages import AIMessage

    from search_ttl.utils.tool_call_recovery import recover_mcp_call_tool_calls

    ai = AIMessage(
        content=[
            {"type": "reasoning", "summary": [{"type": "summary_text",
                                                "text": "I should search."}]},
            {"type": "mcp_call", "id": "mcpc_abc", "name": "search",
             "server_label": "ttl_mcp",
             "arguments": '{"query": "Bleomycin pulmonary fibrosis"}'},
        ],
        tool_calls=[],  # LangChain did not populate from mcp_call
    )
    recovered = recover_mcp_call_tool_calls(ai)
    assert len(recovered) == 1
    assert recovered[0]["name"] == "search"
    assert recovered[0]["args"] == {"query": "Bleomycin pulmonary fibrosis"}
    assert recovered[0]["id"] == "mcpc_abc"
    # Mutation: ai.tool_calls is updated in place so the agent's
    # ``getattr(ai, 'tool_calls', None) or []`` next line sees it.
    assert ai.tool_calls == recovered


def test_recover_mcp_call_does_not_double_count_function_call_sibling():
    """When vLLM emits BOTH mcp_call AND function_call with the same
    call_id, LangChain already populated tool_calls from function_call.
    We must not duplicate."""
    from langchain_core.messages import AIMessage

    from search_ttl.utils.tool_call_recovery import recover_mcp_call_tool_calls

    ai = AIMessage(
        content=[
            {"type": "mcp_call", "id": "call_xxx", "name": "search",
             "arguments": '{"query": "x"}'},
            {"type": "function_call", "call_id": "call_xxx", "name": "search",
             "arguments": '{"query": "x"}'},
        ],
        tool_calls=[{"name": "search", "args": {"query": "x"},
                     "id": "call_xxx", "type": "tool_call"}],
    )
    recovered = recover_mcp_call_tool_calls(ai)
    assert recovered == []
    assert len(ai.tool_calls) == 1  # not duplicated


def test_recover_mcp_call_is_noop_on_string_content():
    """chat_completions path: ai.content is a string. Helper must be
    a strict no-op so chat_completions trajectories are untouched."""
    from langchain_core.messages import AIMessage

    from search_ttl.utils.tool_call_recovery import recover_mcp_call_tool_calls

    ai = AIMessage(
        content="Let me search.",
        tool_calls=[{"name": "search", "args": {"query": "y"},
                     "id": "call_xxx", "type": "tool_call"}],
    )
    recovered = recover_mcp_call_tool_calls(ai)
    assert recovered == []
    assert ai.content == "Let me search."
    assert len(ai.tool_calls) == 1


def test_recover_mcp_call_tolerates_malformed_arguments_json():
    from langchain_core.messages import AIMessage

    from search_ttl.utils.tool_call_recovery import recover_mcp_call_tool_calls

    ai = AIMessage(
        content=[
            {"type": "mcp_call", "id": "mcpc_z", "name": "search",
             "arguments": '{not valid json'},
        ],
        tool_calls=[],
    )
    recovered = recover_mcp_call_tool_calls(ai)
    assert len(recovered) == 1
    assert recovered[0]["name"] == "search"
    # Falls back to raw passthrough so the MCP dispatch surfaces a
    # useful "missing query" hint rather than silently dropping it.
    assert "_raw" in recovered[0]["args"]


def test_e2e_promote_then_sanitize_preserves_tool_call_when_only_mcp_call_emitted():
    """Full pipeline trace: simulate vLLM gpt-oss emitting ONLY
    mcp_call (the case that previously caused the empty-turn nudge),
    run BOTH helpers in the order the agent loop calls them
    (promote → sanitize), and confirm:

      1. ai.tool_calls is populated → agent loop will dispatch search
      2. mcp_call is stripped from ai.content → next request won't 400
      3. After LangChain serializes, the wire payload has a function_call
         with the matching call_id and the function_call_output pairs.
    """
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    from langchain_openai.chat_models.base import _construct_responses_api_input

    from search_ttl.utils.tool_call_recovery import (
        recover_mcp_call_tool_calls,
        sanitize_responses_message_for_round_trip,
    )

    # vLLM gpt-oss response with ONLY mcp_call (no parallel function_call).
    ai = AIMessage(
        content=[
            {"type": "reasoning", "id": "rs_1",
             "summary": [{"type": "summary_text", "text": "look it up."}]},
            {"type": "mcp_call", "id": "call_xxx", "name": "search",
             "server_label": "ttl_mcp",
             "arguments": '{"query": "test"}'},
        ],
        tool_calls=[],  # LangChain leaves this empty for mcp_call-only output
    )

    # Step 1: promote — the agent loop now sees tool_calls populated.
    recovered = recover_mcp_call_tool_calls(ai)
    assert len(recovered) == 1
    assert ai.tool_calls[0]["name"] == "search"

    # Step 2: sanitize — strips the redundant mcp_call so vLLM's
    # harmony serializer won't 400 on the echo.
    n_stripped = sanitize_responses_message_for_round_trip(ai)
    assert n_stripped == 1

    # Step 3: wire payload looks right.
    tool_msg = ToolMessage(content='[]', tool_call_id="call_xxx")
    payload = _construct_responses_api_input([
        HumanMessage(content="q"),
        ai,
        tool_msg,
    ])
    types_on_wire = [it.get("type") for it in payload]
    assert "mcp_call" not in types_on_wire
    assert "function_call" in types_on_wire
    fc = next(it for it in payload if it.get("type") == "function_call")
    fco = next(it for it in payload if it.get("type") == "function_call_output")
    assert fc["call_id"] == fco["call_id"] == "call_xxx"


def test_all_react_sites_wire_mcp_call_promotion():
    """Every ReAct loop must call ``recover_mcp_call_tool_calls`` and
    must do so BEFORE the sanitizer (otherwise the strip deletes the
    tool-call evidence before the agent can extract it)."""
    paths = [
        "src/search_ttl/agent/search_agent.py",
        "src/search_ttl/reflector/querychain_tactic.py",
        "src/search_ttl/reflector/querychain_staged.py",
        "src/search_ttl/curator/react_tactic.py",
        "src/search_ttl/curator/react_staged.py",
    ]
    for p in paths:
        src = Path(p).read_text()
        assert "recover_mcp_call_tool_calls" in src, (
            f"{p} missing mcp_call → tool_calls promotion"
        )
        # Ordering: the promote helper must appear before the strip
        # helper in the source file (both are imported together and
        # called in sequence right after ``ainvoke``).
        promote_pos = src.index("recover_mcp_call_tool_calls(ai)")
        strip_pos = src.index("sanitize_responses_message_for_round_trip(ai)")
        assert promote_pos < strip_pos, (
            f"{p} calls sanitize before promote — strip would delete "
            "the mcp_call before promotion can extract it"
        )


def test_all_react_sites_wire_responses_sanitizer():
    """Every ReAct loop (agent, reflectors, curators) must
    each call the sanitizer right after ``ainvoke`` so the next turn
    doesn't ship an mcp_call block back to vLLM."""
    paths = [
        "src/search_ttl/agent/search_agent.py",
        "src/search_ttl/reflector/querychain_tactic.py",
        "src/search_ttl/reflector/querychain_staged.py",
        "src/search_ttl/curator/react_tactic.py",
        "src/search_ttl/curator/react_staged.py",
    ]
    for p in paths:
        src = Path(p).read_text()
        assert "sanitize_responses_message_for_round_trip" in src, (
            f"{p} missing Responses-API sanitizer wiring"
        )


def test_all_react_sites_use_recover_text_tool_calls():
    """Every ReAct loop must route through ``recover_text_tool_calls``
    so adding a new model family's tool-call shape only requires
    extending one parser, not patching every call site."""
    paths = [
        "src/search_ttl/agent/search_agent.py",
        "src/search_ttl/reflector/querychain_tactic.py",
        "src/search_ttl/reflector/querychain_staged.py",
        "src/search_ttl/curator/react_tactic.py",
        "src/search_ttl/curator/react_staged.py",
    ]
    for p in paths:
        src = Path(p).read_text()
        assert "recover_text_tool_calls" in src, f"{p} missing harmony recovery wiring"
        # Defense in depth: make sure no site reverted to the old
        # Qwen3-only literal check.
        assert 'if not tool_calls and isinstance(ai.content, str) and "<tool_call>" in ai.content' not in src, (
            f"{p} still has the Qwen3-only fallback; should call recover_text_tool_calls"
        )
