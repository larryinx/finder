"""Pin the vLLM Responses-API status patch.

vLLM's strict pydantic ``ResponseOutputMessage`` requires
``status: 'in_progress' | 'completed' | 'incomplete'``. LangChain's
``_construct_responses_api_input`` omits ``status`` entirely on
assistant message items, causing vLLM to 400 on multi-turn ReAct
with a multi-page union-decomposition error.

Our monkey-patch in ``utils/responses_api_compat`` injects
``status='completed'`` on assistant message items. These tests pin:

  - the patch installs idempotently
  - assistant items get status; user/system/function_call items don't
  - the patch reaches the actual wire payload through ChatOpenAI's
    ``_get_request_payload`` (the real HTTP-bound path)
  - chat_completions flavor is untouched (patch only matters for
    Responses-API output_version)
"""

from __future__ import annotations


def test_patch_adds_status_to_assistant_message_items():
    """Direct call: assistant message items pick up status='completed';
    user / system / non-message items are untouched."""
    from langchain_core.messages import AIMessage, HumanMessage

    from search_ttl.utils.responses_api_compat import (
        _install_responses_api_status_patch,
    )
    _install_responses_api_status_patch()

    # Look up via module attribute so we get the patched binding.
    import langchain_openai.chat_models.base as _lc_base

    ai = AIMessage(
        content=[
            {"type": "text", "id": "msg_1", "text": "intermediate commentary"},
        ],
        tool_calls=[],
    )
    payload = _lc_base._construct_responses_api_input([
        HumanMessage(content="q"),
        ai,
    ])

    # Find every item and check status by role.
    for item in payload:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "message" and item.get("role") == "assistant":
            assert item.get("status") == "completed", (
                f"patch failed to inject status on assistant: {item}"
            )
        elif item.get("type") == "message" and item.get("role") in {"user", "system"}:
            assert "status" not in item, (
                f"patch wrongly injected status on {item.get('role')}: {item}"
            )


def test_patch_is_idempotent():
    """Re-installing should not stack; status appears once, value
    unchanged across re-installs."""
    from langchain_core.messages import AIMessage, HumanMessage

    from search_ttl.utils.responses_api_compat import (
        _install_responses_api_status_patch,
    )
    _install_responses_api_status_patch()
    _install_responses_api_status_patch()
    _install_responses_api_status_patch()

    import langchain_openai.chat_models.base as _lc_base
    payload = _lc_base._construct_responses_api_input([
        HumanMessage(content="q"),
        AIMessage(content=[{"type": "text", "id": "m1", "text": "hi"}]),
    ])
    asst = [it for it in payload
            if isinstance(it, dict)
            and it.get("type") == "message"
            and it.get("role") == "assistant"]
    assert len(asst) == 1
    assert asst[0]["status"] == "completed"


def test_patch_preserves_existing_status():
    """If LangChain ever starts emitting status itself (e.g., a future
    version), our patch must not overwrite it."""
    from langchain_core.messages import AIMessage, HumanMessage

    from search_ttl.utils.responses_api_compat import (
        _install_responses_api_status_patch,
    )
    _install_responses_api_status_patch()

    import langchain_openai.chat_models.base as _lc_base
    orig = _lc_base._construct_responses_api_input

    # Have the original return an assistant item that ALREADY has a
    # status — simulating a future LangChain. Wrap in a temporary
    # replacement to drive the test.
    def _stub(messages):
        return [{"type": "message", "role": "assistant",
                 "id": "m1", "status": "in_progress",
                 "content": [{"type": "output_text", "text": "x",
                              "annotations": []}]}]

    _lc_base._construct_responses_api_input = _stub
    try:
        # Re-install on top of the stub.
        _install_responses_api_status_patch()
        out = _lc_base._construct_responses_api_input(messages=[])
        # status must remain 'in_progress', not overwritten to 'completed'.
        assert out[0].get("status") == "in_progress"
    finally:
        _lc_base._construct_responses_api_input = orig
        # Re-install the proper patch on the real function so other
        # tests see correct behavior.
        _install_responses_api_status_patch()


def test_patch_reaches_real_request_payload_via_build_chat_llm():
    """End-to-end via ChatOpenAI's ``_get_request_payload`` — the
    actual production code path that runs before HTTP send."""
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    from search_ttl.utils.llm_builder import build_chat_llm

    # build_chat_llm installs the patch as a side effect when flavor
    # is "responses".
    llm = build_chat_llm(
        model="x", base_url="http://x", api_key="x",
        llm_api_flavor="responses",
        enable_thinking=True,
    )
    messages = [
        HumanMessage(content="q"),
        AIMessage(
            content=[
                {"type": "text", "id": "msg_1", "text": "commentary"},
                {"type": "function_call", "id": "fc_1", "call_id": "call_1",
                 "name": "search", "arguments": '{"query": "x"}'},
            ],
            tool_calls=[{"name": "search", "args": {"query": "x"},
                         "id": "call_1", "type": "tool_call"}],
        ),
        ToolMessage(content='[]', tool_call_id="call_1"),
    ]
    payload = llm._get_request_payload(messages)
    asst_items = [it for it in payload["input"]
                  if isinstance(it, dict)
                  and it.get("type") == "message"
                  and it.get("role") == "assistant"]
    assert len(asst_items) >= 1
    for a in asst_items:
        assert a.get("status") == "completed"


def test_patch_does_not_affect_chat_completions_path():
    """chat_completions doesn't go through ``_construct_responses_api_input``
    at all, so the patch is a non-issue. Still, instantiate a
    chat_completions LLM and serialize messages through ChatOpenAI's
    request builder to confirm no unexpected status field appears."""
    from langchain_core.messages import AIMessage, HumanMessage

    from search_ttl.utils.llm_builder import build_chat_llm

    llm = build_chat_llm(
        model="x", base_url="http://x", api_key="x",
        llm_api_flavor="chat_completions",
        enable_thinking=False,
    )
    payload = llm._get_request_payload([
        HumanMessage(content="q"),
        AIMessage(content="response"),
    ])
    # chat_completions uses ``messages`` not ``input``.
    assert "messages" in payload
    assert "input" not in payload
    # No injected status on chat_completions message dicts.
    for m in payload["messages"]:
        assert "status" not in m, f"unexpected status on chat_completions msg: {m}"
