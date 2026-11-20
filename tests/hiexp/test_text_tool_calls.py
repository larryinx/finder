from __future__ import annotations

from hiexp.graph import _parse_text_tool_calls


def test_single_tool_call():
    text = '<tool_call>\n{"name": "search", "arguments": {"query": "x"}}\n</tool_call>'
    out = _parse_text_tool_calls(text)
    assert len(out) == 1
    assert out[0]["name"] == "search"
    assert out[0]["args"] == {"query": "x"}
    assert out[0]["id"] == "call_text_0"


def test_multiple_tool_calls_distinct_ids():
    text = (
        '<tool_call>{"name": "search", "arguments": {"query": "a"}}</tool_call>'
        ' some prose '
        '<tool_call>{"name": "search", "arguments": {"query": "b"}}</tool_call>'
    )
    out = _parse_text_tool_calls(text)
    assert [c["args"]["query"] for c in out] == ["a", "b"]
    assert {c["id"] for c in out} == {"call_text_0", "call_text_1"}


def test_no_tool_calls_returns_empty():
    assert _parse_text_tool_calls("just plain text") == []
    assert _parse_text_tool_calls("") == []


def test_args_key_aliases_to_arguments():
    text = '<tool_call>{"name": "x", "args": {"q": 1}}</tool_call>'
    out = _parse_text_tool_calls(text)
    assert out[0]["args"] == {"q": 1}


def test_fenced_json_inside_tool_call():
    text = '<tool_call>\n```json\n{"name": "search", "arguments": {"query": "z"}}\n```\n</tool_call>'
    out = _parse_text_tool_calls(text)
    assert out[0]["name"] == "search"
    assert out[0]["args"] == {"query": "z"}


def test_skips_blocks_without_name():
    text = '<tool_call>{"arguments": {}}</tool_call><tool_call>{"name": "ok"}</tool_call>'
    out = _parse_text_tool_calls(text)
    assert len(out) == 1
    assert out[0]["name"] == "ok"


def test_hybrid_json_name_xml_parameter_blocks():
    """Some Qwen-family models emit a malformed mix when vLLM has no tool parser:

        <tool_call>
        {"name": "search", "<parameter=query>
        2002 three-day event ...
        </parameter>
        </function>
        </tool_call>

    We extract the name + ``<parameter=...>`` args anyway.
    """
    text = (
        "<tool_call>\n"
        '{"name": "search", "<parameter=query>\n'
        "2002 three-day event Thursday Saturday\n"
        "</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )
    out = _parse_text_tool_calls(text)
    assert len(out) == 1
    assert out[0]["name"] == "search"
    assert "2002" in out[0]["args"]["query"]


def test_function_open_tag_alternative():
    text = (
        "<tool_call>\n"
        "<function=search>\n"
        "<parameter=query>foo</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )
    out = _parse_text_tool_calls(text)
    assert len(out) == 1
    assert out[0]["name"] == "search"
    assert out[0]["args"]["query"] == "foo"


def test_function_block_outside_tool_call_envelope():
    """Some SGLang/Qwen3 outputs emit an empty ``<tool_call></tool_call>``
    followed by a free ``<function=name><parameter=...>...</parameter></function>``.
    The standalone <function=…> must still be recovered.
    """
    text = (
        "<tool_call> </tool_call> <function=search>\n"
        "<parameter=query>\n"
        "Ghanaian doctor Belgian ship Copacabana Scotland\n"
        "</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )
    out = _parse_text_tool_calls(text)
    assert len(out) == 1
    assert out[0]["name"] == "search"
    assert "Ghanaian doctor" in out[0]["args"]["query"]


def test_pure_function_block_no_tool_call_wrapper():
    """``<function=…>…</function>`` alone, no <tool_call> at all."""
    text = (
        "<function=search>\n"
        "<parameter=query>Paris capital</parameter>\n"
        "</function>"
    )
    out = _parse_text_tool_calls(text)
    assert len(out) == 1
    assert out[0]["name"] == "search"
    assert out[0]["args"]["query"] == "Paris capital"


def test_function_block_multiple_params():
    text = (
        "<function=lookup>"
        "<parameter=key>x</parameter>"
        "<parameter=field>title</parameter>"
        "</function>"
    )
    out = _parse_text_tool_calls(text)
    assert len(out) == 1
    assert out[0]["args"] == {"key": "x", "field": "title"}


def test_no_args_function_block_skipped():
    """Empty body → no args → don't emit a half-formed tool call."""
    out = _parse_text_tool_calls("<function=search></function>")
    assert out == []
