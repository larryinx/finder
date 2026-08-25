from __future__ import annotations

from hiexp.graph import _coerce_tool_output_to_str, _extract_docids, _unwrap_mcp_text

# A realistic MCP TextContent payload from BCP's ``search`` tool: one
# text block whose ``text`` field is the JSON-stringified list of
# {docid, score, snippet} dicts.
_MCP_WRAPPER = [
    {
        "type": "text",
        "text": '[{"docid": "86190", "score": 10.87, "snippet": "Queen Arwa University..."}]',
    }
]


def test_unwrap_parses_json_inside_text_block():
    out = _unwrap_mcp_text(_MCP_WRAPPER)
    assert isinstance(out, list)
    assert out[0]["docid"] == "86190"


def test_unwrap_passes_through_plain_lists():
    raw = [{"docid": "1", "snippet": "x"}]
    assert _unwrap_mcp_text(raw) is raw


def test_unwrap_falls_back_to_concatenated_text():
    wrapped = [{"type": "text", "text": "hello "}, {"type": "text", "text": "world"}]
    assert _unwrap_mcp_text(wrapped) == "hello world"


def test_coerce_unwraps_then_dumps_json():
    s = _coerce_tool_output_to_str(_MCP_WRAPPER)
    assert "86190" in s
    # No MCP envelope leaks through to the LLM.
    assert "type" not in s.split("snippet")[0] or '"type": "text"' not in s


def test_extract_docids_handles_mcp_wrapper():
    assert _extract_docids(_MCP_WRAPPER) == ["86190"]


def test_extract_docids_handles_plain_list():
    raw = [{"docid": "x"}, {"docid": "y"}, {"no_docid": "z"}]
    assert _extract_docids(raw) == ["x", "y"]


def test_extract_docids_handles_json_string():
    s = '[{"docid": "alpha"}, {"docid": "beta"}]'
    assert _extract_docids(s) == ["alpha", "beta"]
