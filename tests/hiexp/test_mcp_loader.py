from __future__ import annotations

from typing import Any

import pytest
from langchain_core.tools import BaseTool

from hiexp import HiExpConfig
from hiexp.retrieval import MCPToolLoader


class _StubTool(BaseTool):
    """Minimal BaseTool stand-in for filter-logic tests."""

    name: str = ""
    description: str = "stub"

    def _run(self, *args, **kwargs) -> Any:
        return ""

    async def _arun(self, *args, **kwargs) -> Any:
        return ""


def _make_cfg(**overrides) -> HiExpConfig:
    base = dict(
        mcp_url="http://127.0.0.1:8080/mcp",
        enable_hek=False,  # skip hek_path requirement
    )
    base.update(overrides)
    cfg = HiExpConfig(**base)
    cfg.validate()
    return cfg


# ── Connection builder ────────────────────────────────────────────────────


def test_build_connection_sse():
    loader = MCPToolLoader(_make_cfg(mcp_transport="sse"))
    conn = loader._build_connection()
    assert conn == {"transport": "sse", "url": "http://127.0.0.1:8080/mcp"}


def test_build_connection_streamable_http_with_hyphen():
    loader = MCPToolLoader(_make_cfg(mcp_transport="streamable-http"))
    conn = loader._build_connection()
    # langchain-mcp-adapters wants underscore form
    assert conn["transport"] == "streamable_http"


def test_build_connection_streamable_http_with_underscore():
    loader = MCPToolLoader(_make_cfg(mcp_transport="streamable_http"))
    conn = loader._build_connection()
    assert conn["transport"] == "streamable_http"


def test_build_connection_rejects_stdio():
    loader = MCPToolLoader(_make_cfg(mcp_transport="stdio"))
    with pytest.raises(ValueError, match="stdio"):
        loader._build_connection()


def test_build_connection_unknown_transport():
    cfg = _make_cfg()
    cfg.mcp_transport = "websocket"  # type: ignore[assignment]
    loader = MCPToolLoader(cfg)
    with pytest.raises(ValueError, match="Unsupported"):
        loader._build_connection()


# ── Filtering ─────────────────────────────────────────────────────────────


def test_filter_keeps_search_and_get_document():
    cfg = _make_cfg()
    loader = MCPToolLoader(cfg)
    tools = [
        _StubTool(name="search"),
        _StubTool(name="get_document"),
        _StubTool(name="some_other"),
    ]
    kept = loader._filter(tools)
    assert [t.name for t in kept] == ["search", "get_document"]


def test_filter_orders_search_first():
    cfg = _make_cfg()
    loader = MCPToolLoader(cfg)
    tools = [_StubTool(name="get_document"), _StubTool(name="search")]
    kept = loader._filter(tools)
    assert [t.name for t in kept] == ["search", "get_document"]


def test_filter_drops_optional_get_document_when_disabled():
    cfg = _make_cfg(mcp_get_document_tool_name=None)
    loader = MCPToolLoader(cfg)
    tools = [_StubTool(name="search"), _StubTool(name="get_document")]
    kept = loader._filter(tools)
    assert [t.name for t in kept] == ["search"]


def test_filter_tolerates_missing_optional_tool():
    cfg = _make_cfg()  # default get_document expected but not exposed
    loader = MCPToolLoader(cfg)
    tools = [_StubTool(name="search")]
    kept = loader._filter(tools)
    assert [t.name for t in kept] == ["search"]


def test_filter_raises_when_search_missing():
    cfg = _make_cfg()
    loader = MCPToolLoader(cfg)
    tools = [_StubTool(name="get_document"), _StubTool(name="other")]
    with pytest.raises(RuntimeError, match="search"):
        loader._filter(tools)


def test_filter_uses_custom_search_tool_name():
    cfg = _make_cfg(mcp_search_tool_name="retrieve")
    loader = MCPToolLoader(cfg)
    tools = [_StubTool(name="retrieve"), _StubTool(name="search")]
    kept = loader._filter(tools)
    assert [t.name for t in kept] == ["retrieve"]


# ── Async loading via mocked client ───────────────────────────────────────


@pytest.mark.asyncio
async def test_aload_uses_mocked_client(monkeypatch):
    cfg = _make_cfg()
    loader = MCPToolLoader(cfg)

    class _FakeClient:
        def __init__(self, *args, **kwargs):
            self._calls: list[dict] = []

        async def get_tools(self, *, server_name: str | None = None):
            self._calls.append({"server_name": server_name})
            return [
                _StubTool(name="search"),
                _StubTool(name="get_document"),
                _StubTool(name="ignore_me"),
            ]

    monkeypatch.setattr("hiexp.retrieval.mcp.MultiServerMCPClient", _FakeClient)
    tools = await loader.aload()
    assert [t.name for t in tools] == ["search", "get_document"]


# ── Misc ──────────────────────────────────────────────────────────────────


def test_loader_requires_mcp_url():
    cfg = _make_cfg()
    cfg.mcp_url = ""  # bypass validate()
    loader = MCPToolLoader(cfg)
    with pytest.raises(ValueError, match="mcp_url"):
        loader._build_connection()
