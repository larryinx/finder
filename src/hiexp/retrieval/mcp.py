"""MCP search-tool loader.

Wraps :class:`langchain_mcp_adapters.client.MultiServerMCPClient` and
selects the tools the HiExp agent needs by name (``search`` and
optionally ``get_document``, matching
``third_party/BrowseComp-Plus/searcher/tools.py:21,69``).

Design notes:
- Loading MCP tools is async; we expose both an async (:meth:`aload`) and
  a sync (:meth:`load`) facade so the agent's ``__init__`` stays
  synchronous.
- The loader does NOT execute tool calls itself — the returned
  ``BaseTool`` instances are bound to the LLM in M5 and invoked by
  LangGraph's tool node.
- The loader does NOT touch HEK retrieval. E1 injection happens *around*
  these tools in the agent's custom ``tool_node``, not inside the tool.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from hiexp.config import HiExpConfig

logger = logging.getLogger(__name__)

# BrowseComp-Plus's mcp_server.py uses ``streamable-http`` (hyphen) as a
# CLI choice, while langchain-mcp-adapters' Connection TypedDict uses
# ``streamable_http`` (underscore). Normalize on the way in.
_TRANSPORT_ALIAS = {
    "sse": "sse",
    "streamable-http": "streamable_http",
    "streamable_http": "streamable_http",
    "stdio": "stdio",
}

# Default name we register the MCP server under in the client connections
# dict. Matches BrowseComp-Plus's ``qwen_client.py:43`` so log messages
# stay comparable.
_SERVER_NAME = "search-server"


class MCPToolLoader:
    """Load and filter the search tools exposed by an MCP server.

    Example:
        >>> from hiexp import HiExpConfig, compose
        >>> cfg = HiExpConfig.from_omegaconf(
        ...     compose(family="hiexp_prompt", benchmark="bcp")
        ... )
        >>> tools = MCPToolLoader(cfg).load()
        >>> [t.name for t in tools]
        ['search', 'get_document']
    """

    def __init__(self, config: HiExpConfig) -> None:
        self.config = config
        self._client: MultiServerMCPClient | None = None

    # ── Public API ─────────────────────────────────────────────────────────

    async def aload(self) -> list[BaseTool]:
        """Async: connect to the MCP server and return the filtered tools."""
        client = self._get_or_build_client()
        all_tools = await client.get_tools(server_name=_SERVER_NAME)
        return self._filter(all_tools)

    def load(self) -> list[BaseTool]:
        """Sync facade. Runs the event loop once.

        Avoid calling from inside a running loop (Jupyter, FastAPI
        handler, …); use :meth:`aload` there instead.
        """
        return asyncio.run(self.aload())

    @property
    def client(self) -> MultiServerMCPClient:
        """The underlying client (built on first access)."""
        return self._get_or_build_client()

    # ── Internals ──────────────────────────────────────────────────────────

    def _get_or_build_client(self) -> MultiServerMCPClient:
        if self._client is None:
            self._client = MultiServerMCPClient(
                connections={_SERVER_NAME: self._build_connection()}
            )
        return self._client

    def _build_connection(self) -> dict[str, Any]:
        if not self.config.mcp_url:
            raise ValueError(
                "HiExpConfig.mcp_url is empty; set AGENT_MCP_URL or pass it explicitly"
            )
        transport_alias = _TRANSPORT_ALIAS.get(self.config.mcp_transport)
        if transport_alias is None:
            raise ValueError(
                f"Unsupported mcp_transport={self.config.mcp_transport!r}; "
                f"expected one of {sorted(_TRANSPORT_ALIAS)}"
            )
        if transport_alias == "stdio":
            # The HiExp agent always talks to a remote retrieval server in
            # the BrowseComp-Plus / Wikipedia setups; stdio would require
            # spawning a child process and a different config schema.
            raise ValueError(
                "stdio MCP transport is not supported by HiExp agent; "
                "use sse or streamable-http"
            )
        return {"transport": transport_alias, "url": self.config.mcp_url}

    def _filter(self, all_tools: list[BaseTool]) -> list[BaseTool]:
        wanted = {self.config.mcp_search_tool_name}
        if self.config.mcp_get_document_tool_name:
            wanted.add(self.config.mcp_get_document_tool_name)

        kept = [t for t in all_tools if t.name in wanted]
        names = {t.name for t in kept}

        if self.config.mcp_search_tool_name not in names:
            available = sorted(t.name for t in all_tools)
            raise RuntimeError(
                f"MCP server at {self.config.mcp_url!r} does not expose the "
                f"required search tool {self.config.mcp_search_tool_name!r}. "
                f"Available tools: {available}"
            )

        if (
            self.config.mcp_get_document_tool_name
            and self.config.mcp_get_document_tool_name not in names
        ):
            logger.info(
                "MCP server at %s does not expose optional %r tool; continuing without it",
                self.config.mcp_url,
                self.config.mcp_get_document_tool_name,
            )

        # Stable order: search tool first, then get_document if present.
        kept.sort(key=lambda t: 0 if t.name == self.config.mcp_search_tool_name else 1)
        return kept
