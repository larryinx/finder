"""MCP-only retrieval surface.

The HiExp agent does **not** ship an embedded retriever. All search
calls go through an external MCP server (BrowseComp-Plus
``searcher/mcp_server.py`` or any compatible deployment), reached via
``HiExpConfig.mcp_url``.

Public:
    MCPToolLoader: connects to the MCP server and filters out the
        ``search`` (mandatory) and ``get_document`` (optional) tools.

Standing up the MCP server itself (BM25 / FAISS / Qwen3-Embedding /
multilingual-e5-base) is **out of scope** of this package — see
``third_party/BrowseComp-Plus/searcher`` for that.
"""

from hiexp.retrieval.mcp import MCPToolLoader

__all__ = ["MCPToolLoader"]
