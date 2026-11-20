"""MCP tool loader.

We delegate to ``hiexp.retrieval.mcp.MCPToolLoader`` so all bug fixes
(transport-name aliasing, ``BaseExceptionGroup`` unwrapping during probe,
async-only tool dispatch) stay in one place. See
``scripts/ace/run.py:_probe_mcp_or_die`` for the canonical probe pattern.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class _MCPLoaderConfig:
    """Shape ``MCPToolLoader`` reads. Keeps ``HiExpConfig`` import out of users.

    ``mcp_transport`` accepts ``"streamable_http"`` (modern, default), the
    hyphenated ``"streamable-http"`` alias, or ``"sse"`` for legacy
    SSE-transport MCP servers (BCP's mcp_server defaults to streamable-http
    but some deployments still expose the older SSE endpoint — symptom is a
    ``HTTP 405 Allow: GET, HEAD`` on POST to ``/mcp``).
    """

    mcp_url: str
    mcp_search_tool_name: str = "search"
    mcp_get_document_tool_name: str = ""  # empty = don't expose get_document
    mcp_transport: str = "streamable_http"


def build_mcp_loader(
    *,
    mcp_url: str,
    search_tool: str = "search",
    transport: str = "streamable_http",
):
    """Construct an MCPToolLoader from a minimal config dict.

    Returns the loader (call ``await loader.aload()`` to get the tools).
    """
    from hiexp.retrieval.mcp import MCPToolLoader  # type: ignore[import-not-found]

    cfg = _MCPLoaderConfig(
        mcp_url=mcp_url,
        mcp_search_tool_name=search_tool,
        mcp_transport=transport,
    )
    return MCPToolLoader(cfg)


async def probe_mcp(
    *,
    mcp_url: str,
    transport: str = "streamable_http",
    timeout_s: float = 15.0,
) -> list[str]:
    """Quick connectivity check; returns list of tool names or raises with a clear msg.

    On ``HTTP 405`` we add a hint to switch transport — a common gotcha when
    pointing at SSE servers with the streamable-http client (or vice versa).
    """
    import asyncio

    loader = build_mcp_loader(mcp_url=mcp_url, transport=transport)

    async def _check() -> list[str]:
        tools = await loader.aload()
        return [t.name for t in tools]

    def _hint_for(exc: BaseException) -> str:
        msg = str(exc)
        if "405" in msg:
            other = "sse" if transport == "streamable_http" else "streamable_http"
            return (
                f"\n  hint: HTTP 405 usually means the server speaks the *other* "
                f"transport. Try `agent.mcp_transport={other}` (currently {transport!r})."
            )
        return ""

    try:
        return await asyncio.wait_for(_check(), timeout=timeout_s)
    except BaseExceptionGroup as eg:  # type: ignore[misc]
        inner = eg.exceptions[0] if eg.exceptions else eg
        raise RuntimeError(
            f"MCP probe failed at {mcp_url} (transport={transport}): "
            f"{type(inner).__name__}: {inner}{_hint_for(inner)}"
        ) from inner
    except (asyncio.TimeoutError, Exception) as exc:  # noqa: BLE001
        raise RuntimeError(
            f"MCP probe failed at {mcp_url} (transport={transport}): "
            f"{type(exc).__name__}: {exc}{_hint_for(exc)}"
        ) from exc
