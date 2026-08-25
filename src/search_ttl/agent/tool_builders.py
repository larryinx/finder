"""Tool providers for SearchAgent.

A ToolBuilder hands the agent (1) a list of LangChain tools to bind to
the LLM and (2) a postprocess hook that converts the raw tool result
into the trio (``str_for_LLM``, ``docids``, ``persisted_output``) the
trajectory expects.

Two providers today:

* ``MCPToolBuilder`` — preserves the existing BCP path. Wraps
  ``hiexp.retrieval.mcp.MCPToolLoader`` and unwraps the ``TextContent``
  envelope so the LLM sees parsed JSON, with docids extracted for the
  efficiency stats.
* ``SerperToolBuilder`` — builds a single ``search(query: str)`` tool
  backed by ``GoogleSerperAPIWrapper``. Returns a flat string; there
  are no docids (BrowseComp doesn't have a vetted corpus), so the
  postprocess hook returns an empty list.

The agent's behaviour for ``tool_provider == mcp`` is byte-identical
to the pre-refactor path.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

LOG = logging.getLogger(__name__)


@dataclass
class ToolPostprocess:
    """Result of converting a raw tool output into agent-shaped pieces."""

    tool_str: str
    docids: list[str]
    persisted_output: Any


class ToolBuilder(ABC):
    """Provider-agnostic tool source for SearchAgent.

    A builder is responsible for connecting (potentially re-connecting,
    in the MCP case) and turning a raw tool result into the shape the
    rollout layer persists.
    """

    #: Name of the primary search tool the agent calls. Must match the
    #: name that ``aload`` returns so ``self._tool_by_name`` can resolve
    #: the tool by name in ``_run_tools``.
    search_tool_name: str = "search"

    @abstractmethod
    async def aload(self) -> list:
        """Return a list of LangChain tools to bind to the LLM."""

    async def reconnect(self) -> list:
        """Default: just re-aload. MCP overrides to rebuild its session."""
        return await self.aload()

    @abstractmethod
    def postprocess(self, raw_result: Any) -> ToolPostprocess:
        """Convert the raw output of ``tool.ainvoke`` into agent pieces."""


# ── MCP provider (BCP) ────────────────────────────────────────────────────


class MCPToolBuilder(ToolBuilder):
    """Preserves the pre-refactor MCP path verbatim."""

    def __init__(
        self,
        *,
        mcp_url: str,
        search_tool_name: str = "search",
        transport: str = "streamable_http",
    ) -> None:
        self.mcp_url = mcp_url
        self.search_tool_name = search_tool_name
        self.transport = transport

    async def aload(self) -> list:
        from search_ttl.agent.tools import build_mcp_loader

        loader = build_mcp_loader(
            mcp_url=self.mcp_url,
            search_tool=self.search_tool_name,
            transport=self.transport,
        )
        return await loader.aload()

    def postprocess(self, raw_result: Any) -> ToolPostprocess:
        # Reuse HiExp's MCP envelope unwrapper / docid extractor so all
        # three pipelines (HiExp, ACE, TTL) share one MCP contract.
        from hiexp.graph import (  # type: ignore[import-not-found]
            _coerce_tool_output_to_str,
            _extract_docids,
            _unwrap_mcp_text,
        )

        unwrapped = _unwrap_mcp_text(raw_result)
        tool_str = _coerce_tool_output_to_str(unwrapped)
        docids = list(_extract_docids(unwrapped))
        persisted = (
            unwrapped if isinstance(unwrapped, list) else [{"text": tool_str}]
        )
        return ToolPostprocess(
            tool_str=tool_str, docids=docids, persisted_output=persisted
        )


# ── Serper provider (BrowseComp) ──────────────────────────────────────────


def _serper_key_exhausted(exc: Exception) -> bool:
    """True if the exception looks like a Serper credit / quota / bad-key
    failure (as opposed to a transient network error or a real bug).

    Serper signals an exhausted key as ``HTTP 400`` with body
    ``{"message":"Not enough credits"}`` — the exception *string* is a
    generic "400 Client Error", so we have to read ``response``.
    """
    resp = getattr(exc, "response", None)
    if resp is None:
        return False
    status = getattr(resp, "status_code", None)
    if status in (401, 402, 403, 429):  # unauthorized / payment / forbidden / rate-limited
        return True
    if status == 400:
        body = (getattr(resp, "text", "") or "").lower()
        return any(w in body for w in ("credit", "quota", "limit"))
    return False


class SerperToolBuilder(ToolBuilder):
    """Single ``search(query: str)`` tool backed by Google Serper.

    Accepts one or more API keys. On a credit/quota failure the tool
    fails over to the next key and sticks to it for subsequent calls,
    so an exhausted primary key costs at most one wasted request.
    """

    def __init__(
        self,
        *,
        api_keys: list[str],
        num_results: int = 10,
        gl: str = "us",
        hl: str = "en",
        search_type: str = "search",
        search_tool_name: str = "search",
    ) -> None:
        keys = [k.strip() for k in api_keys if k and k.strip()]
        # Preserve order, drop duplicates (a repeated key buys nothing).
        seen: set[str] = set()
        self.api_keys = [k for k in keys if not (k in seen or seen.add(k))]
        if not self.api_keys:
            raise ValueError(
                "SerperToolBuilder requires an API key — set SERPER_API_KEY "
                "or agent.serper.api_key in the run config."
            )
        self.num_results = int(num_results)
        self.gl = gl
        self.hl = hl
        self.search_type = search_type
        self.search_tool_name = search_tool_name
        # Index of the key to try first; advances on quota failures and
        # then stays there — once a key is exhausted every later call
        # starts from its successor, so the dead key is never retried.
        self._key_idx = 0
        self._wrappers: dict[str, Any] = {}
        # Keys already known exhausted — used to log the failover once
        # (not once per concurrent in-flight call during the switch).
        self._exhausted: set[str] = set()

    def _wrapper_for(self, api_key: str):
        """Lazily build + cache a GoogleSerperAPIWrapper per key."""
        w = self._wrappers.get(api_key)
        if w is None:
            from langchain_community.utilities import GoogleSerperAPIWrapper

            w = GoogleSerperAPIWrapper(
                serper_api_key=api_key,
                k=self.num_results,
                gl=self.gl,
                hl=self.hl,
                type=self.search_type,
            )
            self._wrappers[api_key] = w
        return w

    async def aload(self) -> list:
        from langchain_core.tools import tool

        name = self.search_tool_name

        @tool(name, parse_docstring=False)
        async def search(query: str) -> str:
            """Search the web via Google. Returns a text snippet bundle."""
            n = len(self.api_keys)
            last_exc: Exception | None = None
            # Snapshot the starting key index so the per-call rotation
            # walks distinct keys even though ``self._key_idx`` is
            # mutated mid-loop (for *future* calls).
            start = self._key_idx
            for offset in range(n):
                idx = (start + offset) % n
                key = self.api_keys[idx]
                wrapper = self._wrapper_for(key)
                try:
                    # GoogleSerperAPIWrapper.run is sync; bounce to a
                    # thread so httpx waiting on serper.dev doesn't
                    # block the agent's event loop.
                    return await asyncio.to_thread(wrapper.run, query)
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    if not _serper_key_exhausted(exc):
                        raise  # transient / real error — surface it
                    has_next = offset < n - 1
                    # Warn once per key, even though several concurrent
                    # calls may each hit the dead key during the switch.
                    if key not in self._exhausted:
                        self._exhausted.add(key)
                        if has_next:
                            LOG.warning(
                                "serper key #%d exhausted — switching to key "
                                "#%d for the rest of the run", idx, (idx + 1) % n,
                            )
                        else:
                            LOG.warning(
                                "serper key #%d exhausted — no backup keys left",
                                idx,
                            )
                    if has_next:
                        # Make the successor the new default key.
                        self._key_idx = (idx + 1) % n
                        continue
                    raise
            assert last_exc is not None  # n >= 1, loop ran at least once
            raise last_exc

        return [search]

    def postprocess(self, raw_result: Any) -> ToolPostprocess:
        # Serper returns a flat string; no docid concept.
        tool_str = raw_result if isinstance(raw_result, str) else str(raw_result)
        return ToolPostprocess(
            tool_str=tool_str,
            docids=[],
            persisted_output=[{"text": tool_str}],
        )


# ── DuckDuckGo provider (BrowseComp, key-free) ────────────────────────────


class DuckDuckGoToolBuilder(ToolBuilder):
    """Single ``search(query: str)`` tool backed by DuckDuckGo.

    Free, no API key. Same flat-string contract as the Serper builder
    (``.run()`` returns space-joined result bodies), so postprocess is
    identical — no docids.
    """

    def __init__(
        self,
        *,
        max_results: int = 5,
        region: str = "wt-wt",
        safesearch: str = "moderate",
        time: str | None = "y",
        backend: str = "auto",
        source: str = "text",
        search_tool_name: str = "search",
    ) -> None:
        self.max_results = int(max_results)
        self.region = region
        self.safesearch = safesearch
        self.time = time
        self.backend = backend
        self.source = source
        self.search_tool_name = search_tool_name

    async def aload(self) -> list:
        from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
        from langchain_core.tools import tool

        wrapper = DuckDuckGoSearchAPIWrapper(
            region=self.region,
            safesearch=self.safesearch,
            time=self.time,
            max_results=self.max_results,
            backend=self.backend,
            source=self.source,
        )
        name = self.search_tool_name

        @tool(name, parse_docstring=False)
        async def search(query: str) -> str:
            """Search the web via DuckDuckGo. Returns a text snippet bundle."""
            # DuckDuckGoSearchAPIWrapper.run is sync (httpx under the
            # hood); bounce to a thread so the agent's event loop is
            # not blocked while it waits on the network.
            return await asyncio.to_thread(wrapper.run, query)

        return [search]

    def postprocess(self, raw_result: Any) -> ToolPostprocess:
        # DuckDuckGo returns a flat string; no docid concept.
        tool_str = raw_result if isinstance(raw_result, str) else str(raw_result)
        return ToolPostprocess(
            tool_str=tool_str,
            docids=[],
            persisted_output=[{"text": tool_str}],
        )


# ── Factory ───────────────────────────────────────────────────────────────


def _as_dict(node) -> dict:
    """Coerce a config sub-node (dict / OmegaConf DictConfig / None) to a dict."""
    if not node:
        return {}
    if isinstance(node, dict):
        return node
    try:
        return dict(node)
    except Exception:  # noqa: BLE001
        return {}


def build_tool_builder(agent_cfg) -> ToolBuilder:
    """Pick a ToolBuilder from a ``SearchAgentConfig`` (or duck-typed cfg).

    Reads ``tool_provider`` ∈ {mcp, serper, ddg}. Unknown providers raise.
    """
    provider = getattr(agent_cfg, "tool_provider", "mcp") or "mcp"
    if provider == "mcp":
        return MCPToolBuilder(
            mcp_url=agent_cfg.mcp_url,
            search_tool_name=agent_cfg.mcp_search_tool_name,
            transport=agent_cfg.mcp_transport,
        )
    if provider == "serper":
        s = _as_dict(getattr(agent_cfg, "serper", None))
        # Primary key first, then any backups. ``backup_api_keys`` may
        # be a list or a single string; ``backup_api_key`` is the
        # single-backup shorthand. Empty / missing entries are dropped
        # by SerperToolBuilder.
        raw_backups = s.get("backup_api_keys") or s.get("backup_api_key") or []
        if isinstance(raw_backups, str):
            raw_backups = [raw_backups]
        api_keys = [s.get("api_key"), *raw_backups]
        return SerperToolBuilder(
            api_keys=[k for k in api_keys if k],
            num_results=int(s.get("num_results", 10)),
            gl=str(s.get("gl", "us")),
            hl=str(s.get("hl", "en")),
            search_type=str(s.get("search_type", "search")),
            search_tool_name=agent_cfg.mcp_search_tool_name,
        )
    if provider == "ddg":
        d = _as_dict(getattr(agent_cfg, "ddg", None))
        return DuckDuckGoToolBuilder(
            max_results=int(d.get("max_results", 5)),
            region=str(d.get("region", "wt-wt")),
            safesearch=str(d.get("safesearch", "moderate")),
            time=d.get("time", "y"),
            backend=str(d.get("backend", "auto")),
            source=str(d.get("source", "text")),
            search_tool_name=agent_cfg.mcp_search_tool_name,
        )
    raise ValueError(
        f"unknown agent.tool_provider={provider!r} "
        "(expected 'mcp', 'serper' or 'ddg')"
    )
