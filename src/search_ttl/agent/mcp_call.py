"""Robust MCP tool invocation: timeout + one-shot reconnect on transient errors.

Why this exists: ``langchain-mcp-adapters`` runs an SSE reader as a
background task. If the underlying ``httpx`` connection times out
(``ReadTimeout``) or the network briefly drops, the reader task dies and
the cached ``StructuredTool`` becomes silently broken — every subsequent
``ainvoke`` either hangs or raises. The training run we saw had

    Error in sse_reader   ... httpx.ReadTimeout
    Error in post_writer  ... httpx.ReadTimeout

logged by the MCP library, after which any further reflector / agent
search call would fail.

This helper:

1. Wraps ``ainvoke`` in ``asyncio.wait_for(..., timeout)`` so a wedged
   session can't hang past `timeout_s` (default 90s).
2. On timeout / connection error, calls a caller-supplied
   ``reconnect_fn`` once and retries the call exactly once. Subsequent
   failures fall through and become a ``[tool_error] ...`` ToolMessage.
3. Logs each retry / reconnect at INFO so the failure is visible without
   having to grep the MCP library's stderr.

Both ``SearchAgent._run_tools`` and the reflectors' ``_reflect_one`` loops
go through this — same retry policy everywhere.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable

LOG = logging.getLogger(__name__)

# httpx + httpcore + mcp use these for transient network errors.
_TRANSIENT_EXC_NAMES: frozenset[str] = frozenset({
    "ReadTimeout",
    "ConnectTimeout",
    "WriteTimeout",
    "PoolTimeout",
    "RemoteProtocolError",
    "ConnectError",
    "TimeoutError",
})


def _is_transient(exc: BaseException) -> bool:
    """Whitelist by class name so we don't have to import httpx/httpcore here.

    Also catches ``BaseExceptionGroup`` wrappers from anyio TaskGroups by
    descending into the first inner exception.
    """
    name = type(exc).__name__
    if name in _TRANSIENT_EXC_NAMES:
        return True
    inner_seq = getattr(exc, "exceptions", None)
    if inner_seq:  # ExceptionGroup
        return any(_is_transient(e) for e in inner_seq)
    return False


async def call_with_timeout_and_reconnect(
    *,
    invoke_fn: Callable[[], Awaitable[Any]],
    reconnect_fn: Callable[[], Awaitable[None]] | None,
    timeout_s: float,
    label: str = "tool",
) -> Any:
    """Run ``invoke_fn()`` with a hard timeout; on transient failure, try
    ``reconnect_fn()`` once and retry.

    ``invoke_fn`` is called freshly each attempt (don't cache the
    coroutine — it would already be awaited the second time around).

    Raises whichever exception finally surfaced; callers wrap the call
    in their own ``try / except`` so the LLM gets a ``[tool_error]``
    ToolMessage.
    """
    try:
        return await asyncio.wait_for(invoke_fn(), timeout=timeout_s)
    except (asyncio.TimeoutError, BaseException) as first_exc:  # noqa: BLE001
        if not (isinstance(first_exc, asyncio.TimeoutError) or _is_transient(first_exc)):
            raise

        LOG.warning(
            "%s call hit transient error (%s: %s); reconnecting and retrying once",
            label, type(first_exc).__name__, first_exc,
        )

        if reconnect_fn is not None:
            try:
                await asyncio.wait_for(reconnect_fn(), timeout=timeout_s)
            except Exception as reconnect_exc:  # noqa: BLE001
                LOG.warning(
                    "%s reconnect failed: %s — surfacing original error",
                    label, reconnect_exc,
                )
                raise first_exc from reconnect_exc

        # One-shot retry. If THIS one fails too we let the exception
        # bubble out — the caller will turn it into a [tool_error]
        # ToolMessage so the LLM can decide what to do.
        return await asyncio.wait_for(invoke_fn(), timeout=timeout_s)
