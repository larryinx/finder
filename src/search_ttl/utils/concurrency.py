"""Async helpers — semaphore + gather with limited fanout."""

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, Iterable, TypeVar

T = TypeVar("T")


async def bounded_gather(
    coros_factories: Iterable[Callable[[], Awaitable[T]]],
    *,
    concurrency: int,
    return_exceptions: bool = False,
) -> list[T]:
    """Run a sequence of coroutine factories with a semaphore cap.

    We accept *factories* rather than already-created coroutines so the
    semaphore acquisition happens before the awaitable is materialised —
    avoids spawning thousands of pending tasks when concurrency is small.
    """
    sem = asyncio.Semaphore(concurrency)

    async def _one(factory: Callable[[], Awaitable[T]]) -> T:
        async with sem:
            return await factory()

    return await asyncio.gather(
        *[_one(f) for f in coros_factories],
        return_exceptions=return_exceptions,
    )
