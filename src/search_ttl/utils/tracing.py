"""Thin pass-through to ``hiexp.tracing`` so we don't duplicate weave/W&B logic.

Why a wrapper at all: callers should import from ``search_ttl.utils.tracing``
even when the implementation lives upstream — it gives us a hook to add
TTL-specific tags / phases without rewriting hiexp.

This module also re-exports ``langsmith.traceable`` as ``traceable`` so the
rest of ``search_ttl`` has one import surface for span instrumentation. When
``LANGSMITH_TRACING`` is unset/false the langsmith decorator is a no-op, and
when ``langsmith`` itself isn't installed we fall back to a pass-through so
imports don't break.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

try:  # langsmith ships transitively with langchain-core>=1.3
    from langsmith import traceable as _langsmith_traceable  # type: ignore[import-not-found]
    from langsmith.run_helpers import tracing_context as _langsmith_tracing_context  # type: ignore[import-not-found]

    _HAS_LANGSMITH = True
except Exception:  # noqa: BLE001
    _HAS_LANGSMITH = False

    def _langsmith_traceable(*args: Any, **kwargs: Any):  # type: ignore[no-redef]
        # Mirror @traceable's "with or without parens" calling convention.
        if args and callable(args[0]) and not kwargs:
            return args[0]

        def _decorator(fn):
            return fn

        return _decorator

    @contextmanager
    def _langsmith_tracing_context(**_: Any) -> Iterator[None]:  # type: ignore[no-redef]
        yield


def traceable(*args: Any, **kwargs: Any):
    """Re-exported ``langsmith.traceable``.

    Use as ``@traceable`` (auto-named) or ``@traceable(name=..., run_type="chain",
    metadata={...})``. Becomes a pass-through whenever ``LANGSMITH_TRACING`` is
    unset or langsmith isn't importable, so it's safe to leave on in CI.
    """
    return _langsmith_traceable(*args, **kwargs)


@contextmanager
def langsmith_thread(thread_id: str | None, **extra_metadata: Any) -> Iterator[None]:
    """Tag every ``@traceable`` span under this block with a ``thread_id``.

    LangSmith groups runs that share a ``thread_id`` (or ``session_id`` /
    ``conversation_id``) into one *thread* — useful for "all rollouts from
    training step 7" or "this whole eval pass". Pass ``thread_id=None`` to
    no-op (so callers can wrap unconditionally).
    """
    if not thread_id or not _HAS_LANGSMITH:
        yield
        return
    metadata = {"thread_id": thread_id, **extra_metadata}
    with _langsmith_tracing_context(metadata=metadata):
        yield


def init_weave(
    *,
    run_name: str,
    config: dict[str, Any] | None = None,
    tags: list[str] | None = None,
    job_type: str = "ttl",
    phase: str = "ttl_train",
) -> None:
    """Initialise weave/W&B for a TTL run. No-ops when WEAVE_DISABLED=1."""
    try:
        from hiexp.tracing import init_weave as _init_weave  # type: ignore[import-not-found]
    except ImportError:
        return
    _init_weave(
        run_name=run_name,
        config=config,
        tags=tags or [],
        job_type=job_type,
        phase=phase,
    )


def finish() -> None:
    try:
        from hiexp.tracing import finish as _finish  # type: ignore[import-not-found]
    except ImportError:
        return
    _finish()
