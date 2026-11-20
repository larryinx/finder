"""Shared rich console + small helpers.

Modeled after ``scripts/hiexp_prompt/train/build_hek.py``: one ``Console``
singleton, ``console.rule(...)`` for phase markers, colored tags inside
``console.print`` for status. Stays a thin layer so the trainer doesn't
sprout a logging framework — every helper here is one rich call deep.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

# Single console for the whole TTL pipeline; using one so rich's Live regions
# (Progress) don't fight each other across modules.
CONSOLE: Console = Console()

# Tag every line so user can grep `[ttl]` in long logs.
_PREFIX = "[bold cyan]\\[ttl][/bold cyan]"


def banner(title: str, subtitle: str | None = None) -> None:
    CONSOLE.rule(f"{_PREFIX} [bold]{title}[/bold]")
    if subtitle:
        CONSOLE.print(f"{_PREFIX}   {subtitle}")


def section(title: str) -> None:
    """A phase divider — like ``console.rule('exec ...')`` in build_hek.py."""
    CONSOLE.rule(f"{_PREFIX} {title}")


def info(msg: str) -> None:
    CONSOLE.print(f"{_PREFIX} {msg}")


def warn(msg: str) -> None:
    CONSOLE.print(f"{_PREFIX} [yellow]warn[/yellow] {msg}")


def good(msg: str) -> None:
    CONSOLE.print(f"{_PREFIX} [green]✓[/green] {msg}")


def bad(msg: str) -> None:
    CONSOLE.print(f"{_PREFIX} [red]✗[/red] {msg}")


def kv_block(title: str, pairs: Mapping[str, object]) -> None:
    CONSOLE.print(f"{_PREFIX} [bold]{title}[/bold]")
    width = max((len(k) for k in pairs), default=0)
    for k, v in pairs.items():
        CONSOLE.print(f"{_PREFIX}   {k:<{width}}  {v}")


def progress_bar(description: str, total: int) -> tuple[Progress, int]:
    """Make a transient Progress that reuses the shared CONSOLE.

    Caller pattern::

        with progress_bar('rollout step 1/125', 16) as (progress, task_id):
            ...
            progress.update(task_id, advance=1, description='...')

    Returns a context manager built around the rich Progress.
    """
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        transient=True,
        console=CONSOLE,
    )
    return progress, description, total


def status_counts(items: Iterable[object], key: str = "status") -> dict[str, int]:
    """Count categorical attribute values — used for per-phase summaries."""
    out: dict[str, int] = {}
    for it in items:
        v = getattr(it, key, None)
        if v is None and isinstance(it, dict):
            v = it.get(key)
        out[str(v)] = out.get(str(v), 0) + 1
    return out


def fmt_counts(counts: Mapping[str, int]) -> str:
    if not counts:
        return "(empty)"
    return " ".join(f"{k}={v}" for k, v in sorted(counts.items()))


def fmt_seconds(s: float) -> str:
    if s < 60:
        return f"{s:.1f}s"
    m, sec = divmod(s, 60)
    if m < 60:
        return f"{int(m)}m{sec:.0f}s"
    h, m = divmod(m, 60)
    return f"{int(h)}h{int(m)}m"
