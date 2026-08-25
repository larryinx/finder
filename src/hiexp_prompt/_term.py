"""Terminal helpers for the hiexp CLI scripts.

Rich's progress widgets hide the terminal cursor for cleaner rendering.
When ``asyncio.run(...)`` is interrupted by ``Ctrl-C`` (or any signal
that bypasses Rich's normal teardown), the show-cursor ANSI escape
sometimes never gets emitted and the user is left with an invisible
cursor. This module installs an ``atexit`` + signal handler that
emits ``\\x1b[?25h`` unconditionally on shutdown.

Usage at the top of each CLI script::

    from hiexp_prompt._term import install_cursor_restore
    install_cursor_restore()
"""

from __future__ import annotations

import atexit
import signal
import sys


def _show_cursor() -> None:
    try:
        sys.stdout.write("\x1b[?25h")
        sys.stdout.flush()
    except Exception:
        # Best-effort; never block exit.
        pass


_INSTALLED = False


def install_cursor_restore() -> None:
    """Idempotently install handlers that re-show the terminal cursor.

    Safe to call multiple times. Restores the cursor on:
      * normal interpreter shutdown (``atexit``);
      * SIGINT (Ctrl-C);
      * SIGTERM, SIGHUP where available.

    After a signal fires we re-raise it with the default disposition so
    the process exits with the conventional 130/143/etc. status code.
    """
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    atexit.register(_show_cursor)

    def _on_signal(signum, frame):  # noqa: ARG001 — frame unused
        _show_cursor()
        signal.signal(signum, signal.SIG_DFL)
        try:
            signal.raise_signal(signum)
        except Exception:
            sys.exit(128 + int(signum))

    for sig_name in ("SIGINT", "SIGTERM", "SIGHUP"):
        sig = getattr(signal, sig_name, None)
        if sig is None:
            continue
        try:
            signal.signal(sig, _on_signal)
        except (OSError, ValueError):
            # SIGHUP isn't available on Windows; ignore.
            continue
