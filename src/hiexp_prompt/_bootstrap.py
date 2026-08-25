"""Common bootstrap for hiexp CLI scripts.

Each entry point in ``scripts/hiexp_prompt/`` should call :func:`bootstrap`
*before* typer parses CLI args. Two reasons:

1. **Load the project root ``.env`` into ``os.environ``** so any typer
   ``envvar=...`` defaults can read deployment endpoint values
   (``AGENT_LLM_*``, ``AGENT_MCP_URL``, ``EVAL_JUDGE_*``).
   ``HiExpConfig.from_omegaconf`` also calls ``load_dotenv`` internally,
   but that runs after typer parsing — too late for typer-level
   ``envvar=`` defaults that need the value at CLI-parse time.

2. **Install the terminal cursor-restore handler** so Rich's hidden
   cursor is recovered on Ctrl-C / SIGTERM.

Both actions are idempotent.
"""

from __future__ import annotations

from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DONE = False


def bootstrap() -> None:
    """Load ``<project_root>/.env`` and install the cursor restore handler."""
    global _DONE
    if _DONE:
        return
    _DONE = True

    try:
        from dotenv import load_dotenv

        load_dotenv(_PROJECT_ROOT / ".env", override=False)
    except Exception:
        # python-dotenv is a hard dep, but never block CLI startup if
        # something goes sideways — typer will still see whatever is
        # already in os.environ.
        pass

    from hiexp_prompt._term import install_cursor_restore

    install_cursor_restore()
