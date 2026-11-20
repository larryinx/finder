"""Render search-tool results for downstream prompts at four verbosities.

Used by:

- ``reflector.querychain_tactic`` / ``reflector.querychain_staged``
  user prompt (the agent's own and the contrast trajectory renders)
- the same reflectors' ToolMessage content when they call the search
  tool live (independent knob)

Visibility modes (``Visibility`` literal):

- ``full``    — JSON-serialise the whole list of search hits.
- ``clipped`` — per-hit ``[i] docid=… title=… <snippet head>``; each
                snippet is truncated to ``max_chars_per_hit`` (default
                500), and the whole rendered block is bounded by
                ``max_chars_total`` (default 1500). When a hit would
                exceed the *remaining* total budget, that hit is
                truncated mid-text with a ``…(truncated)`` marker and
                a trailing ``... (N more)`` summarises how many hits
                were dropped. So setting ``total=500`` reliably yields
                ~500 chars of output, even when the first hit's snippet
                alone is longer than 500 — it gets truncated to fit
                instead of being elided entirely.
- ``title``   — per-hit ``[i] docid=… title=…`` only. Title is taken
                from the entry's ``title`` field, falling back to a
                regex on the YAML-style front matter that BCP's MCP
                server prepends to each snippet
                (``---\\ntitle: ...\\n---``).
- ``none``    — just ``N docs (ids: [...])`` — no titles, no snippets.

The agent's own loop always runs in effective ``full`` mode (it needs
the snippets to find the answer); these knobs control what *secondary*
consumers (analyzer / reflector / reflector-tool-message) see.

Helper ``describe_visibility_for_prompt`` returns a one-paragraph
explainer to inject into the analyzer / reflector system prompt so the
LLM knows what shape of data to expect (especially that snippets may be
absent or truncated).
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterable, Literal

Visibility = Literal["full", "clipped", "title", "none"]
_VALID_VISIBILITIES: frozenset[str] = frozenset({"full", "clipped", "title", "none"})

# Default per-hit snippet budget for ``clipped`` mode. Overridable via
# ``render_tool_result(max_chars_per_hit=...)`` so reward / reflector can
# tune how much of each individual search hit's snippet body the
# downstream LLM sees.
_DEFAULT_CLIPPED_PER_HIT_CHARS = 500

# Regex used to fish a title out of BCP's snippet front matter. Example:
#     "---\ntitle: Foo paper\nauthor: ...\ndate: ...\n---\n<body>"
# matches "Foo paper" (group 1).
_FRONT_MATTER_TITLE_RE = re.compile(
    r"^\s*---\s*\n.*?\btitle:\s*(.+?)\s*\n", re.IGNORECASE | re.DOTALL
)


def _entries_from_output(raw: Any) -> list[dict]:
    """Coerce whatever the trajectory step / tool returned into list[dict]."""
    if raw is None:
        return []
    if isinstance(raw, list):
        return [e if isinstance(e, dict) else {"text": str(e)} for e in raw]
    if isinstance(raw, dict):
        return [raw]
    if isinstance(raw, str):
        try:
            v = json.loads(raw)
            if isinstance(v, list):
                return [e if isinstance(e, dict) else {"text": str(e)} for e in v]
            if isinstance(v, dict):
                return [v]
        except Exception:  # noqa: BLE001
            pass
        return [{"text": raw}]
    return [{"text": str(raw)}]


def _docids(entries: Iterable[dict]) -> list[str]:
    out: list[str] = []
    for e in entries:
        d = e.get("docid") or e.get("doc_id") or e.get("id")
        if d is not None:
            out.append(str(d))
    return out


def _entry_title(entry: dict) -> str:
    """Best-effort title for a search hit.

    Order of preference:
      1. ``entry.title`` (if non-empty)
      2. Front-matter ``title:`` line at the head of ``entry.snippet``
         (BCP's docs are stored as markdown with a YAML-like header)
      3. First non-empty line of the snippet, capped at ~140 chars.
    """
    title = (entry.get("title") or "").strip()
    if title:
        return title[:200]
    snippet = entry.get("snippet") or entry.get("content") or entry.get("text") or ""
    snippet = str(snippet)
    m = _FRONT_MATTER_TITLE_RE.match(snippet)
    if m:
        return m.group(1).strip()[:200]
    # Fallback: first non-empty, non-`---` line.
    for line in snippet.splitlines():
        line = line.strip()
        if line and not line.startswith("---"):
            return line[:140]
    return ""


_TOOL_CALL_BLOCK_RE = re.compile(
    r"<tool_call>\s*\{.*?\}\s*</tool_call>", re.DOTALL | re.IGNORECASE
)


def strip_tool_call_blocks(text: str) -> str:
    """Remove inline ``<tool_call>{...}</tool_call>`` blocks from
    assistant-content text.

    Some inference servers (vLLM/SGLang serving Qwen3 without
    ``--tool-call-parser``) emit tool calls as text inside ``content``
    rather than as structured ``ai.tool_calls``. The rollout agent's
    text-form fallback parses these into a real tool_call step that
    appears later in the trajectory — so leaving the raw
    ``<tool_call>`` block inside the rendered ``assistant:`` line just
    duplicates information for the downstream LLM (it would see the
    query twice: once inside the block, once as the next ``search:``
    step). Strip it.

    Returns the input with all such blocks removed and whitespace
    collapsed at the seams. The caller decides whether the remainder
    is substantive enough to render at all (see ``has_substantive_text``).
    """
    if not text or "<tool_call>" not in text:
        return text
    cleaned = _TOOL_CALL_BLOCK_RE.sub("", text)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


def has_substantive_text(text: str, *, min_chars: int = 4) -> bool:
    """True iff ``text`` has more than ``min_chars`` non-whitespace
    characters after stripping. Used to decide whether an
    ``output_text`` step is worth rendering / counts as evidence of
    assistant reasoning post tool-call removal."""
    return len((text or "").strip()) >= min_chars


def _strip_front_matter(snippet: str) -> str:
    """Drop the leading ``---\\n…\\n---\\n`` block so the snippet body
    begins with actual content, not metadata duplicating the title.

    No-op if there's no front matter.
    """
    if not snippet:
        return snippet
    m = re.match(r"^\s*---\s*\n.*?\n---\s*\n", snippet, re.DOTALL)
    if m:
        return snippet[m.end():]
    return snippet


def render_tool_result(
    raw: Any,
    *,
    visibility: Visibility = "clipped",
    max_chars_total: int = 1500,
    max_chars_per_hit: int = _DEFAULT_CLIPPED_PER_HIT_CHARS,
    head_n: int = 10,
) -> str:
    """Stringify a search-tool result for prompts. See module docstring.

    Args:
        max_chars_per_hit: per-snippet truncation in ``clipped`` mode.
            Each search hit's snippet body is truncated to this many
            chars before the header (`[i] docid=… title=…`) is prefixed.
            Ignored in ``none`` / ``title`` / ``full`` modes.
        max_chars_total: hard ceiling on the *total* rendered size in
            ``clipped`` mode; once accumulated chars would exceed this,
            remaining hits are elided as ``... (N more)``. Set this
            comfortably larger than ``max_chars_per_hit × expected_k``
            (typical search returns 5 hits) — otherwise the first hit
            already trips the ceiling and you get an empty render.
    """
    if visibility not in _VALID_VISIBILITIES:
        raise ValueError(
            f"unknown tool_result visibility {visibility!r}; "
            f"expected one of {sorted(_VALID_VISIBILITIES)}"
        )
    entries = _entries_from_output(raw)
    if not entries:
        return "(no results)"

    if visibility == "none":
        ids = _docids(entries)
        return f"{len(entries)} docs (ids: {ids[:head_n]})"

    if visibility == "title":
        lines: list[str] = []
        for i, e in enumerate(entries[:head_n]):
            doc_id = e.get("docid") or e.get("doc_id") or e.get("id") or "?"
            lines.append(f"[{i}] docid={doc_id} title={_entry_title(e)!r}")
        if len(entries) > head_n:
            lines.append(f"... ({len(entries) - head_n} more)")
        return "\n".join(lines) or "(no results)"

    if visibility == "full":
        try:
            return json.dumps(entries, ensure_ascii=False, default=str)
        except Exception:  # noqa: BLE001
            return str(entries)

    # ──── clipped ─────────────────────────────────────────────────────────
    # ``max_chars_total`` is treated as a *char budget to fill*, NOT as
    # an "all-or-nothing" cliff: when the next block would exceed the
    # remaining budget we truncate IT to fit (and append a `…(truncated)`
    # marker), then stop with a `... (N more)` tail summarising how many
    # hits were dropped. This means setting ``total=500`` reliably yields
    # ~500 chars of rendered output instead of collapsing to just the
    # `... (5 more)` line when the very first hit is bigger than 500.
    rendered: list[str] = []
    used = 0
    _SEP_LEN = 2  # "\n\n" between blocks
    n_total = len(entries)
    for i, e in enumerate(entries[:head_n]):
        doc_id = e.get("docid") or e.get("doc_id") or e.get("id") or "?"
        title = _entry_title(e)
        snippet = e.get("snippet") or e.get("content") or e.get("text") or ""
        snippet = _strip_front_matter(str(snippet))[:max_chars_per_hit]
        block = f"[{i}] docid={doc_id} title={title!r}\n{snippet}".rstrip()

        remaining = max_chars_total - used
        if remaining <= 0:
            rendered.append(f"... ({n_total - i} more)")
            break
        if len(block) > remaining:
            rendered.append(block[:remaining].rstrip() + " …(truncated)")
            if i + 1 < n_total:
                rendered.append(f"... ({n_total - i - 1} more)")
            break
        rendered.append(block)
        used += len(block) + _SEP_LEN
    return "\n\n".join(rendered) or "(no results)"


def describe_visibility_for_prompt(
    visibility: Visibility,
    *,
    max_chars_total: int,
    max_chars_per_hit: int = _DEFAULT_CLIPPED_PER_HIT_CHARS,
    reasoning_chars_per_step: int | None = None,
) -> str:
    """Human-readable explainer for the secondary LLM.

    Embed inside the analyzer / reflector system or user prompt so the
    judge knows what data shape to expect — particularly that snippet
    text may be missing (modes ``none`` / ``title``) or truncated
    (``clipped``), and that reasoning steps are also truncated.
    """
    if visibility not in _VALID_VISIBILITIES:
        raise ValueError(
            f"unknown visibility {visibility!r}; expected one of "
            f"{sorted(_VALID_VISIBILITIES)}"
        )
    descriptions = {
        "none": (
            "the trajectory shows ONLY the count of returned docs and their docids — "
            "no titles, no snippet text. Judge query design from query text alone."
        ),
        "title": (
            "each `results:` block lists hits as `[i] docid=… title=…` — titles only, "
            "no snippet body."
        ),
        "clipped": (
            f"each `results:` block lists hits as `[i] docid=… title=… <snippet>`; "
            f"each snippet is truncated to ~{max_chars_per_hit} chars, and the whole "
            f"`results:` block is bounded by ~{max_chars_total} chars total. If a "
            "hit would exceed the remaining budget it is truncated mid-text with a "
            "`…(truncated)` marker, and `... (N more)` summarises the dropped hits."
        ),
        "full": (
            "each `results:` block contains the complete JSON list of search hits, "
            "including full snippet bodies. Use sparingly — this is verbose."
        ),
    }
    out = "Tool-result visibility: " + descriptions[visibility]
    if reasoning_chars_per_step is not None:
        out += (
            f" Reasoning steps may be truncated to ~{reasoning_chars_per_step} "
            "chars per turn — assume earlier reasoning is condensed, not always "
            "complete."
        )
    return out
