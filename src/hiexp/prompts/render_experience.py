"""Render E1/E2/E3 hits into the prompt.

Two functions:
- :func:`render_e2_block` — formats strategy-level hits to fill the
  ``{experience}`` slot in the Table 7 system prompt.
- :func:`render_e1_prefix` — formats case-level hits as a block that
  prefixes a tool response. Mirrors the qualitative example in paper
  Table 9 where ``[Trigger E_1 Experience: <title>] ...`` appears
  *inside* the ``<tool_response>`` envelope.
"""

from __future__ import annotations

from collections.abc import Iterable

from hiexp.experience.schema import ExperienceHit


def render_e2_block(hits: Iterable[ExperienceHit]) -> str:
    """Render strategy-level (E2 / E3) hits as a Markdown bullet list.

    Each bullet uses the *parent* chunk (``description``) per paper §A's
    parent-child architecture: the title is the retrieval key, the
    description is the injected experience.

    Empty input → ``"None."`` so the system prompt still reads well.
    """
    items = list(hits)
    if not items:
        return "None."
    lines = [
        f"- **{h.entry.title.strip()}**: {h.entry.description.strip()}"
        for h in items
    ]
    return "\n".join(lines)


def render_e1_prefix(hits: Iterable[ExperienceHit]) -> str:
    """Render case-level (E1) hits as a prefix block for tool responses.

    Layout follows paper Table 9: ``[Trigger E_1 Experience: <title>]``
    leads each hit, with description as the body. A trailing blank line
    separates the block from the underlying retrieved documents.

    Empty input → empty string (no separator overhead).
    """
    items = list(hits)
    if not items:
        return ""
    lines = ["[Relevant case-based experiences]"]
    for h in items:
        lines.append(
            f"- [Trigger E_1 Experience: {h.entry.title.strip()}] "
            f"{h.entry.description.strip()}"
        )
    return "\n".join(lines) + "\n\n"
