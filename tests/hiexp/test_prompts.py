from __future__ import annotations

from hiexp.experience.schema import ExperienceEntry, ExperienceHit
from hiexp.prompts import (
    BROWSECOMP_PLUS_TEMPLATE,
    HIEXP_BOXED_TEMPLATE,
    HIEXP_SYSTEM_PROMPT,
    render_e1_prefix,
    render_e2_block,
    render_question,
    render_system_prompt,
)


def _hit(level: int, title: str, description: str, score: float = 0.9) -> ExperienceHit:
    return ExperienceHit(
        entry=ExperienceEntry(level=level, title=title, description=description),
        score=score,
    )


# ── system prompt ─────────────────────────────────────────────────────────


def test_system_prompt_paper_anchors():
    out = render_system_prompt("- exp 1")
    # Paper Table 7 protocol anchors live in the format-spec block,
    # which is included by default. Doubled braces in the source
    # collapse to single braces after .format().
    assert "<think>" in out and "</think>" in out
    assert "<tool_call>" in out and "</tool_call>" in out
    assert r"\boxed{}" in out
    assert r"\boxed{20C}" in out
    # Format-example brace collapse — single braces in rendered output.
    assert '{"name": <function-name>' in out
    assert '{{"name"' not in out
    # Experience block content present.
    assert "- exp 1" in out


def test_system_prompt_empty_experience_drops_section():
    """Empty experience → the entire 'Here are some relevant …' line is dropped.

    Previously we rendered ``"None."`` as the slot value; with the move
    to a conditional section, the model no longer sees a placeholder
    line with no content below it.
    """
    out = render_system_prompt("")
    assert "{experience}" not in out
    # The intro line is gone
    assert "Here are some relevant reasoning experience" not in out
    # And the bare "None." sentinel is gone too
    assert "None." not in out


def test_system_prompt_none_sentinel_drops_section():
    """``"None."`` (what render_e2_block returns on no hits) also drops the section."""
    out = render_system_prompt("None.")
    assert "Here are some relevant reasoning experience" not in out


def test_system_prompt_experience_block_kept_when_present():
    out = render_system_prompt("- exp 1\n- exp 2")
    assert "Here are some relevant reasoning experience" in out
    assert "- exp 1" in out
    assert "- exp 2" in out


def test_system_prompt_template_constant_has_slot():
    assert "{experience}" in HIEXP_SYSTEM_PROMPT


def test_system_prompt_drops_format_spec_when_disabled():
    """With include_tool_call_template=False, the entire protocol block
    (think/tool_call/boxed contract + JSON example) is dropped — not
    just the JSON example."""
    full = render_system_prompt("- exp 1", include_tool_call_template=True)
    trimmed = render_system_prompt("- exp 1", include_tool_call_template=False)
    # Experience content stays in both
    assert "- exp 1" in full
    assert "- exp 1" in trimmed
    # The whole protocol block, including <think> and \boxed{}, is now
    # part of the toggleable spec — gone when disabled.
    for needle in ("<think>", r"\boxed{}", '{"name": <function-name>'):
        assert needle in full
        assert needle not in trimmed
    assert len(trimmed) < len(full)


# ── question templates ───────────────────────────────────────────────────


def test_render_question_hiexp_boxed():
    out = render_question("Who wrote Hamlet?", "hiexp_boxed")
    assert r"\boxed{YOUR_ANSWER}" in out
    assert "Who wrote Hamlet?" in out


def test_render_question_browsecomp_plus():
    out = render_question("What is X?", "browsecomp_plus")
    # BrowseComp-Plus contract anchors (prompts.py:12-21)
    assert "Explanation:" in out
    assert "Exact Answer:" in out
    assert "Confidence:" in out
    assert "What is X?" in out


def test_render_question_raw():
    assert render_question("foo", "raw") == "foo"


def test_render_question_unknown_raises():
    import pytest

    with pytest.raises(ValueError, match="unknown"):
        render_question("foo", "bogus")


def test_browsecomp_template_has_no_unfilled_format_slots():
    # Sanity: the literal ``{your explanation ...}`` placeholders are
    # escaped (they show up in the rendered text as literal braces).
    rendered = BROWSECOMP_PLUS_TEMPLATE.format(question="Q")
    assert "{your succinct" in rendered  # literal user-facing placeholder
    assert "Q" in rendered


# ── E2 block ──────────────────────────────────────────────────────────────


def test_render_e2_block_empty():
    assert render_e2_block([]) == "None."


def test_render_e2_block_uses_descriptions():
    hits = [
        _hit(2, "Multi-hop decomposition", "Break the question into time and attribute."),
        _hit(2, "Verify before commit", "Cross-check across two sources."),
    ]
    out = render_e2_block(hits)
    assert "Multi-hop decomposition" in out
    assert "Break the question into time and attribute." in out
    assert "Verify before commit" in out
    # Bullet markers — Markdown style
    assert out.startswith("- ")


# ── E1 prefix ─────────────────────────────────────────────────────────────


def test_render_e1_prefix_empty():
    assert render_e1_prefix([]) == ""


def test_render_e1_prefix_uses_table_9_format():
    hits = [_hit(1, "Avoid premiere date confusion", "Premiere ≠ composition.")]
    out = render_e1_prefix(hits)
    # Anchor on paper Table 9 phrasing.
    assert "[Trigger E_1 Experience: Avoid premiere date confusion]" in out
    assert "Premiere ≠ composition." in out
    assert out.endswith("\n\n")
