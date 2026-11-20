"""Lock the SearchAgent prompts to HiExp baseline parity.

When the experience playbook is empty, the rendered system prompt MUST
match `HIEXP_SYSTEM_PROMPT_BODY` byte-for-byte. The rendered first user
message MUST match `BROWSECOMP_PLUS_TEMPLATE`. These two contracts are
why our pipeline can produce committed answers on this LLM at all
(diagnosed in docs/stage1/ttl-search/12_prompt_and_thinking_diagnosis.md);
breaking them silently regresses accuracy to ~0.

The non-empty case adds the `<experience>...</experience>` block — it
must be additive only (never edit the baseline body).
"""

from __future__ import annotations

from search_ttl.agent.prompts import (
    EMPTY_EXPERIENCE_SENTINEL,
    render_system_prompt,
    render_user_prompt,
)


def test_system_prompt_empty_experience_matches_hiexp_body_exactly():
    from hiexp.prompts.system_prompt import HIEXP_SYSTEM_PROMPT_BODY  # type: ignore[import-not-found]

    assert render_system_prompt(None) == HIEXP_SYSTEM_PROMPT_BODY
    assert render_system_prompt("") == HIEXP_SYSTEM_PROMPT_BODY
    assert render_system_prompt("   ") == HIEXP_SYSTEM_PROMPT_BODY
    assert render_system_prompt(EMPTY_EXPERIENCE_SENTINEL) == HIEXP_SYSTEM_PROMPT_BODY


def test_system_prompt_with_experience_appends_block():
    from hiexp.prompts.system_prompt import HIEXP_SYSTEM_PROMPT_BODY  # type: ignore[import-not-found]

    rendered = render_system_prompt("Add a country prior early.")
    assert rendered.startswith(HIEXP_SYSTEM_PROMPT_BODY), (
        "experience injection must NOT modify the baseline body"
    )
    assert "<experience>" in rendered
    assert "Add a country prior early." in rendered
    assert "</experience>" in rendered


def test_user_prompt_matches_bcp_template_verbatim():
    from hiexp.prompts.question_templates import BROWSECOMP_PLUS_TEMPLATE  # type: ignore[import-not-found]

    q = "What is the capital of France?"
    assert render_user_prompt(q) == BROWSECOMP_PLUS_TEMPLATE.format(question=q)


def test_user_prompt_contains_required_format_anchors():
    """Even if BCP changes upstream, our prompt must keep the autograder anchors."""
    rendered = render_user_prompt("Q")
    assert "Explanation:" in rendered
    assert "Exact Answer:" in rendered
    assert "Confidence:" in rendered
