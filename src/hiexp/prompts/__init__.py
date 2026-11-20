"""Versioned prompt templates and renderers.

- ``system_prompt``: HiExp paper Table 7 system prompt (with ``{experience}`` slot).
- ``question_templates``: hiexp_boxed / browsecomp_plus / raw user-turn templates.
- ``render_experience``: format E2/E3 lists for the system prompt and E1 lists for tool responses.
"""

from hiexp.prompts.question_templates import (
    BROWSECOMP_PLUS_TEMPLATE,
    HIEXP_BOXED_TEMPLATE,
    render_question,
)
from hiexp.prompts.render_experience import render_e1_prefix, render_e2_block
from hiexp.prompts.system_prompt import (
    HIEXP_EXPERIENCE_SECTION,
    HIEXP_SYSTEM_PROMPT,
    HIEXP_SYSTEM_PROMPT_BODY,
    HIEXP_TOOL_CALL_FORMAT_SPEC,
    render_system_prompt,
)

__all__ = [
    "HIEXP_EXPERIENCE_SECTION",
    "HIEXP_SYSTEM_PROMPT",
    "HIEXP_SYSTEM_PROMPT_BODY",
    "HIEXP_TOOL_CALL_FORMAT_SPEC",
    "render_system_prompt",
    "HIEXP_BOXED_TEMPLATE",
    "BROWSECOMP_PLUS_TEMPLATE",
    "render_question",
    "render_e1_prefix",
    "render_e2_block",
]
