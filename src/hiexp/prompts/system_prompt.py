"""System prompt for the HiExp agent.

Three independent parts, composed by :func:`render_system_prompt`:

* :data:`HIEXP_SYSTEM_PROMPT_BODY` — top-of-prompt framing
  ("In this environment you have access to a set of tools …"). Always
  rendered.
* :data:`HIEXP_EXPERIENCE_SECTION` — the HEK injection block
  ("Here are some relevant reasoning experience …" + ``{experience}``
  slot). Rendered *only* when an actual experience block is present
  (HEK enabled and at least one E2/E3 hit). When HEK is off or the
  retrieval returned nothing the section is dropped entirely — no
  ``"None."`` placeholder reaches the model.
* :data:`HIEXP_TOOL_CALL_FORMAT_SPEC` — paper Table 7 protocol body
  (``<think>`` / ``<tool_call>`` / ``\\boxed{}`` contract + the JSON
  format example). Rendered only when
  ``HiExpConfig.include_tool_call_template`` is True; with
  Responses-API + ``bind_tools`` the chat template already injects the
  function schemas so the literal example is redundant on most modern
  models. The whole protocol moved here together because turning the
  format example off without the protocol would leave the model
  guessing how to mark its final answer.

For backward compatibility :data:`HIEXP_SYSTEM_PROMPT` is the full
concatenation of the three parts (with the experience section's slot
unfilled, so ``"{experience}" in HIEXP_SYSTEM_PROMPT`` still holds).
"""

from __future__ import annotations

from typing import Final

# Body — always rendered.
HIEXP_SYSTEM_PROMPT_BODY: Final[str] = """\
In this environment you have access to a set of tools you can use to assist with the user query. You may perform multiple rounds of function calls. In each round, you can call one or more functions.
"""

# Experience injection — rendered only when HEK retrieved hits.
HIEXP_EXPERIENCE_SECTION: Final[str] = """\

Here are some relevant reasoning experience and examples to guide your decision-making process:
{experience}
"""

# Optional protocol + format spec (paper Table 7). Doubled braces are
# collapsed by ``.format()`` at render time, so ``\\boxed{{}}`` in the
# source code becomes ``\boxed{}`` in the rendered string.
# FIXME: This may limit the model's output to "\\boxed{{...}}"
HIEXP_TOOL_CALL_FORMAT_SPEC: Final[str] = """\
In your response, you need to first think about the reasoning process in the mind and then conduct function calling to get the information or perform the actions if needed. The reasoning process and function calling are enclosed within <think> </think> and <tool_call> </tool_call> tags. The results of the function calls will be given back to you after execution, and you can continue to call functions until you get the final answer for the user's question. Finally, if you have got the answer, enclose it within \\boxed{{}} with latex format and do not continue to call functions, i.e., <think> Based on the response from the function call, I get the weather information. </think> The weather in Beijing on 2025-04-01 is \\boxed{{20C}}.

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>
"""

# Backward-compat: full concatenation = body + experience slot + format spec.
HIEXP_SYSTEM_PROMPT: Final[str] = (
    HIEXP_SYSTEM_PROMPT_BODY
    + HIEXP_EXPERIENCE_SECTION
    + "\n"
    + HIEXP_TOOL_CALL_FORMAT_SPEC
)


def _has_experience(block: str | None) -> bool:
    """Whether to render the experience section.

    Drop the section when:
      * the block is None / empty / whitespace-only;
      * the block is the literal sentinel ``"None."`` that
        ``render_e2_block([])`` returns when HEK retrieval came back
        empty (and that ``graph.py`` falls back to when HEK is off).
    """
    if not block:
        return False
    stripped = block.strip()
    return bool(stripped) and stripped != "None."


def render_system_prompt(
    experience_block: str,
    include_tool_call_template: bool = True,
) -> str:
    """Compose the system prompt.

    Args:
        experience_block: rendered E2/E3 hits, or ``"None."`` / empty
            when no experience is available. Empty / ``"None."`` causes
            the entire experience section to be dropped (the
            ``"Here are some relevant reasoning experience …"`` line is
            *not* shown to the model with no content beneath it).
        include_tool_call_template: when True (default) appends the
            paper Table 7 protocol + format example. Drop on servers
            where ``bind_tools`` already documents the schema and the
            literal protocol confuses generation.
    """
    parts: list[str] = [HIEXP_SYSTEM_PROMPT_BODY]
    if _has_experience(experience_block):
        parts.append(HIEXP_EXPERIENCE_SECTION.format(experience=experience_block))
    if include_tool_call_template:
        # ``.format()`` with no args collapses ``{{`` → ``{`` so the
        # literal-brace examples render correctly.
        parts.append("\n" + HIEXP_TOOL_CALL_FORMAT_SPEC.format())
    return "".join(parts)
