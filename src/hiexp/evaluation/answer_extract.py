"""Answer extractors. Used by the eval harness, not by the agent loop."""

from __future__ import annotations

import re

# ``\\boxed{...}`` matcher.
#
# Search-o1's ``evaluate.py:32`` uses ``\\boxed\{(.*)\}`` which is greedy
# and silently merges multiple ``\\boxed{}`` markers into one match. We
# walk the string with a brace counter instead so we can both (a) handle
# nested LaTeX like ``\\boxed{\\text{Yes}}`` or ``\\boxed{\\frac{a}{b}}``
# and (b) still pick the *last* well-formed box when the model emits
# several.
_TEXT_WRAPPER = re.compile(r"\\text\{(.*)\}\s*\Z", re.DOTALL)

# BrowseComp-Plus ``Exact Answer:`` pattern (``prompts.py:9``).
#
# Captures everything from ``Exact Answer:`` up to a newline followed by
# ``Confidence:`` / end-of-string. We allow surrounding ``**`` markdown
# bolding because Qwen3 occasionally emits it.
_EXACT_ANSWER_PATTERN = re.compile(
    r"\*{0,2}Exact Answer\*{0,2}\s*:\s*(.*?)(?=\n\s*\*{0,2}Confidence|\Z)",
    re.IGNORECASE | re.DOTALL,
)


def extract_boxed_answer(text: str) -> str:
    """Return the content of the last ``\\boxed{...}`` in ``text``.

    Returns an empty string if no boxed answer is present. A single
    enclosing ``\\text{...}`` wrapper (common from reasoning models that
    treat ``\\boxed{}`` as math and add ``\\text`` for prose answers) is
    peeled.

    HiExp paper Table 7 specifies ``\\boxed{}`` as the final-answer form
    for the multi-hop QA reproduction.
    """
    needle = r"\boxed{"
    last = ""
    i = 0
    while True:
        start = text.find(needle, i)
        if start == -1:
            break
        j = start + len(needle)
        depth = 1
        while j < len(text) and depth > 0:
            c = text[j]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
            j += 1
        if depth == 0:
            content = text[start + len(needle) : j - 1]
            m = _TEXT_WRAPPER.fullmatch(content.strip())
            if m:
                content = m.group(1)
            last = content
            i = j
        else:
            # Unbalanced — stop scanning rather than loop forever.
            break
    return last


def extract_exact_answer(text: str) -> str:
    """Return the content of the first ``Exact Answer:`` block in ``text``.

    Empty string if absent. Mirrors the BrowseComp-Plus query template
    contract (``third_party/BrowseComp-Plus/search_agent/prompts.py``).
    Tolerates surrounding ``**`` markdown emphasis on the value.
    """
    match = _EXACT_ANSWER_PATTERN.search(text)
    if not match:
        return ""
    value = match.group(1).strip()
    # Strip surrounding markdown emphasis (e.g. ``**Paris**``)
    if value.startswith("**") and value.endswith("**"):
        value = value[2:-2].strip()
    elif value.endswith("**"):
        value = value[:-2].strip()
    elif value.startswith("**"):
        value = value[2:].strip()
    return value.rstrip(".").strip()
