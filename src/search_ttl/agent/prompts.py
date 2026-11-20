"""SearchAgent prompts — matches the HiExp baseline that scores 27%.

Verified against ``config/hiexp_prompt/base.yaml`` (the file the baseline
``run_bcp_baseline_qwen3_8b.sh`` uses):

    enable_thinking:            true
    preserve_thinking:          true
    include_tool_call_template: false   ← format spec OMITTED
    question_template:          browsecomp_plus  (set in benchmark/bcp.yaml)

So the baseline's system prompt is a SINGLE LINE
(``HIEXP_SYSTEM_PROMPT_BODY``); the BCP template lives in the first user
message. Earlier we added a long format-spec block here and a ``commit
now`` nudge — both made things worse because the model already commits
fine when given the minimal prompt + thinking-on (re-verified live with
``scripts/testing/ttl/diagnose_zero_accuracy.py``).

We add ONE thing on top of baseline: an ``<experience>`` slot. That is
TTL-specific and must not change baseline behaviour when the slot is
empty / sentinel — when ``experience`` is unset we render the baseline
system prompt byte-for-byte.
"""

from __future__ import annotations

EMPTY_EXPERIENCE_SENTINEL = "(none yet)"

# Verbatim from src/hiexp/prompts/system_prompt.py:HIEXP_SYSTEM_PROMPT_BODY.
_BODY = (
    "In this environment you have access to a set of tools you can use to "
    "assist with the user query. You may perform multiple rounds of function "
    "calls. In each round, you can call one or more functions.\n"
)

# Experience slot, rendered ONLY when the slot has real content. We
# borrow the wording from src/hiexp/prompts/system_prompt.py:HIEXP_EXPERIENCE_SECTION.
_EXPERIENCE_BLOCK = (
    "\nHere are some relevant reasoning experience and examples to guide your "
    "decision-making process:\n"
    "<experience>\n{experience}\n</experience>\n"
)

# Verbatim BCP QUERY_TEMPLATE_NO_GET_DOCUMENT (same one HiExp uses via
# question_template=browsecomp_plus).
USER_PROMPT_TEMPLATE = (
    "You are a deep research agent. You need to answer the given question by "
    "interacting with a search engine, using the search tool provided. "
    "Please perform reasoning and use the tool step by step, in an interleaved manner. "
    "You may use the search tool multiple times.\n\n"
    "Question: {question}\n\n"
    "Your response should be in the following format:\n"
    "Explanation: {{your explanation for your final answer. For this "
    "explanation section only, you should cite your evidence documents inline "
    "by enclosing their docids in square brackets [] at the end of sentences. "
    "For example, [20].}}\n"
    "Exact Answer: {{your succinct, final answer}}\n"
    "Confidence: {{your confidence score between 0% and 100% for your answer}}"
)

# Official BrowseComp QUERY_TEMPLATE — no docid citation, no
# "deep research agent" preamble. Source:
# https://openai.com/index/browsecomp/ via the simple-evals release.
# We keep the same Explanation/Exact Answer/Confidence triple so the
# shared bcp_grader judge template parses the response the same way.
BROWSECOMP_USER_PROMPT_TEMPLATE = (
    "You answer the given question by interacting with a web search "
    "tool. Reason and call the search tool step by step, in an "
    "interleaved manner. You may use the search tool multiple times.\n\n"
    "Question: {question}\n\n"
    "Your response should be in the following format:\n"
    "Explanation: {{your explanation for your final answer}}\n"
    "Exact Answer: {{your succinct, final answer}}\n"
    "Confidence: {{your confidence score between 0% and 100% for your answer}}"
)

# HiExp paper Table 7 — the model emits the final answer inside
# ``\boxed{...}``. Used for Search-o1 / HiExp-style multi-hop QA
# (HotpotQA / 2Wiki / Musique / Bamboogle) where evaluation is
# F1/CEM/EM against a list of golden answers (qa_em_f1 reward).
BOXED_ANSWER_USER_PROMPT_TEMPLATE = (
    "Please answer the following question by interacting with the search tool "
    "as needed. You should think step by step and provide your final answer in "
    "the format \\boxed{{YOUR_ANSWER}}.\n\n"
    "Question:\n{question}\n"
)

_USER_PROMPT_TEMPLATES: dict[str, str] = {
    "bcp": USER_PROMPT_TEMPLATE,
    "browsecomp": BROWSECOMP_USER_PROMPT_TEMPLATE,
    "boxed_answer": BOXED_ANSWER_USER_PROMPT_TEMPLATE,
}


def render_system_prompt(experience: str | None) -> str:
    text = (experience or "").strip()
    if not text or text == EMPTY_EXPERIENCE_SENTINEL:
        # Match baseline byte-for-byte when nothing has been learned yet.
        return _BODY
    return _BODY + _EXPERIENCE_BLOCK.format(experience=text)


def render_user_prompt(question: str, *, template: str = "bcp") -> str:
    """Render the first-user-message body.

    ``template`` selects between the BCP variant (asks for docid
    citations) and the BrowseComp variant (no docids). Unknown
    template names fall back to ``bcp`` to preserve old behaviour.
    """
    tmpl = _USER_PROMPT_TEMPLATES.get(template, USER_PROMPT_TEMPLATE)
    return tmpl.format(question=question)
