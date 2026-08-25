"""User-message templates for the question turn.

Two contracts are provided, picked by ``HiExpConfig.question_template``:

- ``hiexp_boxed`` — paper Table 7 contract; the model is instructed to
  emit the final answer inside ``\\boxed{...}``. Used for HiExp's
  multi-hop QA reproduction.
- ``browsecomp_plus`` — verbatim copy of
  ``third_party/BrowseComp-Plus/search_agent/prompts.py``
  ``QUERY_TEMPLATE_NO_GET_DOCUMENT``; the model emits ``Explanation /
  Exact Answer / Confidence`` blocks.
- ``raw`` — pass the question through unchanged; useful for ablation.
"""

from __future__ import annotations

# Paper Table 7 framing leaves the user turn minimal — the agent's
# behavioural contract (``\boxed{}``, think/tool_call tags) lives in the
# system message. We just give the model the question.
HIEXP_BOXED_TEMPLATE = (
    "Please answer the following question by interacting with the search tool "
    "as needed. You should think step by step and provide your final answer in "
    "the format \\boxed{{YOUR_ANSWER}}.\n\n"
    "Question:\n{question}\n"
)

# 1:1 copy of third_party/BrowseComp-Plus/search_agent/prompts.py
# ``QUERY_TEMPLATE_NO_GET_DOCUMENT`` (lines 12-21).
BROWSECOMP_PLUS_TEMPLATE = (
    "You are a deep research agent. You need to answer the given question by "
    "interacting with a search engine, using the search tool provided. Please "
    "perform reasoning and use the tool step by step, in an interleaved "
    "manner. You may use the search tool multiple times.\n\n"
    "Question: {question}\n\n"
    "Your response should be in the following format:\n"
    "Explanation: {{your explanation for your final answer. For this "
    "explanation section only, you should cite your evidence documents inline "
    "by enclosing their docids in square brackets [] at the end of sentences. "
    "For example, [20].}}\n"
    "Exact Answer: {{your succinct, final answer}}\n"
    "Confidence: {{your confidence score between 0% and 100% for your answer}}"
)


def render_question(question: str, template: str) -> str:
    """Render ``question`` according to ``template`` (one of the literal names).

    The unknown / ``raw`` case returns the question verbatim so callers
    can wrap it however they like.
    """
    if template == "hiexp_boxed":
        return HIEXP_BOXED_TEMPLATE.format(question=question)
    if template == "browsecomp_plus":
        return BROWSECOMP_PLUS_TEMPLATE.format(question=question)
    if template == "raw":
        return question
    raise ValueError(f"unknown question_template: {template!r}")
