"""Evaluation harness — strictly separate from the agent.

The agent (``hiexp.agent.HiExpAgent``) only produces a trajectory plus a
raw final response. Everything below is *downstream*:

- ``answer_extract``: pull a final answer out of the raw response.
  Two extractors:
    * ``extract_boxed_answer``: HiExp paper Table 7 ``\\boxed{...}`` style,
      used for HotpotQA / 2Wiki / Musique / Bamboogle reproduction.
    * ``extract_exact_answer``: BrowseComp-Plus ``Exact Answer:`` style
      (``third_party/BrowseComp-Plus/search_agent/prompts.py:5-9``).
- ``metrics``: 1:1 port of Search-o1 ``evaluate.py`` — ``normalize_answer_qa``
  + EM + CEM (substring acc) + token F1.
- ``persist``: dump runs in BrowseComp-Plus ``run_*.json`` shape so we can
  reuse ``third_party/BrowseComp-Plus/scripts_evaluation/evaluate_run.py``.
- ``judge`` (M6): optional LLM-as-judge using BrowseComp-Plus
  ``GRADER_TEMPLATE``.
- ``runner`` (M6): batch evaluation pipeline.
"""

from hiexp.evaluation.answer_extract import (
    extract_boxed_answer,
    extract_exact_answer,
)
from hiexp.evaluation.benchmarks import (
    BenchmarkSpec,
    JudgeScorer,
    Scorer,
    WordLevelScorer,
    get_spec,
    list_benchmarks,
    make_scorer,
    register,
)
from hiexp.evaluation.judge import (
    GRADER_TEMPLATE,
    calibration_error_l2,
    compute_citation_metrics,
    extract_citations_from_response,
    parse_judge_response,
    render_grader_prompt,
)
from hiexp.evaluation.metrics import (
    QaMetric,
    aggregate,
    normalize_answer_qa,
    score_one,
)

__all__ = [
    "extract_boxed_answer",
    "extract_exact_answer",
    "QaMetric",
    "score_one",
    "aggregate",
    "normalize_answer_qa",
    "BenchmarkSpec",
    "Scorer",
    "WordLevelScorer",
    "JudgeScorer",
    "get_spec",
    "list_benchmarks",
    "register",
    "make_scorer",
    "GRADER_TEMPLATE",
    "render_grader_prompt",
    "parse_judge_response",
    "extract_citations_from_response",
    "compute_citation_metrics",
    "calibration_error_l2",
]
