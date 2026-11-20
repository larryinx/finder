"""DataProcessor for ACE on BrowseComp-Plus.

Reads ``data/BCP/`` (built by ``scripts/data_prep/hiexp/build_bcp_data.py``)
and provides the three methods ACE expects:

* :meth:`process_task_data` — pass-through (loaders below already produce
  the standardized ``{context, question, target, query_id}`` shape ACE
  needs);
* :meth:`answer_is_correct` — **LLM-as-judge** via BCP's ``GRADER_TEMPLATE``,
  not exact-match. Used by ACE's controller to compute the verdict that
  drives reflection / curation. Both with-GT and no-GT modes use this:
  ``no_ground_truth`` only governs whether the *Reflector* sees the gold,
  not how the orchestrator decides correctness;
* :meth:`evaluate_accuracy` — mean of ``answer_is_correct`` over a list.

The judge LLM falls back ``EVAL_JUDGE_*`` → ``AGENT_LLM_*`` env vars
(matching the rest of the codebase). It always runs with
``enable_thinking=False`` and low temperature; structured output, no CoT
needed.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, Dict, List

import weave

# Library module: rely on the caller (run.py / tests/conftest.py) for
# sys.path setup. Package depends on:
#   - hiexp.evaluation.judge   (in src/, always reachable)
#   - hiexp_prompt._llm_utils  (in scripts/, on path via run.py / conftest)
#   - hiexp_prompt._shared     (same)
from hiexp.evaluation.judge import (
    parse_judge_response,
    render_grader_prompt,
)
from hiexp_prompt._llm_utils import achat_once, make_chat_model
from hiexp_prompt._shared import load_browsecomp_plus_split

# The project root is 4 levels above this file:
#   src/ace_react/bcp/data_processor.py → src/ace_react/bcp → src/ace_react → src → <root>
_PROJECT_ROOT = Path(__file__).resolve().parents[3]


# ── BCP split locations ──────────────────────────────────────────────────


_BCP_ROOT = _PROJECT_ROOT / "data" / "BCP"
_DEFAULT_DECRYPTED = _BCP_ROOT / "data" / "browsecomp_plus_decrypted.jsonl"


def load_bcp_split(
    queries_tsv: str | Path,
    *,
    decrypted_jsonl: str | Path | None = None,
) -> List[Dict[str, Any]]:
    """Return ACE-shaped task dicts for a BCP split.

    Args:
        queries_tsv: Absolute or repo-root-relative path to a
            ``queries.tsv`` produced by
            ``scripts/data_prep/hiexp/build_bcp_data.py`` (e.g.
            ``data/BCP/topics-qrels-200/queries.tsv``).
        decrypted_jsonl: Path to ``browsecomp_plus_decrypted.jsonl``.
            Defaults to ``data/BCP/data/browsecomp_plus_decrypted.jsonl``
            under the project root.

    Each item: ``{query_id, question, context (""), target}``. ``context``
    is empty because BCP supplies no per-query context — the agent
    discovers everything via its search tool.
    """
    qtsv = Path(queries_tsv)
    if not qtsv.is_absolute():
        qtsv = _PROJECT_ROOT / qtsv
    djsonl = Path(decrypted_jsonl) if decrypted_jsonl is not None else _DEFAULT_DECRYPTED
    if not djsonl.is_absolute():
        djsonl = _PROJECT_ROOT / djsonl
    if not qtsv.exists() or not djsonl.exists():
        raise FileNotFoundError(
            f"BCP split files missing ({qtsv} or {djsonl}). "
            "Run `bash scripts/data_prep/prepare.sh hiexp` to materialize them."
        )
    examples = load_browsecomp_plus_split(
        queries_tsv=qtsv, decrypted_jsonl=djsonl
    )
    return [
        {
            "query_id": ex.query_id,
            "question": ex.question,
            "context": "",
            "target": (ex.golden_answers[0] if ex.golden_answers else ""),
        }
        for ex in examples
    ]


# ── ACE-compatible processor ─────────────────────────────────────────────


class DataProcessor:
    """ACE-compatible BCP processor.

    ``answer_is_correct`` calls a judge LLM with BCP's ``GRADER_TEMPLATE``
    and parses ``correct: yes|no`` out of the response.

    A cheap exact-match fast-path runs first to save the round trip on
    obvious matches (e.g. when both sides are identical short strings).
    """

    task_name = "browsecomp_plus"

    def __init__(
        self,
        *,
        judge_model: str | None = None,
        judge_base_url: str | None = None,
        judge_api_key: str | None = None,
        judge_max_tokens: int = 1024,
    ) -> None:
        self._llm = make_chat_model(
            model=judge_model
            or os.getenv("EVAL_JUDGE_MODEL")
            or os.getenv("AGENT_LLM_MODEL"),
            base_url=judge_base_url
            or os.getenv("EVAL_JUDGE_BASE_URL")
            or os.getenv("AGENT_LLM_BASE_URL"),
            api_key=judge_api_key
            or os.getenv("EVAL_JUDGE_API_KEY")
            or os.getenv("AGENT_LLM_API_KEY")
            or "EMPTY",
            temperature=0.0,
            top_p=1.0,
            max_tokens=judge_max_tokens,
            enable_thinking=False,
        )

    # ── ACE interface ────────────────────────────────────────────────────

    def process_task_data(self, raw_data: List[Dict]) -> List[Dict]:
        """No-op: ``load_bcp_split`` already returns the right shape."""
        return raw_data

    @weave.op(name="ace.DataProcessor.answer_is_correct")
    def answer_is_correct(self, predicted: str, ground_truth: str) -> bool:
        if not (predicted and ground_truth):
            return False
        # Fast-path: skip the LLM call for trivial matches.
        if predicted.strip().lower() == ground_truth.strip().lower():
            return True
        prompt = render_grader_prompt(
            question="(grading)",
            response=predicted,
            correct_answer=ground_truth,
        )
        try:
            text = asyncio.run(achat_once(self._llm, prompt))
        except RuntimeError:
            # Already inside an event loop. ACE's evaluator uses threads;
            # within a thread there's no loop, so this branch normally
            # doesn't trigger. Provide a fallback that uses a fresh loop.
            loop = asyncio.new_event_loop()
            try:
                text = loop.run_until_complete(achat_once(self._llm, prompt))
            finally:
                loop.close()
        return bool(parse_judge_response(text).get("correct"))

    def evaluate_accuracy(
        self,
        predictions: List[str],
        ground_truths: List[str],
    ) -> float:
        if len(predictions) != len(ground_truths):
            raise ValueError(
                f"predictions ({len(predictions)}) ≠ ground_truths "
                f"({len(ground_truths)})"
            )
        n = len(predictions)
        if n == 0:
            return 0.0
        correct = sum(
            1 for p, g in zip(predictions, ground_truths) if self.answer_is_correct(p, g)
        )
        return correct / n


__all__ = ["DataProcessor", "load_bcp_split"]
