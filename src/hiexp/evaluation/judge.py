"""LLM-as-judge primitives — single source of truth.

Vendors BrowseComp-Plus's ``GRADER_TEMPLATE`` (verbatim) plus the
parsing / citation / calibration helpers from
``third_party/BrowseComp-Plus/scripts_evaluation/evaluate_with_openai.py``.

These primitives are deliberately stateless and async-friendly so they
can be reused by:

* ``hiexp.evaluation.benchmarks.JudgeScorer`` — used by *both*
  ``run_eval.py`` (final scoring) and ``sample_rollouts.py`` (pos/neg
  bucketing during HEK construction).
* ``scripts/hiexp_prompt/eval/run_judge.py`` — standalone post-hoc
  judge over BCP-shaped run JSONs.

No I/O. No CLI. The orchestration belongs to the caller.
"""

from __future__ import annotations

import re

# ── BCP grader template ──────────────────────────────────────────────────


GRADER_TEMPLATE = """
Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.


confidence: The extracted confidence score between 0|\\%| and 100|\\%| from [response]. Put 100 if there is no confidence score available.
""".strip()


def render_grader_prompt(*, question: str, response: str, correct_answer: str) -> str:
    return GRADER_TEMPLATE.format(
        question=question, response=response, correct_answer=correct_answer
    )


# ── Judge response parsing ───────────────────────────────────────────────


def parse_judge_response(text: str) -> dict:
    """Pull ``extracted_final_answer / reasoning / correct / confidence`` out of
    the judge's free-form output.

    Mirrors BCP's ``parse_judge_response`` line-for-line: bold-form first,
    then unmarked. Returns ``parse_error=True`` when ``correct`` is missing.
    """
    result: dict = {
        "extracted_final_answer": None,
        "reasoning": None,
        "correct": None,
        "confidence": None,
        "parse_error": False,
    }
    if not text:
        result["parse_error"] = True
        return result

    answer_patterns = [
        r"\*\*extracted_final_answer:\*\*\s*(.*?)(?=\n|$)",
        r"\*\*extracted_final_answer\*\*:\s*(.*?)(?=\n|$)",
        r"extracted_final_answer:\s*(.*?)(?=\n|$)",
    ]
    for pat in answer_patterns:
        m = re.search(pat, text, re.IGNORECASE | re.DOTALL)
        if m:
            result["extracted_final_answer"] = m.group(1).strip()
            break

    reasoning_patterns = [
        r"\*\*reasoning:\*\*\s*(.*?)(?=\n\*\*correct:\*\*|\n\*\*correct\*\*:|\ncorrect:|$)",
        r"\*\*reasoning\*\*:\s*(.*?)(?=\n\*\*correct:\*\*|\n\*\*correct\*\*:|\ncorrect:|$)",
        r"reasoning:\s*(.*?)(?=\ncorrect:|$)",
    ]
    for pat in reasoning_patterns:
        m = re.search(pat, text, re.IGNORECASE | re.DOTALL)
        if m:
            result["reasoning"] = m.group(1).strip()
            break

    correct_patterns = [
        r"\*\*correct:\*\*\s*(yes|no)",
        r"\*\*correct\*\*:\s*(yes|no)",
        r"correct:\s*(yes|no)",
    ]
    for pat in correct_patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            result["correct"] = m.group(1).lower() == "yes"
            break

    confidence_patterns = [
        r"\*\*confidence:\*\*\s*(\d+(?:\.\d+)?)\s*%?",
        r"\*\*confidence\*\*:\s*(\d+(?:\.\d+)?)\s*%?",
        r"confidence:\s*(\d+(?:\.\d+)?)\s*%?",
    ]
    for pat in confidence_patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            conf = float(m.group(1))
            result["confidence"] = min(conf, 100.0)
            break

    if result["correct"] is None:
        result["parse_error"] = True
    return result


# ── Citation / retrieval helpers (BCP shape) ─────────────────────────────


def extract_citations_from_response(response_text: str) -> list[str]:
    """Mirrors BCP: ``[docid]`` and ``【docid】`` (oss finetune format)."""
    if not response_text:
        return []

    single_matches = re.findall(r"\[(\d+)\]", response_text)
    multi_matches = re.findall(r"\[([^\[\]]*?)\]", response_text)
    single_fw = re.findall(r"【(\d+)】", response_text)
    multi_fw = re.findall(r"【([^【】]*?)】", response_text)

    docids: set[str] = set()
    docids.update(single_matches)
    docids.update(single_fw)
    for match in multi_matches:
        if match in single_matches:
            continue
        docids.update(re.findall(r"\d+", match))
    for match in multi_fw:
        if match in single_fw:
            continue
        docids.update(re.findall(r"\d+", match))
    return list(docids)


def compute_citation_metrics(
    cited_docids: list[str], relevant_docids: list[str]
) -> dict:
    metrics = {
        "num_citations": len(cited_docids),
        "num_relevant": len(relevant_docids),
        "precision": 0.0,
        "recall": 0.0,
    }
    if not cited_docids:
        return metrics
    cited_set = set(cited_docids)
    relevant_set = set(relevant_docids)
    inter = cited_set & relevant_set
    metrics["precision"] = len(inter) / len(cited_docids)
    if relevant_docids:
        metrics["recall"] = len(inter) / len(relevant_docids)
    return metrics


def calibration_error_l2(
    confidences: list[float], correctness: list[bool], beta: int = 100
) -> float:
    """ECE-L2 in percent, matching BCP's ``calculate_calibration_error``.

    BCP short-circuits when fewer than ``beta`` samples are available; we
    do the same — returning ``0.0`` so callers can include it in summaries
    unconditionally.
    """
    import numpy as np

    if len(confidences) < beta:
        return 0.0
    confidence = np.array(confidences) / 100.0
    correct = np.array(correctness, dtype=float)

    idxs = np.argsort(confidence)
    confidence = confidence[idxs]
    correct = correct[idxs]
    bins = [[i * beta, (i + 1) * beta] for i in range(len(confidence) // beta)]
    bins[-1] = [bins[-1][0], len(confidence)]

    cerr = 0.0
    total = len(confidence)
    for i in range(len(bins) - 1):
        lo, hi = bins[i]
        bc = confidence[lo:hi]
        br = correct[lo:hi]
        if len(bc) == 0:
            continue
        diff = abs(float(np.nanmean(bc)) - float(np.nanmean(br)))
        cerr += len(bc) / total * (diff ** 2)
    return float(np.sqrt(cerr) * 100.0)


__all__ = [
    "GRADER_TEMPLATE",
    "render_grader_prompt",
    "parse_judge_response",
    "extract_citations_from_response",
    "compute_citation_metrics",
    "calibration_error_l2",
]
