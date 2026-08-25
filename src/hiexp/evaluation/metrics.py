"""QA metrics — token-level F1, EM, CEM (Cover EM).

Direct 1:1 port of the QA branch of
``related/Search-o1/scripts/evaluate.py:46-117`` so our prompt-based
HotpotQA / 2Wiki / Musique / Bamboogle numbers are bit-identical to
the Search-o1 reference. The only relabel is ``acc → cem``: that
``acc`` field in Search-o1 is the *Cover Exact Match* metric the HiExp
paper §4.1 reports as "CEM".
"""

from __future__ import annotations

import re
import string
from collections import Counter
from dataclasses import dataclass


def _remove_articles(text: str) -> str:
    return re.sub(r"\b(a|an|the)\b", " ", text)


def _white_space_fix(text: str) -> str:
    return " ".join(text.strip().split())


def _remove_punc(text: str) -> str:
    exclude = set(string.punctuation)
    return "".join(ch for ch in text if ch not in exclude)


def normalize_answer_qa(s: str) -> str:
    """Search-o1 ``normalize_answer_qa`` (``evaluate.py:51-61``)."""
    if not isinstance(s, str):
        s = "" if s is None else str(s)
    return _white_space_fix(_remove_articles(_remove_punc(s.lower())))


@dataclass(frozen=True)
class QaMetric:
    """Per-example metric values."""

    em: int
    cem: int  # Search-o1 ``acc``: ground-truth substring of normalized prediction
    f1: float


def _f1(pred_tokens: list[str], gold_tokens: list[str]) -> float:
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def score_one(prediction: str, gold_answers: list[str] | str) -> QaMetric:
    """Score a single prediction against one or more gold answers.

    Multiple golds are handled the same way as Search-o1
    (``evaluate.py:71-87``): take the max EM/CEM/F1 over all golds.
    """
    if isinstance(gold_answers, str):
        gold_answers = [gold_answers]

    pred_norm = normalize_answer_qa(prediction)
    pred_tokens = pred_norm.split()

    em, cem, f1 = 0, 0, 0.0
    for gold in gold_answers:
        gold_norm = normalize_answer_qa(gold)
        gold_tokens = gold_norm.split()
        em = max(em, int(pred_norm == gold_norm))
        cem = max(cem, int(gold_norm in pred_norm))
        f1 = max(f1, _f1(pred_tokens, gold_tokens))
    return QaMetric(em=em, cem=cem, f1=f1)


def aggregate(metrics: list[QaMetric]) -> dict[str, float]:
    """Return mean EM/CEM/F1 over a list of per-example metrics."""
    if not metrics:
        return {"em": 0.0, "cem": 0.0, "f1": 0.0, "n": 0}
    n = len(metrics)
    return {
        "em": sum(m.em for m in metrics) / n,
        "cem": sum(m.cem for m in metrics) / n,
        "f1": sum(m.f1 for m in metrics) / n,
        "n": n,
    }
