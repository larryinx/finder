"""QaEmF1Reward — no-judge reward for HotpotQA-style multi-hop QA.

Scores the agent's extracted answer against the FULL list of
``golden_answers`` using the Search-o1 / HiExp QA metrics (token-level
F1, Cover-EM, and EM — all max over golds). No LLM judge, no
analyzer; purely deterministic.

Correctness flag: ``cem == 1`` (the gold answer is a substring of the
normalized prediction). This is the more forgiving of the two binary
metrics and is what the HiExp paper headlines as "CEM". The exact
EM and F1 numbers are still recorded per-trajectory in
``RewardReport.extra`` so the Evaluator can aggregate them into
``metrics.json``.

Designed as a drop-in for ``search_agent_lasj``: same ``RewardReport``
shape (``judge`` filled with a synthetic JudgeResult so reflectors
that read ``judge.correct`` / ``judge.extracted_answer`` keep working),
same efficiency-stats path.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from search_ttl.reward.manager import RewardManager
from search_ttl.reward.schemas import (
    EfficiencyStats,
    FailureMode,
    JudgeResult,
    RewardReport,
)
from search_ttl.rollout.schemas import Trajectory
from search_ttl.trainer.registry import register_reward
from search_ttl.utils.concurrency import bounded_gather
from search_ttl.utils.tracing import traceable

LOG = logging.getLogger(__name__)


@dataclass
class _Cfg:
    concurrency: int = 16
    # Which of {em, cem} drives ``RewardReport.correctness`` (and thus
    # best.md tracking). Default ``cem`` matches the HiExp paper's
    # headline number for HotpotQA / 2Wiki / Musique / Bamboogle.
    correctness_metric: str = "cem"


@register_reward("qa_em_f1")
class QaEmF1Reward(RewardManager):
    def __init__(self, cfg: _Cfg) -> None:
        self.cfg = cfg
        if cfg.correctness_metric not in ("em", "cem"):
            raise ValueError(
                f"reward.correctness_metric must be 'em' or 'cem', "
                f"got {cfg.correctness_metric!r}"
            )

    @classmethod
    def from_omegaconf(cls, cfg) -> "QaEmF1Reward":
        return cls(
            _Cfg(
                concurrency=int(cfg.get("concurrency", 16)),
                correctness_metric=str(cfg.get("correctness_metric", "cem")),
            )
        )

    @traceable(name="Reward.score", run_type="chain")
    async def score(self, trajectories: list[Trajectory]) -> list[RewardReport]:
        factories = [(lambda t=t: self._score_one(t)) for t in trajectories]
        return await bounded_gather(factories, concurrency=self.cfg.concurrency)

    async def _score_one(self, traj: Trajectory) -> RewardReport:
        # Lazy import: keeps the registry import cheap and dodges any
        # circular-import risk through hiexp.evaluation.__init__.
        from hiexp.evaluation.metrics import score_one  # type: ignore[import-not-found]

        prediction = (traj.extracted_answer or traj.raw_response or "").strip()
        golds = list(traj.golden_answers) or ([traj.target] if traj.target else [])
        metric = score_one(prediction, golds)

        is_correct = bool(getattr(metric, self.cfg.correctness_metric))
        eff = self._compute_efficiency(traj)
        fm: FailureMode = (
            "ok"
            if is_correct
            else "max_search"
            if traj.status == "max_search"
            else "max_turn"
            if traj.status == "max_turn"
            else "error"
            if traj.status == "error"
            else "parse_error"
            if not prediction
            else "bad_query"
        )

        # Synthesize a JudgeResult so reflectors that read
        # ``RewardReport.judge`` (the querychain reflectors) still work.
        synth_judge = JudgeResult(
            correct=is_correct,
            extracted_answer=prediction or None,
            reasoning=(
                f"qa_em_f1: em={metric.em} cem={metric.cem} "
                f"f1={metric.f1:.3f} (correctness={self.cfg.correctness_metric})"
            ),
            confidence=100.0 if is_correct else 0.0,
            parse_error=not prediction,
            raw_response="",
        )

        text_for_reflector = (
            f"qid={traj.qid} r_idx={traj.r_idx} "
            f"verdict={'correct' if is_correct else 'incorrect'} "
            f"em={metric.em} cem={metric.cem} f1={metric.f1:.3f} "
            f"n_search={eff.n_search} n_turn={eff.n_turn}\n"
            f"prediction={prediction!r} gold={golds!r}"
        )

        return RewardReport(
            qid=traj.qid,
            r_idx=traj.r_idx,
            correctness=is_correct,
            judge=synth_judge,
            per_query_analysis=[],
            efficiency=eff,
            failure_mode=fm,
            text_for_reflector=text_for_reflector,
            extra={
                "em": int(metric.em),
                "cem": int(metric.cem),
                "f1": float(metric.f1),
                "summary": text_for_reflector,
            },
        )

    def _compute_efficiency(self, traj: Trajectory) -> EfficiencyStats:
        n_search = sum(traj.tool_call_counts.values())
        n_turn = int(traj.metrics.get("turns", 0))
        normed = [_norm_query(q.query) for q in traj.queries]
        seen: set[str] = set()
        red = 0
        for q in normed:
            if q in seen:
                red += 1
            else:
                seen.add(q)
        total_returned = sum(q.n_results for q in traj.queries) or 1
        novel_doc_ratio = len(traj.retrieved_docids) / total_returned
        return EfficiencyStats(
            n_search=n_search,
            n_turn=n_turn,
            hit_max_search=traj.status == "max_search",
            hit_max_turn=traj.status == "max_turn",
            redundant_query_count=red,
            novel_doc_ratio=round(novel_doc_ratio, 3),
        )


def _norm_query(q: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", q.lower())).strip()
