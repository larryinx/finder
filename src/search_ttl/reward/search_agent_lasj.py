"""SearchAgentRewardLASJ — judge-only (LLM-as-judge) reward.

One LLM call per trajectory: the judge (correctness + extracted answer
+ reasoning). Everything else is rule-based:

- efficiency stats (search / turn counts, redundant queries, novel-doc
  ratio) computed from the trajectory with no LLM, and
- ``text_for_reflector`` rendered deterministically from the judge
  result + efficiency (no analyzer / summary LLM call).

The ``querychain_*`` reflectors consume ``RewardReport.judge`` directly
(they state the judged verdict to the agent itself) and contrast an
incorrect trajectory against the shortest correct peer, so no prose
summary is needed.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field

from search_ttl.reward.judge import Judge, JudgeConfig
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
class _RewardCfg:
    judge: JudgeConfig = field(
        default_factory=lambda: JudgeConfig(model="", base_url="", api_key="")
    )
    eval_lite: bool = False
    concurrency: int = 4
    max_search_limit_for_traj: int = 10
    # How many chars of the judge's reasoning to keep in
    # ``text_for_reflector``. The querychain reflectors read
    # ``RewardReport.judge`` directly, so this string is mostly for the
    # rewards.jsonl audit trail.
    judge_reasoning_max_chars: int = 400


@register_reward("search_agent_lasj")
class SearchAgentRewardLASJ(RewardManager):
    def __init__(self, cfg: _RewardCfg) -> None:
        self.cfg = cfg
        self._judge = Judge(cfg.judge)

    @classmethod
    def from_omegaconf(cls, cfg) -> "SearchAgentRewardLASJ":
        judge_node = cfg.get("judge", {}) or {}
        rcfg = _RewardCfg(
            judge=JudgeConfig(
                model=str(judge_node.get("model", ""))
                or os.environ.get("EVAL_JUDGE_MODEL", "")
                or os.environ.get("AGENT_LLM_MODEL", ""),
                base_url=str(judge_node.get("base_url", ""))
                or os.environ.get("EVAL_JUDGE_BASE_URL", "")
                or os.environ.get("AGENT_LLM_BASE_URL", ""),
                api_key=str(judge_node.get("api_key", ""))
                or os.environ.get("EVAL_JUDGE_API_KEY", "")
                or os.environ.get("AGENT_LLM_API_KEY", ""),
                template=str(judge_node.get("template", "bcp_grader")),
                max_response_tokens=int(judge_node.get("max_response_tokens", 1024)),
                temperature=float(judge_node.get("temperature", 0.0)),
                top_p=float(judge_node.get("top_p", 1.0)),
                enable_thinking=bool(judge_node.get("enable_thinking", False)),
                llm_provider=str(judge_node.get("llm_provider", "openai_compat")),
            ),
            eval_lite=bool(cfg.get("eval_lite", False)),
            concurrency=int(cfg.get("concurrency", 4)),
            max_search_limit_for_traj=int(cfg.get("max_search_limit_for_traj", 10)),
            judge_reasoning_max_chars=int(cfg.get("judge_reasoning_max_chars", 400)),
        )
        return cls(rcfg)

    # ── public ────────────────────────────────────────────────────────────
    @traceable(name="Reward.score", run_type="chain")
    async def score(self, trajectories: list[Trajectory]) -> list[RewardReport]:
        factories = [(lambda t=t: self._score_one(t)) for t in trajectories]
        return await bounded_gather(factories, concurrency=self.cfg.concurrency)

    # ── per-traj scoring ──────────────────────────────────────────────────
    @traceable(name="Reward.score_one", run_type="chain")
    async def _score_one(self, traj: Trajectory) -> RewardReport:
        judge = await self._judge.ajudge(
            question=traj.question,
            prediction=traj.extracted_answer or traj.raw_response or "",
            target=traj.target,
        )
        eff = self._compute_efficiency(traj)
        fm: FailureMode = (
            "ok"
            if judge.correct
            else "max_search"
            if traj.status == "max_search"
            else "max_turn"
            if traj.status == "max_turn"
            else "error"
            if traj.status == "error"
            else "parse_error"
            if judge.parse_error
            else "bad_query"
        )

        text_for_reflector = _render_reward_text(
            traj, judge, eff, fm, self.cfg.judge_reasoning_max_chars
        )
        return RewardReport(
            qid=traj.qid,
            r_idx=traj.r_idx,
            correctness=judge.correct,
            judge=judge,
            per_query_analysis=[],  # judge-only: no structured per-query analysis
            efficiency=eff,
            failure_mode=fm,
            text_for_reflector=text_for_reflector,
            extra={"summary": text_for_reflector},
        )

    # ── rule-based efficiency (cheap, no LLM) ────────────────────────────
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


def _render_reward_text(
    traj: Trajectory,
    judge: JudgeResult,
    eff: EfficiencyStats,
    fm: FailureMode,
    reasoning_max_chars: int,
) -> str:
    """Deterministic, LLM-free reward text.

    One head line (verdict + efficiency) plus the judge's own
    extracted-answer / reasoning. No analyzer paragraph.
    """
    head = (
        f"qid={traj.qid} r_idx={traj.r_idx} "
        f"verdict={'correct' if judge.correct else 'incorrect'} "
        f"failure_mode={fm} "
        f"n_search={eff.n_search} n_turn={eff.n_turn} "
        f"redundant={eff.redundant_query_count} "
        f"novel_doc_ratio={eff.novel_doc_ratio}"
    )
    extracted = (judge.extracted_answer or "").strip()
    reason = (judge.reasoning or "").strip()
    if reasoning_max_chars and len(reason) > reasoning_max_chars:
        reason = reason[:reasoning_max_chars].rstrip() + " …"
    judge_line = (
        f"judge: extracted={extracted!r} parse_error={judge.parse_error}"
    )
    if reason:
        judge_line += f" | reasoning: {reason}"
    return f"{head}\n{judge_line}"
