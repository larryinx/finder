# Adding a reward

A reward turns each `Trajectory` into a `RewardReport`: a correctness
verdict (drives `best.md` and the reflector's "correct / incorrect"
framing), rule-based efficiency stats, and a short text the reflector
can read. It never produces a scalar for gradient descent — the report is
consumed as prose and structure.

Existing implementations: `search_agent_lasj` (LLM-as-judge, BCP grader
template) and `qa_em_f1` (no LLM; EM / cover-EM / F1 against a list of
gold answers) in `src/search_ttl/reward/`.

## Interface

```python
# src/search_ttl/reward/manager.py
class RewardManager(ABC):
    @classmethod
    def from_omegaconf(cls, cfg) -> "RewardManager": ...
    async def score(self, trajectories: list[Trajectory]) -> list[RewardReport]:
        """Same length as input; one RewardReport per trajectory."""
```

`score()` is called by the trainer once per micro-batch chunk and by the
Evaluator on a whole split, so it should fan out its own LLM calls with
`bounded_gather(..., concurrency=self.cfg.concurrency)`.

`RewardReport` (`reward/schemas.py`):

| Field | Contract |
|---|---|
| `qid`, `r_idx` | copy from the trajectory |
| `correctness: bool` | the binary verdict used everywhere downstream |
| `judge: JudgeResult` | `correct`, `extracted_answer`, `reasoning`, `confidence`, `parse_error`, `raw_response`. Reflectors read `judge.correct` / `judge.extracted_answer` directly — fill it even when you have no LLM judge (`qa_em_f1` synthesises one) |
| `efficiency: EfficiencyStats` | `n_search`, `n_turn`, `hit_max_search`, `hit_max_turn`, `redundant_query_count`, `novel_doc_ratio` — rule-based, no LLM; copy `_compute_efficiency` from `search_agent_lasj.py` |
| `failure_mode` | one of `ok`, `bad_query`, `max_search`, `max_turn`, `parse_error`, `hallucination`, `off_topic`, `error` |
| `text_for_reflector` | short, deterministic summary line(s); this is what the reflector prompt shows |
| `per_query_analysis` | `[]` unless you produce per-query `QueryAnalysis` entries |
| `extra` | free dict; if you put `f1` / `cem` / `em` here the Evaluator averages them into `metrics.json` |

Useful building blocks: `search_ttl.reward.judge.Judge` /
`JudgeConfig` (async LLM judge with an exact-match fast path; renders the
shared BCP grader prompt), `search_ttl.utils.concurrency.bounded_gather`,
`@traceable` from `search_ttl.utils.tracing`.

## Skeleton

```python
# src/search_ttl/reward/my_reward.py
from dataclasses import dataclass, field
from search_ttl.reward.judge import Judge, JudgeConfig
from search_ttl.reward.manager import RewardManager
from search_ttl.reward.schemas import EfficiencyStats, JudgeResult, RewardReport
from search_ttl.rollout.schemas import Trajectory
from search_ttl.trainer.registry import register_reward
from search_ttl.utils.concurrency import bounded_gather
from search_ttl.utils.tracing import traceable


@dataclass
class _Cfg:
    judge: JudgeConfig
    concurrency: int = 4


@register_reward("my_reward")
class MyReward(RewardManager):
    def __init__(self, cfg: _Cfg) -> None:
        self.cfg = cfg
        self._judge = Judge(cfg.judge)

    @classmethod
    def from_omegaconf(cls, cfg) -> "MyReward":
        j = cfg.get("judge", {}) or {}
        return cls(_Cfg(
            judge=JudgeConfig(
                model=str(j.get("model", "")), base_url=str(j.get("base_url", "")),
                api_key=str(j.get("api_key", "")),
                llm_provider=str(j.get("llm_provider", "openai_compat")),
                enable_thinking=bool(j.get("enable_thinking", False)),
            ),
            concurrency=int(cfg.get("concurrency", 4)),
        ))

    @traceable(name="Reward.score", run_type="chain")
    async def score(self, trajectories: list[Trajectory]) -> list[RewardReport]:
        return await bounded_gather(
            [(lambda t=t: self._score_one(t)) for t in trajectories],
            concurrency=self.cfg.concurrency,
        )

    async def _score_one(self, traj: Trajectory) -> RewardReport:
        judge = await self._judge.ajudge(
            question=traj.question,
            prediction=traj.extracted_answer or traj.raw_response or "",
            target=traj.target,
        )
        eff = ...  # EfficiencyStats from traj.queries / traj.metrics / traj.status
        fm = "ok" if judge.correct else ("max_search" if traj.status == "max_search" else "bad_query")
        return RewardReport(
            qid=traj.qid, r_idx=traj.r_idx, correctness=judge.correct, judge=judge,
            per_query_analysis=[], efficiency=eff, failure_mode=fm,
            text_for_reflector=f"verdict={'correct' if judge.correct else 'incorrect'} …",
        )
```

## Register and configure

1. Add `import search_ttl.reward.my_reward  # noqa: F401` to
   `import_default_plugins()` in `src/search_ttl/trainer/registry.py`.
2. Create `config/ttl/reward/my_reward.yaml`:

```yaml
reward:
  kind: my_reward
  judge:
    model: ${oc.env:EVAL_JUDGE_MODEL,${oc.env:AGENT_LLM_MODEL,Qwen/Qwen3.6-35B-A3B}}
    base_url: ${oc.env:EVAL_JUDGE_BASE_URL,${oc.env:AGENT_LLM_BASE_URL,http://localhost:8000/v1}}
    api_key: ${oc.env:EVAL_JUDGE_API_KEY,${oc.env:AGENT_LLM_API_KEY,EMPTY}}
    llm_provider: ${oc.env:EVAL_JUDGE_PROVIDER,${oc.env:AGENT_LLM_PROVIDER,openai_compat}}
    template: bcp_grader
    max_response_tokens: 8192
    temperature: 0.0
    enable_thinking: true
  concurrency: 4
```

3. Use it: `python scripts/ttl/run.py --reward my_reward …`,
   `scripts/ttl/eval_experience.py --reward my_reward`, or re-score an
   existing run with `scripts/ttl/rejudge_rollouts.py <eval_dir> --reward my_reward`.

## Tests to add or update

- `tests/search_ttl/test_registry.py` — assert `"my_reward" in REWARDS`.
- `tests/search_ttl/test_llm_builder.py::test_reward_judge_*` — `from_omegaconf`
  reads `judge.llm_provider` (copy the `search_agent_lasj` cases).
- A rule-based test in the spirit of `tests/search_ttl/test_reward_decision.py`
  for the efficiency / failure-mode logic (no LLM needed: construct
  `Trajectory` objects directly).
