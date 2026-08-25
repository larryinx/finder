"""RolloutManager — generate G trajectories per question with bounded concurrency.

Take a batch of prompts (here: ``QuestionRecord``), produce N×G trajectories.
Differences:
- We don't ship a `DataProto` with token tensors; the agent loop yields a
  Python ``Trajectory`` directly because reflection consumes prose.
- Persistence is a single ``rollouts.jsonl`` per step, not a
  folder of one-file-per-question — append-only, auditable, easy to scan.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from search_ttl.agent.search_agent import SearchAgent, SearchAgentConfig
from search_ttl.rollout.schemas import Trajectory  # noqa: F401  (used in TYPE_CHECKING via callable signature)
from search_ttl.rollout.schemas import QuestionRecord, RolloutBatchSlice, Trajectory
from search_ttl.utils.concurrency import bounded_gather
from search_ttl.utils.persistence import (
    append_jsonl,
    hash_text,
    trajectory_to_bcp_record,
)
from search_ttl.utils.tracing import traceable

LOG = logging.getLogger(__name__)


class RolloutManager:
    """Single-process, asyncio-bounded rollout pool."""

    def __init__(
        self,
        agent_config: SearchAgentConfig,
        *,
        concurrency: int = 8,
        seed: int = 42,
        agent_factory: Callable[[str], SearchAgent] | None = None,
    ) -> None:
        self.agent_config = agent_config
        self.concurrency = concurrency
        self.seed = seed
        # Allow tests to inject a mock factory; default builds a real one.
        self._agent_factory = agent_factory or (
            lambda exp_text: SearchAgent(self.agent_config, experience_text=exp_text)
        )

    @traceable(name="RolloutManager.generate", run_type="chain")
    async def generate(
        self,
        questions: list[QuestionRecord],
        experience_text: str,
        *,
        n: int = 4,
        save_jsonl: Path | None = None,
        step: int = 0,
        concurrency: int | None = None,
        on_traj_done: Callable[["Trajectory"], None] | None = None,
    ) -> list[RolloutBatchSlice]:
        """Returns one ``RolloutBatchSlice`` per question, each holding G trajectories.

        ``concurrency`` overrides ``self.concurrency`` for this call. Used by
        Evaluator to pass ``eval.concurrency``; training rollouts should
        omit it so they pick up ``rollout.concurrency`` from __init__.

        ``on_traj_done`` is called once per trajectory the moment it
        completes (in completion order, not submission order). Used by the
        trainer / evaluator to drive a rich progress bar. Exceptions inside
        the callback are swallowed so a misbehaving UI never breaks the run.
        """
        # All G rollouts share the same experience text.
        agent = self._agent_factory(experience_text)
        exp_hash = hash_text(experience_text or "")

        async def _one(q: QuestionRecord, r_idx: int) -> Trajectory:
            try:
                traj = await agent.arun(qid=q.qid, question=q.question, target=q.target)
            except Exception as exc:  # noqa: BLE001
                LOG.exception("rollout failed qid=%s r_idx=%s", q.qid, r_idx)
                traj = Trajectory(
                    qid=q.qid,
                    question=q.question,
                    target=q.target,
                    experience_snapshot=experience_text,
                    experience_hash=exp_hash,
                    status="error",
                    raw_response=f"[rollout_error] {type(exc).__name__}: {exc}",
                )
            # Always populate golden_answers (the agent sees only
            # ``target``; downstream rewards / qa scorers want the
            # full list of accepted answers).
            traj.golden_answers = list(q.golden_answers)
            traj.r_idx = r_idx
            traj.step = step
            traj.metadata_ttl = {
                "step": step,
                "rollout_index": r_idx,
                "experience_hash": exp_hash,
            }
            if save_jsonl is not None:
                record = trajectory_to_bcp_record(
                    trajectory=traj,
                    model_name=self.agent_config.llm_model,
                )
                # Add a `r_idx` top-level key for easy filtering downstream.
                record["r_idx"] = r_idx
                await append_jsonl(save_jsonl, record)
            if on_traj_done is not None:
                try:
                    on_traj_done(traj)
                except Exception:  # noqa: BLE001
                    LOG.exception("on_traj_done callback failed; swallowing")
            return traj

        factories = [
            (lambda q=q, r=r: _one(q, r))
            for q in questions
            for r in range(n)
        ]
        eff_conc = concurrency if concurrency is not None else self.concurrency
        flat = await bounded_gather(factories, concurrency=eff_conc)

        # Group back into per-question slices in the original question order.
        slices: list[RolloutBatchSlice] = []
        for i, q in enumerate(questions):
            slices.append(
                RolloutBatchSlice(
                    qid=q.qid,
                    question=q.question,
                    target=q.target,
                    trajectories=flat[i * n : (i + 1) * n],
                )
            )
        return slices
