"""Concurrency knobs are independent of batch sizes.

- ``rollout.concurrency`` caps simultaneous in-flight rollouts.
- ``eval.concurrency`` is an *override* for the eval phase only.
- Neither changes how many trajectories are produced.
"""

from __future__ import annotations

import asyncio

import pytest

from search_ttl.agent.search_agent import SearchAgentConfig
from search_ttl.rollout.manager import RolloutManager
from search_ttl.rollout.schemas import QuestionRecord, Trajectory


class _Probe:
    """Records semaphore peak via inflight counter, plus output count."""

    def __init__(self) -> None:
        self.inflight = 0
        self.peak = 0
        self._lock = asyncio.Lock()

    async def __call__(self, qid: str, question: str, target: str = "") -> Trajectory:
        async with self._lock:
            self.inflight += 1
            self.peak = max(self.peak, self.inflight)
        # Simulate latency.
        await asyncio.sleep(0.01)
        async with self._lock:
            self.inflight -= 1
        return Trajectory(
            qid=qid,
            question=question,
            target=target,
            experience_snapshot="",
            experience_hash="h",
            status="completed",
        )


def _mgr(probe: _Probe, *, concurrency: int) -> RolloutManager:
    cfg = SearchAgentConfig(
        llm_model="x", llm_base_url="http://x", llm_api_key="x"
    )

    class _StubAgent:
        def __init__(self, *_a, **_kw) -> None:
            self._probe = probe

        async def arun(self, qid, question, target=""):
            return await self._probe(qid, question, target)

    return RolloutManager(
        cfg, concurrency=concurrency, agent_factory=lambda exp: _StubAgent()
    )


def _qs(n: int) -> list[QuestionRecord]:
    return [QuestionRecord(qid=f"q{i}", question=f"Q{i}?", target="t") for i in range(n)]


@pytest.mark.parametrize("concurrency", [1, 4, 16])
def test_concurrency_does_not_change_output_count(concurrency: int) -> None:
    """Whatever the cap, we still get train_batch_size × n trajectories."""
    probe = _Probe()
    mgr = _mgr(probe, concurrency=concurrency)
    slices = asyncio.run(mgr.generate(_qs(4), "exp", n=4))
    assert sum(len(s.trajectories) for s in slices) == 16
    # Peak in-flight ≤ cap.
    assert probe.peak <= concurrency


def test_eval_concurrency_overrides_per_call() -> None:
    """`generate(concurrency=K)` overrides the manager's default."""
    probe = _Probe()
    mgr = _mgr(probe, concurrency=2)  # default tight
    asyncio.run(mgr.generate(_qs(4), "exp", n=1, concurrency=4))
    # With override=4 the peak should reach 4 (vs the default 2).
    assert probe.peak >= 3  # noisy measurement; allow ±1


def test_eval_concurrency_default_falls_back_to_rollout_concurrency() -> None:
    """Omitting the override keeps the manager's value."""
    probe = _Probe()
    mgr = _mgr(probe, concurrency=2)
    asyncio.run(mgr.generate(_qs(8), "exp", n=1))
    assert probe.peak <= 2
