"""TTLTrainer.step schedule: batch rollout, micro-batch reward+reflector,
mini-batch curator.

Mocks the four managers to verify call ordering + counts under different
batch settings.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from search_ttl.experience.staged_playbook import StagedPlaybook
from search_ttl.curator.schemas import CuratorStepResult
from search_ttl.reflector.schemas import ReflectionOutput
from search_ttl.reward.schemas import (
    EfficiencyStats,
    JudgeResult,
    RewardReport,
)
from search_ttl.rollout.schemas import (
    QuestionRecord,
    RolloutBatchSlice,
    Trajectory,
)


def _traj(qid: str, r_idx: int) -> Trajectory:
    return Trajectory(
        qid=qid,
        question=f"q-{qid}",
        target="t",
        experience_snapshot="exp",
        experience_hash="h",
        r_idx=r_idx,
        status="completed",
    )


def _reward(qid: str, r_idx: int) -> RewardReport:
    return RewardReport(
        qid=qid,
        r_idx=r_idx,
        correctness=False,
        judge=JudgeResult(False, None, None, None, False, ""),
        per_query_analysis=[],
        efficiency=EfficiencyStats(0, 0, False, False, 0, 0.0),
        failure_mode="bad_query",
        text_for_reflector="",
    )


class _MockRolloutManager:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], int]] = []

    async def generate(self, questions, experience_text, *, n,
                       save_jsonl=None, step=0,
                       concurrency=None, on_traj_done=None):
        self.calls.append(([q.qid for q in questions], n))
        slices = [
            RolloutBatchSlice(
                qid=q.qid,
                question=q.question,
                target=q.target,
                trajectories=[_traj(q.qid, r) for r in range(n)],
            )
            for q in questions
        ]
        if on_traj_done is not None:
            for slc in slices:
                for t in slc.trajectories:
                    on_traj_done(t)
        return slices


class _MockReward:
    def __init__(self) -> None:
        self.score_call_sizes: list[int] = []  # how many trajs each call got
        self.score_call_qids: list[list[str]] = []

    async def score(self, trajectories):
        self.score_call_sizes.append(len(trajectories))
        self.score_call_qids.append([t.qid for t in trajectories])
        return [_reward(t.qid, t.r_idx) for t in trajectories]


class _MockReflector:
    def __init__(self) -> None:
        self.call_qids: list[str] = []
        self.call_sizes: list[int] = []

    async def reflect(self, qrec, trajectories, rewards, experience_text, *, save_dir=None):
        self.call_qids.append(qrec.qid)
        self.call_sizes.append(len(trajectories))
        return [
            ReflectionOutput(
                qid=t.qid,
                r_idx=t.r_idx,
                reflector_kind="mock",
                status="ok",
                structured_payload={"status": "ok"},
                text_for_curator="",
            )
            for t in trajectories
        ]


class _MockCurator:
    def __init__(self) -> None:
        self.calls: list[int] = []

    async def update(self, experience, reflections, *, step, save_dir=None):
        self.calls.append(len(reflections))
        return experience, CuratorStepResult(
            step=step,
            ops=[],
            final_summary="",
            leftover_reflection_ids=[],
            react_turns_used=0,
        )


def _trainer(tmp_path: Path, *, train_batch_size: int, n: int,
             mini: int, micro: int, epochs: int = 1):
    from search_ttl.trainer.ttl_trainer import TTLTrainer

    cfg = OmegaConf.create(
        {
            "exp_name": "test",
            "data": {
                "benchmark": "test",
                "decrypted_jsonl": str(tmp_path / "fake.jsonl"),
                "splits": {
                    "train": str(tmp_path / "t.tsv"),
                    "val": str(tmp_path / "v.tsv"),
                    "test": str(tmp_path / "te.tsv"),
                },
                "train_batch_size": train_batch_size,
            },
            "agent": {"llm_model": "x", "llm_base_url": "http://x",
                      "llm_api_key": "x", "mcp_url": "http://x"},
            "rollout": {"n": n, "concurrency": 16, "seed": 0},
            "actor": {
                "mini_batch_size": mini,
                "micro_batch_size": micro,
                "epochs": epochs,
            },
            "reward": {"kind": "search_agent_lasj"},
            "reflector": {"kind": "querychain_staged"},
            "curator": {
                "kind": "react_staged",
                "max_leftover_size": 50,
                "conservativeness": {
                    "max_add_per_step": 5, "max_modify_per_step": 3,
                    "max_delete_per_step": 2, "max_total_edits_per_step": 8,
                },
            },
            "experience": {"kind": "staged_playbook", "init_from": None},
            "eval": {"eval_every_n_steps": 0, "eval_split": "val", "save_best": False},
            "trainer": {
                "save_path": str(tmp_path / "run"),
                "save_freq": 0, "keep_last_n_checkpoints": -1,
                "resume_mode": "disable", "resume_from_path": None,
                "val_before_train": False,
                "total_epochs": 1, "total_training_steps": 1,
            },
            "tracing": {"weave_disabled": True, "tags": []},
        }
    )
    qs = [QuestionRecord(qid=f"q{i}", question=f"Q{i}?", target="t")
          for i in range(train_batch_size)]
    t = TTLTrainer.__new__(TTLTrainer)
    t.cfg = cfg
    t.train_questions = qs
    t.global_step = 0
    t.save_root = tmp_path / "run"
    t.save_root.mkdir(parents=True, exist_ok=True)
    t.config_hash = "test"
    t.experience = StagedPlaybook.from_omegaconf(cfg.experience)
    t.reward_mgr = _MockReward()
    t.reflector_mgr = _MockReflector()
    t.curator_mgr = _MockCurator()
    t.rollout_mgr = _MockRolloutManager()

    # Deterministic non-duplicate sampler for testing — returns the questions
    # in original order. Avoids RandomSampler's with-replacement collisions
    # collapsing into the reflector's groupby.
    class _SeqSampler:
        def __init__(self, qs):
            self.qs = qs

        def sample(self, n):
            return list(self.qs[:n])

        def state(self):
            return {"seed": 0, "cursor": 0}

        def load_state(self, d):
            pass

    t.sampler = _SeqSampler(qs)
    from search_ttl.trainer.checkpoint import CheckpointManager

    t.ckpt_mgr = CheckpointManager(t.save_root, save_freq=0)
    t._leftover = []
    t._max_leftover = 50
    t.steps_per_epoch = max(1, len(qs) // train_batch_size)
    t.total_training_steps = 1
    return t


def test_default_on_policy(tmp_path):
    """train_bs=4, n=4, mini=16, micro=4, epochs=1.

    Expect: 1 rollout call (16 trajs), 4 reward chunks (4 each), 4 reflector
    calls (one per task chunk, 4 trajs each), 1 curator call (16 reflections).

    The fixture installs a sequential (deterministic) sampler so the 4 qids
    are unique q0..q3; this isolates schedule order from RNG collisions.
    """
    t = _trainer(tmp_path, train_batch_size=4, n=4, mini=16, micro=4)
    asyncio.run(t.step())

    # 1 batched rollout with all 4 sampled questions (in sample order).
    assert len(t.rollout_mgr.calls) == 1
    sampled_qids, n = t.rollout_mgr.calls[0]
    assert len(sampled_qids) == 4 and n == 4

    # Reward sees 4 chunks of 4. Each chunk is one task's G trajectories,
    # so all qids inside one chunk are identical (= sampled_qids[i]).
    assert t.reward_mgr.score_call_sizes == [4, 4, 4, 4]
    for i, qids_in_chunk in enumerate(t.reward_mgr.score_call_qids):
        assert qids_in_chunk == [sampled_qids[i]] * 4

    # Reflector called per chunk (= per task) in sample order.
    assert t.reflector_mgr.call_qids == sampled_qids
    assert t.reflector_mgr.call_sizes == [4, 4, 4, 4]

    # 1 curator call with all 16 reflections.
    assert t.curator_mgr.calls == [16]


def test_off_policy_mini_equals_micro(tmp_path):
    """train_bs=4, n=4, mini=4, micro=4 → curator fires 4× per step."""
    t = _trainer(tmp_path, train_batch_size=4, n=4, mini=4, micro=4)
    asyncio.run(t.step())

    sampled_qids = t.rollout_mgr.calls[0][0]
    assert t.reward_mgr.score_call_sizes == [4, 4, 4, 4]
    assert t.reflector_mgr.call_qids == sampled_qids
    assert t.curator_mgr.calls == [4, 4, 4, 4]


def test_two_tasks_per_chunk(tmp_path):
    """micro=8 with G=4, train_bs=4 → 2 reward chunks of 8; each chunk
    spans 2 tasks → 2 reflector calls per chunk = 4 total."""
    t = _trainer(tmp_path, train_batch_size=4, n=4, mini=16, micro=8)
    asyncio.run(t.step())

    sampled_qids = t.rollout_mgr.calls[0][0]
    assert t.reward_mgr.score_call_sizes == [8, 8]
    # Chunk 0 covers tasks 0–1; chunk 1 covers tasks 2–3.
    assert t.reward_mgr.score_call_qids[0] == (
        [sampled_qids[0]] * 4 + [sampled_qids[1]] * 4
    )
    assert t.reward_mgr.score_call_qids[1] == (
        [sampled_qids[2]] * 4 + [sampled_qids[3]] * 4
    )
    # Reflector still per-task.
    assert t.reflector_mgr.call_qids == sampled_qids
    assert t.curator_mgr.calls == [16]


def test_epochs_repeats_curator(tmp_path):
    """epochs=2 → curator.update called 2× for the same mini-batch."""
    t = _trainer(tmp_path, train_batch_size=4, n=4, mini=16, micro=4,
                 epochs=2)
    asyncio.run(t.step())
    # 1 mini-batch × 2 epochs = 2 curator calls (each over 16 reflections).
    assert t.curator_mgr.calls == [16, 16]
