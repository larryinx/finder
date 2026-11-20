"""TTLTrainer — main training loop:

    rollout      → SearchAgent rollouts
    reward       → structured RewardReport
    advantage    → (skipped; reward text cross-fed)
    update       → ReActEditCurator over Experience
    save ckpt    → CheckpointManager.maybe_save
    resume       → CheckpointManager.maybe_resume (auto/disable/resume_path)

Batch sizing:
    data.train_batch_size           = # questions per training step
    rollout.n                       = G (rollouts per question)
    actor.mini_batch_size       = # trajectories per Curator call
    actor.micro_batch_size      = # trajectories per reward+reflector chunk
    actor.epochs                = # Curator passes per mini-batch
    trainer.total_epochs            = # full passes through the train set
    trainer.total_training_steps    = hard cap on curator steps (null = derive)

The *.concurrency knobs are ORTHOGONAL to batch sizes — they cap how many
things run in parallel inside a phase. They do NOT decide how much data is
processed.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from search_ttl.agent.search_agent import SearchAgentConfig
from search_ttl.experience.base import Experience
from search_ttl.curator.manager import CuratorManager
from search_ttl.curator.schemas import CuratorStepResult
from search_ttl.reflector.manager import ReflectorManager
from search_ttl.reflector.schemas import ReflectionOutput
from search_ttl.reward.manager import RewardManager
from search_ttl.rollout.manager import RolloutManager
from search_ttl.rollout.schemas import QuestionRecord, RolloutBatchSlice, Trajectory
from search_ttl.trainer.checkpoint import CheckpointManager
from search_ttl.trainer.registry import (
    build_experience,
    build_curator,
    build_reflector,
    build_reward,
)
from search_ttl.trainer.sampler import RandomSampler
from search_ttl.utils.logging import (
    CONSOLE,
    banner,
    fmt_counts,
    fmt_seconds,
    good,
    info,
    kv_block,
    section,
    status_counts,
    warn,
)
from search_ttl.utils.persistence import append_jsonl, write_json
from search_ttl.utils.tracing import traceable

LOG = logging.getLogger(__name__)


class TTLTrainer:
    def __init__(self, cfg: DictConfig, train_questions: list[QuestionRecord]) -> None:
        self.cfg = cfg
        self.train_questions = train_questions
        self.global_step = 0

        self.save_root = Path(cfg.trainer.save_path)
        self.save_root.mkdir(parents=True, exist_ok=True)
        self.config_hash = _hash_config(cfg)

        # Build pluggables.
        self.experience: Experience = build_experience(cfg.experience)
        self.reward_mgr: RewardManager = build_reward(cfg.reward)
        self.reflector_mgr: ReflectorManager = build_reflector(cfg.reflector)
        self.curator_mgr: CuratorManager = build_curator(cfg.curator)

        agent_cfg = SearchAgentConfig.from_omegaconf(cfg.agent)
        self.rollout_mgr = RolloutManager(
            agent_cfg,
            concurrency=int(cfg.rollout.concurrency),
            seed=int(cfg.rollout.seed),
        )

        self.sampler = RandomSampler(train_questions, seed=int(cfg.rollout.seed))

        self.ckpt_mgr = CheckpointManager(
            self.save_root,
            save_freq=int(cfg.trainer.save_freq),
            keep_last_n=int(cfg.trainer.get("keep_last_n_checkpoints", -1)),
        )

        # Leftover reflections (FIFO; carried across mini-batch / step).
        self._leftover: list[ReflectionOutput] = []
        self._max_leftover = int(cfg.curator.get("max_leftover_size", 200))

        # Pre-derive the total step count.
        self.steps_per_epoch = max(
            1, len(train_questions) // int(cfg.data.train_batch_size)
        )
        self.total_training_steps = self._resolve_total_training_steps()

    # ── public ────────────────────────────────────────────────────────────
    # NOTE: ``fit`` is intentionally NOT @traceable. We want each
    # ``step`` and each ``run_split`` (eval) to be its own top-level
    # LangSmith trace, not children of one giant fit-trace.
    async def fit(self) -> None:
        self._maybe_resume()
        self._save_initial_experience()
        self._log_run_banner()

        eval_every = int(self.cfg.eval.eval_every_n_steps)
        fit_t0 = _now()

        if (
            bool(self.cfg.trainer.get("val_before_train", False))
            and self.global_step == 0
        ):
            await self._maybe_eval(tag="step_0000")

        # Outer epoch loop (total_epochs × steps_per_epoch).
        # In practice we let total_training_steps cap the inner loop, so
        # epoch is informational unless caller wants epoch-aligned LR
        # schedules (we have none yet).
        total_epochs = int(self.cfg.trainer.get("total_epochs", 1))
        for epoch_idx in range(total_epochs):
            if self.global_step >= self.total_training_steps:
                break
            section(
                f"epoch {epoch_idx + 1}/{total_epochs} "
                f"(global_step={self.global_step}/{self.total_training_steps})"
            )
            while self.global_step < self.total_training_steps:
                # We can't tell here whether this is the last step of the
                # current epoch (depends on epoch boundary, which depends
                # on dataset size). It's the last *training* step iff this
                # is the absolute last step.
                is_last_step = self.global_step + 1 >= self.total_training_steps
                await self.step()
                self.global_step += 1
                self._save_step_experience()

                self.ckpt_mgr.maybe_save(
                    self.global_step,
                    self.experience.to_text(),
                    self.sampler.state(),
                    self._trainer_state(),
                    is_last_step=is_last_step,
                )

                if eval_every > 0 and (
                    is_last_step or self.global_step % eval_every == 0
                ):
                    await self._maybe_eval(
                        tag=f"step_{self.global_step:04d}"
                    )

                if is_last_step:
                    break

        (self.save_root / "experience" / "final.md").write_text(
            self.experience.to_text(), encoding="utf-8"
        )
        section("TTL-Search complete")
        good(
            f"final.md written ({self.experience.char_count()} chars) — "
            f"runs/{self.save_root.name}/experience/final.md"
        )
        best_meta = self.save_root / "experience" / "best.meta.json"
        if best_meta.exists():
            try:
                meta = json.loads(best_meta.read_text(encoding="utf-8"))
                good(
                    f"best.md acc={meta.get('accuracy'):.3f} at {meta.get('tag')}"
                )
            except Exception:  # noqa: BLE001
                pass
        info(f"total wall time: {fmt_seconds(_now() - fit_t0)}")

    @traceable(name="TTLTrainer.step", run_type="chain")
    async def step(self) -> None:
        """One *training* step.

        Schedule (uses batch sizing — see module docstring)::

            ┌─ Phase 1 — Rollout ────────────────────────────────────────┐
            │ Sample data.train_batch_size questions; generate G rollouts │
            │ each. All train_batch_size × G trajectories run            │
            │ concurrently (capped by rollout.concurrency).              │
            └────────────────────────────────────────────────────────────┘
                              │
                              ▼  flat_trajs is task-grouped:
                              │  [q0r0,q0r1,...,q0rG-1, q1r0,..., qNrG-1]
                              ▼
            ┌─ Phase 2/3 — micro-batch loop ─────────────────────────────┐
            │ for chunk in chunks(flat_trajs, micro_batch_size):     │
            │     rewards = await reward.score(chunk)                    │
            │     for task in chunk grouped-by-qid:                      │
            │         reflections = reflector.reflect(task)              │
            │         accumulated.extend(reflections)                    │
            │     while len(accumulated) >= mini_batch_size:         │
            │         curator.update(experience, mini_batch) × epochs│
            └────────────────────────────────────────────────────────────┘

        Constraints (validated at startup, see config/compose.py:validate):
            micro_batch_size % rollout.n  == 0   (chunks land on task boundaries)
            mini_batch_size  % micro_batch_size == 0
        """
        train_batch_size = int(self.cfg.data.train_batch_size)
        G = int(self.cfg.rollout.n)
        mini_batch_size = int(self.cfg.actor.mini_batch_size)
        micro_batch_size = int(self.cfg.actor.micro_batch_size)
        epochs = int(self.cfg.actor.get("epochs", 1))

        step_dir = self.save_root / "steps" / f"step_{self.global_step + 1:04d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        rollouts_path = step_dir / "rollouts.jsonl"
        rewards_path = step_dir / "rewards.jsonl"
        # Reset any leftover per-step jsonl from a prior run that landed
        # in the same ``--save-path``. Without this, ``append_jsonl``
        # accumulates across re-runs and the step-level summary line
        # ("correct N/total") reports cumulative counts (the chunk lines
        # above it stay correct because they count this run's rewards
        # in-memory, but the file-based summary divereges.
        for _p in (rollouts_path, rewards_path,
                   step_dir / "reflections.jsonl"):
            if _p.exists():
                _p.unlink()

        total_epochs = int(self.cfg.trainer.get("total_epochs", 1))
        epoch_idx = min(
            total_epochs, self.global_step // self.steps_per_epoch
        )
        step_in_epoch = (self.global_step % self.steps_per_epoch) + 1
        step_label = (
            f"step {self.global_step + 1}/{self.total_training_steps} "
            f"(epoch {epoch_idx + 1}/{total_epochs} · "
            f"step {step_in_epoch}/{self.steps_per_epoch})"
        )
        step_t0 = _now()

        # ── Phase 1 — Batch rollout (all train_batch_size × G trajectories) ──
        batch_qs = self.sampler.sample(train_batch_size)
        section(
            f"{step_label} · phase 1 rollout · "
            f"{train_batch_size} q × G={G} = {train_batch_size * G} traj "
            f"(concurrency={self.cfg.rollout.concurrency})"
        )
        info(f"sampled qids: {[q.qid for q in batch_qs]}")

        roll_t0 = _now()
        from rich.progress import (
            BarColumn,
            MofNCompleteColumn,
            Progress,
            SpinnerColumn,
            TextColumn,
            TimeElapsedColumn,
        )
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            transient=True,
            console=CONSOLE,
        ) as progress:
            task_id = progress.add_task(
                f"{step_label} rollout", total=train_batch_size * G
            )

            def _on_traj(t: "Trajectory") -> None:
                progress.update(
                    task_id,
                    advance=1,
                    description=f"{step_label} rollout · {t.qid} r{t.r_idx} {t.status}",
                )

            slices: list[RolloutBatchSlice] = await self.rollout_mgr.generate(
                batch_qs,
                self.experience.render(),
                n=G,
                save_jsonl=rollouts_path,
                step=self.global_step + 1,
                on_traj_done=_on_traj,
            )
        flat_trajs: list[Trajectory] = [t for slc in slices for t in slc.trajectories]
        qrec_by_qid = {q.qid: q for q in batch_qs}
        roll_dt = _now() - roll_t0
        traj_status = status_counts(flat_trajs, key="status")
        info(
            f"phase 1 done in {fmt_seconds(roll_dt)} "
            f"(avg {roll_dt / max(1, len(flat_trajs)):.1f}s/traj) — {fmt_counts(traj_status)}"
        )

        # ── Phase 2/3 — micro-batch loop ────────────────────────────────────
        accumulated: list[ReflectionOutput] = list(self._leftover)
        if self._leftover:
            info(f"carrying {len(self._leftover)} leftover reflections from prior step")
        self._leftover = []

        n_chunks = (len(flat_trajs) + micro_batch_size - 1) // micro_batch_size
        section(
            f"{step_label} · phase 2/3 reward+reflector · "
            f"{n_chunks} chunks of {micro_batch_size} (=G means 1 task / chunk)"
        )

        for chunk_idx, chunk_start in enumerate(
            range(0, len(flat_trajs), micro_batch_size)
        ):
            chunk = flat_trajs[chunk_start : chunk_start + micro_batch_size]

            # Reward over the whole chunk (concurrent up to reward.concurrency).
            rwd_t0 = _now()
            rewards = await self.reward_mgr.score(chunk)
            for rw in rewards:
                await append_jsonl(rewards_path, asdict(rw))
            n_correct = sum(1 for r in rewards if r.correctness)
            fm_counts = status_counts(rewards, key="failure_mode")

            # Reflector is per-task. Group chunk by qid (preserves order).
            from collections import OrderedDict
            grouped: OrderedDict[str, list[tuple[Trajectory, Any]]] = OrderedDict()
            for traj, rw in zip(chunk, rewards):
                grouped.setdefault(traj.qid, []).append((traj, rw))

            ref_t0 = _now()
            chunk_reflections: list[ReflectionOutput] = []
            for qid, items in grouped.items():
                trajs = [t for t, _ in items]
                rwds = [r for _, r in items]
                qrec = qrec_by_qid[qid]
                reflections = await self.reflector_mgr.reflect(
                    qrec,
                    trajs,
                    rwds,
                    self.experience.render(),
                    save_dir=step_dir,
                )
                for rfl in reflections:
                    await append_jsonl(
                        step_dir / "reflections.jsonl", _ref_record(rfl)
                    )
                chunk_reflections.extend(reflections)
            accumulated.extend(chunk_reflections)
            ref_dt = _now() - ref_t0
            ref_status = status_counts(chunk_reflections, key="status")

            qids_in_chunk = ",".join(grouped.keys())
            info(
                f"chunk {chunk_idx + 1}/{n_chunks} ({qids_in_chunk}) "
                f"reward {n_correct}/{len(rewards)} correct ({fmt_counts(fm_counts)}) "
                f"in {fmt_seconds(_now() - rwd_t0)} → "
                f"reflector {fmt_counts(ref_status)} in {fmt_seconds(ref_dt)} "
                f"[accumulated={len(accumulated)}/{mini_batch_size}]"
            )

            # Curator triggers once per filled mini-batch.
            #
            # IMPORTANT: a reflection fed into ONE mini-batch is spent. We do
            # NOT re-queue reflections that the curator didn't cite as a
            # source — those were considered and dropped, by definition. An
            # earlier version put them back, which combined with the next
            # chunk's fresh reflections caused the curator to fire over and
            # over inside a single step (each call ≈ 30s, ≈ 11 calls/step ≈
            # 6 min wasted). End-of-step `self._leftover` only carries an
            # under-filled tail (< mini_batch_size) to the next step.
            while len(accumulated) >= mini_batch_size:
                mini_batch = accumulated[:mini_batch_size]
                accumulated = accumulated[mini_batch_size:]
                section(
                    f"{step_label} · phase 4 curator · "
                    f"{len(mini_batch)} reflections × {epochs} epoch(s)"
                )
                opt_t0 = _now()
                last_opt_result: CuratorStepResult | None = None
                for pass_idx in range(epochs):
                    new_exp, opt_result = await self.curator_mgr.update(
                        self.experience,
                        mini_batch,
                        step=self.global_step + 1,
                        save_dir=step_dir,
                    )
                    self.experience = new_exp
                    last_opt_result = opt_result
                    write_json(
                        step_dir / "curator" / f"summary_pass_{pass_idx}.json",
                        {
                            "step": opt_result.step,
                            "pass_idx": pass_idx,
                            "n_ops": len(opt_result.ops),
                            "react_turns_used": opt_result.react_turns_used,
                            "rejected": opt_result.rejected,
                            "rejection_reason": opt_result.rejection_reason,
                            "final_summary": opt_result.final_summary,
                            "considered_reflections": len(mini_batch),
                            "uncited_reflection_ids": opt_result.leftover_reflection_ids,
                        },
                    )
                    op_counts = status_counts(opt_result.ops, key="op")
                    n_uncited = len(opt_result.leftover_reflection_ids)
                    if opt_result.rejected:
                        warn(
                            f"pass {pass_idx + 1}/{epochs} REJECTED — "
                            f"{opt_result.rejection_reason}"
                        )
                    else:
                        info(
                            f"pass {pass_idx + 1}/{epochs}: "
                            f"{fmt_counts(op_counts) or 'no ops'} "
                            f"(react_turns={opt_result.react_turns_used}, "
                            f"{len(mini_batch) - n_uncited}/{len(mini_batch)} "
                            f"reflections cited; {n_uncited} dropped)"
                        )

                good(
                    f"curator mini-batch done in {fmt_seconds(_now() - opt_t0)} · "
                    f"playbook chars={self.experience.char_count()}"
                )

        # End-of-step: save undersized tail to leftover for next step.
        self._leftover = accumulated[-self._max_leftover :]
        if self._leftover:
            warn(
                f"{len(self._leftover)} reflections deferred to next step "
                f"(mini_batch under-fill)"
            )

        # Step-level summary — total correctness from the rewards we wrote.
        all_rewards = _read_rewards_summary(rewards_path)
        step_dt = _now() - step_t0
        info(
            f"{step_label} done in {fmt_seconds(step_dt)} · "
            f"correct {all_rewards['n_correct']}/{all_rewards['n_total']} "
            f"({all_rewards['accuracy']:.1%}) · "
            f"avg n_search={all_rewards['avg_n_search']:.1f} · "
            f"avg n_turn={all_rewards['avg_n_turn']:.1f}"
        )

    # ── internals ─────────────────────────────────────────────────────────
    def _resolve_total_training_steps(self) -> int:
        """Derive from total_epochs when total_training_steps is null/<=0.

            total_training_steps = total_epochs × steps_per_epoch
            steps_per_epoch      = len(train_set) // data.train_batch_size

        Set ``trainer.total_training_steps`` explicitly to override the
        derivation (use a smaller value for quick experiments).
        """
        explicit = self.cfg.trainer.get("total_training_steps", None)
        if explicit not in (None, "null") and int(explicit) > 0:
            return int(explicit)
        return int(self.cfg.trainer.get("total_epochs", 1)) * self.steps_per_epoch

    def _maybe_resume(self) -> None:
        bundle = self.ckpt_mgr.maybe_resume(
            mode=str(self.cfg.trainer.resume_mode),
            resume_from_path=self.cfg.trainer.get("resume_from_path", None),
        )
        if bundle is None:
            return
        if bundle.config_hash and bundle.config_hash != self.config_hash:
            LOG.warning(
                "[ckpt] resuming with config_hash mismatch: "
                "ckpt=%s vs current=%s",
                bundle.config_hash,
                self.config_hash,
            )
        self.global_step = bundle.global_step
        # Reconstruct via the SAME backend class that was built from
        # cfg.experience in __init__ (runs before resume) so the
        # playbook is not mis-parsed on resume.
        self.experience = type(self.experience).from_text(
            bundle.experience_text, cfg=self.cfg.experience
        )
        self.sampler.load_state(bundle.sampler_state)
        good(f"resumed from {bundle.path} → global_step={self.global_step}")

    def _save_initial_experience(self) -> None:
        if self.global_step == 0:
            d = self.save_root / "experience"
            d.mkdir(parents=True, exist_ok=True)
            (d / "step_0000_initial.md").write_text(
                self.experience.to_text(), encoding="utf-8"
            )

    def _save_step_experience(self) -> None:
        d = self.save_root / "experience"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"step_{self.global_step:04d}_after_curator.md").write_text(
            self.experience.to_text(), encoding="utf-8"
        )

    async def _maybe_eval(self, *, tag: str) -> None:
        from search_ttl.evaluator.runner import Evaluator

        if "val" not in self.cfg.data.splits:
            return
        evaluator = Evaluator(self.cfg, self.rollout_mgr, self.reward_mgr)
        await evaluator.run_split(
            self.experience.render(),
            split_path=str(self.cfg.data.splits[self.cfg.eval.eval_split]),
            decrypted_jsonl=str(self.cfg.data.decrypted_jsonl),
            tag=tag,
        )

    def _trainer_state(self) -> dict[str, Any]:
        return {
            "global_step": self.global_step,
            "config_hash": self.config_hash,
            "total_training_steps": self.total_training_steps,
        }

    def _log_run_banner(self) -> None:
        n_train = len(self.train_questions)
        per_step_traj = (
            int(self.cfg.data.train_batch_size) * int(self.cfg.rollout.n)
        )
        steps_breakdown = (
            f"{self.total_training_steps}  "
            f"(= total_epochs {self.cfg.trainer.total_epochs} "
            f"× steps_per_epoch {self.steps_per_epoch})"
        )
        banner("TTL-Search train run", f"exp_name={self.cfg.exp_name}")
        kv_block(
            "config",
            {
                "benchmark": self.cfg.data.benchmark,
                "train_split": f"{n_train} questions ({self.cfg.data.splits.get('train','?')})",
                "agent": f"{self.cfg.agent.llm_model} @ {self.cfg.agent.llm_base_url}",
                "mcp": f"{self.cfg.agent.mcp_url} (transport={self.cfg.agent.get('mcp_transport','streamable_http')})",
                "data.train_batch_size": self.cfg.data.train_batch_size,
                "rollout.n / concurrency": f"{self.cfg.rollout.n} / {self.cfg.rollout.concurrency}",
                "actor.mini/micro/epochs": (
                    f"{self.cfg.actor.mini_batch_size} / "
                    f"{self.cfg.actor.micro_batch_size} / "
                    f"{self.cfg.actor.epochs}"
                ),
                "trajectories per step": per_step_traj,
                "steps_per_epoch": self.steps_per_epoch,
                "trainer.total_epochs": self.cfg.trainer.total_epochs,
                "trainer.total_training_steps": steps_breakdown,
                "trainer.save_freq": self.cfg.trainer.save_freq,
                "trainer.resume_mode": self.cfg.trainer.resume_mode,
                "eval.eval_every_n_steps": self.cfg.eval.eval_every_n_steps,
                "eval.split": self.cfg.eval.eval_split,
                "save_path": str(self.save_root),
            },
        )


def _now() -> float:
    """Module-level time helper — hides `time.perf_counter` import noise."""
    import time

    return time.perf_counter()


def _read_rewards_summary(rewards_path: Path) -> dict[str, float]:
    """Aggregate the rewards.jsonl we just wrote for a one-line step summary."""
    if not rewards_path.exists():
        return {"n_total": 0, "n_correct": 0, "accuracy": 0.0,
                "avg_n_search": 0.0, "avg_n_turn": 0.0}
    n_total = 0
    n_correct = 0
    sum_search = 0
    sum_turn = 0
    with rewards_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            n_total += 1
            if rec.get("correctness"):
                n_correct += 1
            eff = rec.get("efficiency") or {}
            sum_search += int(eff.get("n_search") or 0)
            sum_turn += int(eff.get("n_turn") or 0)
    return {
        "n_total": n_total,
        "n_correct": n_correct,
        "accuracy": n_correct / n_total if n_total else 0.0,
        "avg_n_search": sum_search / n_total if n_total else 0.0,
        "avg_n_turn": sum_turn / n_total if n_total else 0.0,
    }


def _ref_record(r: ReflectionOutput) -> dict:
    return {
        "qid": r.qid,
        "r_idx": r.r_idx,
        "reflector_kind": r.reflector_kind,
        "status": r.status,
        "structured_payload": r.structured_payload,
        "text_for_curator": r.text_for_curator,
        "used_tool_calls": r.used_tool_calls,
        "used_react_turns": r.used_react_turns,
        "transcript_path": str(r.transcript_path) if r.transcript_path else None,
    }


def _hash_config(cfg: DictConfig) -> str:
    """Cheap config fingerprint — used to detect resume vs current drift."""
    container = OmegaConf.to_container(cfg, resolve=True)
    blob = json.dumps(container, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]
