"""Evaluator — runs a split end-to-end with `n=1` and writes BCP-style metrics.

We deliberately reuse RolloutManager (no new agent code) and RewardManager
(in `eval_lite` mode, the per-query analyzer is skipped).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from search_ttl.reward.manager import RewardManager
from search_ttl.rollout.manager import RolloutManager
from search_ttl.rollout.schemas import QuestionRecord
from search_ttl.utils.logging import (
    CONSOLE,
    fmt_seconds,
    good,
    info,
    section,
)
from search_ttl.utils.persistence import (
    append_jsonl,
    read_jsonl,
    write_json,
    write_jsonl,
)
from search_ttl.utils.tracing import traceable


@dataclass
class EvalResult:
    accuracy: float
    n_correct: int
    n_total: int
    avg_n_search: float
    avg_n_turn: float
    metrics_path: Path


class Evaluator:
    def __init__(
        self,
        cfg: DictConfig,
        rollout_mgr: RolloutManager,
        reward_mgr: RewardManager,
    ) -> None:
        self.cfg = cfg
        self.rollout_mgr = rollout_mgr
        self.reward_mgr = reward_mgr

    @traceable(name="Evaluator.run_split", run_type="chain")
    async def run_split(
        self,
        experience_text: str,
        *,
        split_path: str,
        decrypted_jsonl: str,
        tag: str,
        resume: bool = False,
    ) -> EvalResult:
        """Roll out + score a split.

        When ``resume`` is True and an ``eval/<tag>/rollouts.jsonl``
        already exists, rollouts that finished cleanly are kept and
        only the failed (``status == "error"``) or missing questions
        are re-run. ``rollouts.jsonl`` / ``rewards.jsonl`` are trimmed
        to the kept set first, so the final files are exactly one
        record per question.
        """
        from search_ttl.evaluator.loaders import load_split

        rows = load_split(
            str(self.cfg.data.benchmark), split_path, decrypted_jsonl=decrypted_jsonl
        )
        questions = [QuestionRecord.from_bcp_dict(r) for r in rows]
        # Drop any rows the loader produced without questions/golds — shouldn't
        # happen with the curated splits, but guards against an unintentional
        # blank line during data prep.
        questions = [q for q in questions if q.question]

        eval_dir = Path(self.cfg.trainer.save_path) / "eval" / tag
        eval_dir.mkdir(parents=True, exist_ok=True)
        rollouts_path = eval_dir / "rollouts.jsonl"
        rewards_path = eval_dir / "rewards.jsonl"

        # Resume: keep clean rollouts, re-run only failed / missing ones.
        todo = questions
        n_resumed = 0
        if resume:
            todo, n_resumed = self._prune_for_resume(
                questions, rollouts_path, rewards_path
            )
            if n_resumed:
                info(
                    f"resume [{tag}]: kept {n_resumed} finished rollouts, "
                    f"re-running {len(todo)} failed/missing"
                )
                if not todo:
                    info("resume: nothing left to run — re-scoring from disk")
            else:
                info(f"resume [{tag}]: no reusable rollouts — running full split")

        # eval.concurrency is the parallelism cap for THIS eval call.
        # Defaults (when null) to rollout.concurrency — same behaviour as
        # before the knob existed.
        eval_concurrency = self.cfg.eval.get("concurrency", None)
        if eval_concurrency in (None, "null"):
            eval_concurrency = int(self.cfg.rollout.concurrency)
        eval_concurrency = int(eval_concurrency)

        section(
            f"eval [{tag}] · {len(todo)} questions @ "
            f"{Path(split_path).name} (concurrency={eval_concurrency})"
            + (f" · resumed +{n_resumed}" if resume and n_resumed else "")
        )

        from rich.progress import (
            BarColumn,
            MofNCompleteColumn,
            Progress,
            SpinnerColumn,
            TextColumn,
            TimeElapsedColumn,
        )
        import time

        roll_t0 = time.perf_counter()
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            transient=True,
            console=CONSOLE,
        ) as progress:
            task_id = progress.add_task(f"eval [{tag}] rollout", total=len(todo))

            def _on(t):
                progress.update(
                    task_id,
                    advance=1,
                    description=f"eval [{tag}] · {t.qid} {t.status}",
                )

            slices = await self.rollout_mgr.generate(
                todo,
                experience_text,
                n=1,
                save_jsonl=rollouts_path,
                step=-1,
                concurrency=eval_concurrency,
                on_traj_done=_on,
            )
        roll_dt = time.perf_counter() - roll_t0
        info(f"eval rollouts done in {fmt_seconds(roll_dt)}; scoring...")
        score_t0 = time.perf_counter()
        trajs = [s.trajectories[0] for s in slices]
        rewards = await self.reward_mgr.score(trajs)
        info(f"eval scoring done in {fmt_seconds(time.perf_counter() - score_t0)}")

        # Persist this batch's rewards. On a resume run _prune_for_resume
        # has already trimmed rewards.jsonl to the kept set, so this
        # append rebuilds the full file; on a fresh run it is the only
        # writer.
        from dataclasses import asdict

        for rw in rewards:
            await append_jsonl(rewards_path, asdict(rw))

        metrics_path = eval_dir / "metrics.json"
        if resume:
            # Metrics span kept + freshly-run rollouts → recompute from
            # the on-disk jsonl files rather than the in-memory batch.
            n_correct, n_total, accuracy, avg_n_search, avg_n_turn, qa = (
                self._metrics_from_disk(rollouts_path, rewards_path)
            )
        else:
            n_correct = sum(1 for rw in rewards if rw.correctness)
            n_total = len(rewards)
            accuracy = n_correct / n_total if n_total else 0.0
            avg_n_search = (
                sum(t.metrics.get("search_count", 0) for t in trajs) / n_total
                if n_total
                else 0.0
            )
            avg_n_turn = (
                sum(t.metrics.get("turns", 0) for t in trajs) / n_total
                if n_total
                else 0.0
            )
            qa = _aggregate_qa_metrics(rewards)

        out_metrics: dict = {
            "benchmark": self.cfg.data.benchmark,
            "experience_step": tag,
            "n": n_total,
            "n_correct": n_correct,
            "accuracy": accuracy,
            "avg_n_search": avg_n_search,
            "avg_n_turn": avg_n_turn,
            "agent_model": self.cfg.agent.llm_model,
        }
        if qa:
            out_metrics.update(qa)  # f1 / cem / em
        write_json(metrics_path, out_metrics)

        # Maintain best.md if this beats previous best.
        new_best = False
        if bool(self.cfg.eval.get("save_best", True)):
            new_best = self._maybe_update_best(experience_text, accuracy, tag)

        qa_tail = (
            f" · F1={qa['f1']:.3f} CEM={qa['cem']:.3f} EM={qa['em']:.3f}"
            if qa
            else ""
        )
        good(
            f"eval [{tag}] accuracy={accuracy:.3f} ({n_correct}/{n_total}) · "
            f"avg n_search={avg_n_search:.1f} · avg n_turn={avg_n_turn:.1f}"
            + qa_tail
            + ("  [bold green]new best[/bold green]" if new_best else "")
        )

        return EvalResult(
            accuracy=accuracy,
            n_correct=n_correct,
            n_total=n_total,
            avg_n_search=avg_n_search,
            avg_n_turn=avg_n_turn,
            metrics_path=metrics_path,
        )

    # ── resume helpers ────────────────────────────────────────────────────
    @staticmethod
    def _prune_for_resume(
        questions: list[QuestionRecord],
        rollouts_path: Path,
        rewards_path: Path,
    ) -> tuple[list[QuestionRecord], int]:
        """Trim ``rollouts.jsonl`` / ``rewards.jsonl`` to the kept set.

        A rollout is *kept* when it exists for a current-split qid and
        its ``status`` is not ``"error"`` (errors are infra/LLM
        failures worth re-running; ``max_turn`` / ``max_search`` are
        legitimate agent outcomes and are kept). Returns the questions
        still needing a rollout plus the kept count.
        """
        if not rollouts_path.exists():
            return questions, 0
        valid_qids = {q.qid for q in questions}
        kept_by_qid: dict[str, dict] = {}
        for rec in read_jsonl(rollouts_path):
            qid = str(rec.get("query_id", ""))
            if qid not in valid_qids or rec.get("status") == "error":
                continue
            kept_by_qid.setdefault(qid, rec)  # first clean record wins
        done = set(kept_by_qid)
        write_jsonl(rollouts_path, list(kept_by_qid.values()))
        if rewards_path.exists():
            kept_rewards = [
                rw for rw in read_jsonl(rewards_path) if str(rw.get("qid", "")) in done
            ]
            write_jsonl(rewards_path, kept_rewards)
        todo = [q for q in questions if q.qid not in done]
        return todo, len(done)

    @staticmethod
    def _metrics_from_disk(
        rollouts_path: Path, rewards_path: Path
    ) -> tuple[int, int, float, float, float, dict[str, float]]:
        """Recompute aggregates from the on-disk jsonl files — used by the
        resume path where the kept rollouts are not held in memory.

        Returns ``(n_correct, n_total, accuracy, avg_n_search, avg_n_turn,
        qa_metrics)`` where ``qa_metrics`` is an empty dict for non-QA
        rewards or ``{"f1": ..., "cem": ..., "em": ...}`` when the reward
        is qa_em_f1 (or anything else that puts those keys in
        ``RewardReport.extra``).
        """
        rollouts = read_jsonl(rollouts_path)
        rewards = read_jsonl(rewards_path)
        correct_by_qid = {
            str(rw.get("qid", "")): bool(rw.get("correctness")) for rw in rewards
        }
        n_total = len(rollouts)
        n_correct = sum(
            1 for r in rollouts if correct_by_qid.get(str(r.get("query_id", "")), False)
        )

        def _search(rec: dict) -> int:
            m = rec.get("metrics") or {}
            if "search_count" in m:
                return int(m["search_count"])
            return sum(int(v) for v in (rec.get("tool_call_counts") or {}).values())

        def _turn(rec: dict) -> int:
            m = rec.get("metrics") or {}
            if "turns" in m:
                return int(m["turns"])
            # Legacy record (written before `metrics` was persisted):
            # approximate turns by counting tool-call steps.
            return sum(
                1 for s in (rec.get("result") or []) if s.get("type") == "tool_call"
            )

        accuracy = n_correct / n_total if n_total else 0.0
        avg_n_search = sum(_search(r) for r in rollouts) / n_total if n_total else 0.0
        avg_n_turn = sum(_turn(r) for r in rollouts) / n_total if n_total else 0.0
        qa = _aggregate_qa_metrics_from_dicts(rewards)
        return n_correct, n_total, accuracy, avg_n_search, avg_n_turn, qa

    def _maybe_update_best(self, experience_text: str, accuracy: float, tag: str) -> bool:
        best_path = Path(self.cfg.trainer.save_path) / "experience" / "best.md"
        meta_path = Path(self.cfg.trainer.save_path) / "experience" / "best.meta.json"
        prev = -1.0
        if meta_path.exists():
            import json as _json

            prev = float(_json.loads(meta_path.read_text(encoding="utf-8")).get("accuracy", -1.0))
        if accuracy > prev:
            best_path.parent.mkdir(parents=True, exist_ok=True)
            best_path.write_text(experience_text, encoding="utf-8")
            write_json(meta_path, {"accuracy": accuracy, "tag": tag})
            return True
        return False


# ── module-level QA aggregation helpers ──────────────────────────────────


def _aggregate_qa_metrics(rewards) -> dict[str, float]:
    """Mean F1/CEM/EM over a list of ``RewardReport`` objects.

    Returns ``{}`` when no reward stashed those keys in ``extra`` (i.e.
    the reward isn't qa_em_f1 or compatible). The Evaluator merges
    the result into ``metrics.json`` only when it's non-empty.
    """
    if not rewards:
        return {}
    f1s, cems, ems = [], [], []
    for rw in rewards:
        extra = getattr(rw, "extra", {}) or {}
        if "f1" not in extra:
            continue
        f1s.append(float(extra.get("f1", 0.0)))
        cems.append(float(extra.get("cem", 0)))
        ems.append(float(extra.get("em", 0)))
    if not f1s:
        return {}
    n = len(f1s)
    return {"f1": sum(f1s) / n, "cem": sum(cems) / n, "em": sum(ems) / n}


def _aggregate_qa_metrics_from_dicts(reward_dicts: list[dict]) -> dict[str, float]:
    """Same as ``_aggregate_qa_metrics`` but for dict-shaped reward rows
    read straight off ``rewards.jsonl`` (resume path)."""
    f1s, cems, ems = [], [], []
    for rw in reward_dicts:
        extra = rw.get("extra") or {}
        if "f1" not in extra:
            continue
        f1s.append(float(extra.get("f1", 0.0)))
        cems.append(float(extra.get("cem", 0)))
        ems.append(float(extra.get("em", 0)))
    if not f1s:
        return {}
    n = len(f1s)
    return {"f1": sum(f1s) / n, "cem": sum(cems) / n, "em": sum(ems) / n}
