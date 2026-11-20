"""Re-score an existing QA eval dir with the qa_em_f1 reward.

Companion to ``rejudge_rollouts.py``, but for the no-judge HotpotQA /
2Wiki / Musique / Bamboogle path. Two reasons you'd run this:

1. The boxed-answer extractor was buggy (``src/hiexp/evaluation/
   answer_extract.py`` used to silently return ``""`` for
   ``\\boxed{\\text{...}}`` — every such rollout scored 0). The fix is
   in place, but old rollout dirs still hold the stale empty
   ``prediction_exact``. This script re-extracts from
   ``prediction_raw`` so already-collected rollouts benefit from the
   fix WITHOUT re-running the agent.
2. You tweaked the qa_em_f1 reward (e.g. switched ``correctness_metric``
   from ``cem`` to ``em``) and want to refresh the on-disk numbers.

What it does:

  1. Reads ``rollouts.jsonl`` from the eval dir.
  2. Re-extracts ``extracted_answer`` from each ``prediction_raw`` via
     the (now fixed) ``extract_boxed_answer`` — unless ``--no-reextract``
     is given.
  3. Rebuilds ``Trajectory`` objects (including ``golden_answers`` for
     multi-gold benchmarks) and scores them through the qa_em_f1
     reward.
  4. Moves any existing ``metrics.json`` / ``rewards.jsonl`` into a
     timestamped backup subfolder (rollouts.jsonl is left untouched).
  5. Writes a fresh ``rewards.jsonl`` and a fresh ``metrics.json`` that
     is the OLD metrics.json with ONLY the score fields overwritten
     (``n_correct``, ``accuracy``, ``f1``, ``cem``, ``em``) — every
     other field (avg_n_search, avg_n_turn, agent_model, …) is
     preserved verbatim.

No LLM endpoint, no API key, pure CPU. ``--concurrency`` controls only
the in-process fanout of ``RewardManager.score`` (cheap).

Example::

    python scripts/ttl/rejudge_qa_em_f1.py \\
        runs/ttl/TTL_v2_6_oss/eval/final_hotpotqa \\
        --benchmark hotpotqa
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import datetime as _dt
import json
import shutil
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT / "third_party" / "ace"))

try:
    from dotenv import load_dotenv  # type: ignore[import-not-found]

    load_dotenv(_PROJECT_ROOT / ".env", override=False)
except Exception:
    pass

from hiexp.evaluation.answer_extract import extract_boxed_answer  # noqa: E402
from search_ttl.config import compose  # noqa: E402
from search_ttl.rollout.schemas import QueryRecord, Trajectory  # noqa: E402
from search_ttl.trainer.registry import build_reward, import_default_plugins  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "eval_dir",
        type=Path,
        help="Eval dir containing rollouts.jsonl "
             "(e.g. runs/ttl/TTL_v2_6_oss/eval/final_hotpotqa).",
    )
    ap.add_argument(
        "--reward",
        default="qa_em_f1",
        help="Reward kind (config/ttl/reward/<name>.yaml). Default: qa_em_f1.",
    )
    ap.add_argument(
        "--benchmark",
        default="hotpotqa",
        help="Benchmark name for cfg composition (only affects defaults; "
             "the reward itself reads golden_answers from each record). "
             "Default: hotpotqa.",
    )
    ap.add_argument(
        "--retriever",
        default="qwen3-8b",
        help="Retriever cfg slot — unused by qa_em_f1, kept for compose() "
             "to produce a well-formed cfg.",
    )
    ap.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="Override reward.concurrency. Default: leave yaml value as-is.",
    )
    ap.add_argument(
        "--no-reextract",
        action="store_true",
        help="Skip re-extracting the boxed answer from prediction_raw; "
             "score the on-disk prediction_exact verbatim. Use only when "
             "you want to test reward changes (not extractor changes).",
    )
    ap.add_argument(
        "--backup-name",
        default=None,
        help="Backup subfolder name. Default: rejudge_backup_<UTC timestamp>.",
    )
    ap.add_argument(
        "-o",
        "--override",
        nargs="*",
        default=[],
        help="Extra OmegaConf dotlist overrides "
             "(e.g. reward.correctness_metric=em).",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Reconstruct + report (including extractor-change count), "
             "but do NOT move files or write outputs.",
    )
    return ap.parse_args()


# ── rollouts.jsonl → Trajectory ──────────────────────────────────────────
#
# Mirrors ``rejudge_rollouts.py:_record_to_trajectory`` but additionally
# (a) re-extracts the boxed answer from ``prediction_raw`` and (b)
# preserves the full ``golden_answers`` list so the qa_em_f1 reward can
# take the max metric over all accepted golds (HotpotQA / 2Wiki rows
# usually have only one but the reward path supports many).


def _rebuild_queries(steps: list[dict]) -> list[QueryRecord]:
    queries: list[QueryRecord] = []
    last_thought = ""
    idx = 0
    for i, s in enumerate(steps):
        if not isinstance(s, dict):
            continue
        t = s.get("type")
        if t in ("reasoning", "output_text"):
            chunks = s.get("output") or s.get("text") or ""
            txt = (
                " ".join(map(str, chunks))
                if isinstance(chunks, list)
                else str(chunks)
            )
            if txt.strip():
                last_thought = txt.strip()[:600]
        elif t == "tool_call":
            args = s.get("arguments") or {}
            q = args.get("query") or args.get("q") or ""
            n_results = 0
            top_ids: list[str] = []
            for nxt in steps[i + 1:]:
                if isinstance(nxt, dict) and nxt.get("type") == "tool_result":
                    out = nxt.get("output")
                    if isinstance(out, list):
                        n_results = len(out)
                        top_ids = [
                            str(h.get("docid"))
                            for h in out
                            if isinstance(h, dict) and h.get("docid") is not None
                        ][:10]
                    break
                if isinstance(nxt, dict) and nxt.get("type") == "tool_call":
                    break
            idx += 1
            queries.append(
                QueryRecord(
                    idx=idx,
                    query=str(q),
                    preceding_thought=last_thought,
                    n_results=n_results,
                    top_doc_ids=top_ids,
                )
            )
            last_thought = ""
    return queries


def _record_to_trajectory(rec: dict, *, reextract: bool) -> tuple[Trajectory, bool]:
    """Build a Trajectory; return (traj, extracted_value_changed)."""
    steps = rec.get("result") or []
    golds = list(rec.get("golden_answers") or [])
    target = str(golds[0]) if golds else str(rec.get("target") or "")
    counts = rec.get("tool_call_counts") or {}
    n_tool_calls = sum(
        1 for s in steps if isinstance(s, dict) and s.get("type") == "tool_call"
    )
    meta = rec.get("metadata") or {}
    raw = str(rec.get("prediction_raw", ""))
    old_extracted = rec.get("prediction_exact") or ""
    if reextract and raw:
        new_extracted = extract_boxed_answer(raw)
    else:
        new_extracted = old_extracted
    changed = bool(reextract) and (new_extracted != old_extracted)
    traj = Trajectory(
        qid=str(rec.get("query_id", "")),
        question=str(rec.get("question", "")),
        target=target,
        experience_snapshot="",
        experience_hash=str((meta.get("ttl") or {}).get("experience_hash", "")),
        golden_answers=golds,
        steps=steps,
        queries=_rebuild_queries(steps),
        final_answer=new_extracted or None,
        extracted_answer=new_extracted or None,
        status=str(rec.get("status", "completed")),
        retrieved_docids=list(rec.get("retrieved_docids") or []),
        tool_call_counts=dict(counts),
        raw_response=raw,
        metrics={"search_count": sum(counts.values()), "turns": n_tool_calls},
        metadata_ttl=dict(meta.get("ttl") or {}),
        r_idx=int(rec.get("r_idx", 0) or 0),
        step=-1,
    )
    return traj, changed


def _load_rollouts(path: Path) -> list[dict]:
    out: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


async def amain(args: argparse.Namespace) -> None:
    eval_dir: Path = args.eval_dir
    rollouts_path = eval_dir / "rollouts.jsonl"
    if not rollouts_path.exists():
        print(f"[fatal] no rollouts.jsonl in {eval_dir}", file=sys.stderr)
        sys.exit(2)

    records = _load_rollouts(rollouts_path)
    pairs = [_record_to_trajectory(r, reextract=not args.no_reextract) for r in records]
    trajs = [t for t, _ in pairs]
    n_changed = sum(1 for _, c in pairs if c)
    n_now_nonempty = sum(
        1
        for rec, (t, c) in zip(records, pairs)
        if c
        and (t.extracted_answer or "").strip()
        and not (rec.get("prediction_exact") or "").strip()
    )

    print(f"[rejudge-qa] loaded {len(trajs)} rollouts from {rollouts_path}")
    if not args.no_reextract:
        print(
            f"[rejudge-qa] re-extracted boxed answers from prediction_raw: "
            f"{n_changed}/{len(trajs)} changed "
            f"({n_now_nonempty} of those went empty → non-empty)"
        )

    overrides = list(args.override)
    if args.concurrency is not None:
        overrides.append(f"reward.concurrency={args.concurrency}")
    cfg = compose(
        benchmark=args.benchmark,
        retriever=args.retriever,
        reward=args.reward,
        reflector="querychain_staged",
        curator="react_staged",
        experience="staged_playbook",
        overrides=overrides,
    )
    import_default_plugins()
    print(
        f"[rejudge-qa] reward.kind={cfg.reward.kind} "
        f"correctness_metric={cfg.reward.get('correctness_metric', 'cem')} "
        f"concurrency={cfg.reward.get('concurrency')}"
    )

    metrics_path = eval_dir / "metrics.json"
    rewards_path = eval_dir / "rewards.jsonl"
    old_metrics: dict | None = None
    if metrics_path.exists():
        old_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    if args.dry_run:
        print(
            f"[dry-run] would back up: "
            f"{[p.name for p in (metrics_path, rewards_path) if p.exists()]}"
        )
        return

    # ── 1. back up + remove existing metrics.json / rewards.jsonl ────────
    to_backup = [p for p in (metrics_path, rewards_path) if p.exists()]
    if to_backup:
        ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup_dir = eval_dir / (args.backup_name or f"rejudge_backup_{ts}")
        backup_dir.mkdir(parents=True, exist_ok=True)
        for p in to_backup:
            shutil.move(str(p), str(backup_dir / p.name))
        (backup_dir / "rejudge_info.json").write_text(
            json.dumps(
                {
                    "rejudged_at_utc": ts,
                    "eval_dir": str(eval_dir),
                    "n_rollouts": len(trajs),
                    "reward_kind": str(cfg.reward.kind),
                    "correctness_metric": str(cfg.reward.get("correctness_metric", "cem")),
                    "reextracted_boxed": not args.no_reextract,
                    "n_extractions_changed": int(n_changed),
                    "n_empty_to_nonempty": int(n_now_nonempty),
                    "backed_up": [p.name for p in to_backup],
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"[rejudge-qa] moved {[p.name for p in to_backup]} → {backup_dir}")
    else:
        print("[rejudge-qa] no existing metrics.json / rewards.jsonl to back up")

    # ── 2. score with the qa_em_f1 reward ───────────────────────────────
    reward_mgr = build_reward(cfg.reward)
    print(f"[rejudge-qa] scoring {len(trajs)} trajectories ...")
    rewards = await reward_mgr.score(trajs)

    n_total = len(rewards)
    n_correct = sum(1 for rw in rewards if rw.correctness)
    accuracy = n_correct / n_total if n_total else 0.0
    f1 = (
        sum(float(rw.extra.get("f1", 0.0)) for rw in rewards) / n_total
        if n_total
        else 0.0
    )
    cem = (
        sum(float(rw.extra.get("cem", 0)) for rw in rewards) / n_total
        if n_total
        else 0.0
    )
    em = (
        sum(float(rw.extra.get("em", 0)) for rw in rewards) / n_total
        if n_total
        else 0.0
    )

    # ── 3. write fresh rewards.jsonl ─────────────────────────────────────
    with rewards_path.open("w", encoding="utf-8") as f:
        for rw in rewards:
            f.write(
                json.dumps(dataclasses.asdict(rw), ensure_ascii=False, default=str)
            )
            f.write("\n")

    # ── 4. write fresh metrics.json — preserve everything, overwrite the
    #       score fields (n_correct, accuracy, f1, cem, em) ───────────────
    new_metrics = dict(old_metrics) if old_metrics else {}
    new_metrics.setdefault("benchmark", str(cfg.data.benchmark))
    new_metrics.setdefault("n", n_total)
    new_metrics["n_correct"] = n_correct
    new_metrics["accuracy"] = accuracy
    new_metrics["f1"] = f1
    new_metrics["cem"] = cem
    new_metrics["em"] = em
    metrics_path.write_text(
        json.dumps(new_metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    old_acc = (old_metrics or {}).get("accuracy")
    delta = (
        f"  (was {old_acc:.4f}, Δ={accuracy - old_acc:+.4f})"
        if isinstance(old_acc, (int, float))
        else ""
    )
    print(
        f"[rejudge-qa] DONE — accuracy={accuracy:.4f} "
        f"({n_correct}/{n_total}){delta}  ·  F1={f1:.3f} CEM={cem:.3f} EM={em:.3f}"
    )
    print(f"[rejudge-qa] wrote {rewards_path}")
    print(f"[rejudge-qa] wrote {metrics_path}")


def main() -> None:
    asyncio.run(amain(parse_args()))


if __name__ == "__main__":
    main()
