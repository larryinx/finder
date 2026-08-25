"""Re-judge an existing eval directory's rollouts with a (different) judge.

Given an eval dir that already holds ``rollouts.jsonl`` (and usually a
``metrics.json`` + ``rewards.jsonl`` written by an earlier judge), this:

  1. moves any existing ``metrics.json`` / ``rewards.jsonl`` into a
     timestamped backup subfolder of the eval dir (store, then remove
     from the eval dir — the rollouts are left untouched),
  2. rebuilds ``Trajectory`` objects straight from ``rollouts.jsonl``
     (NO rollout, NO MCP, NO agent — the agent's prediction is reused
     verbatim, so this is apples-to-apples with the original eval; only
     the judge model changes),
  3. runs them through the reward manager exactly like ``Evaluator``
     does (same ``search_agent_lasj`` judge / fast exact-match path /
     parser),
  4. writes a fresh ``rewards.jsonl``, and a fresh ``metrics.json`` that
     is the OLD ``metrics.json`` with ONLY ``n_correct`` + ``accuracy``
     overwritten (every other field is preserved verbatim).

The judge model / base_url / api_key / provider are read from the
DEDICATED ``EVAL_JUDGE_*`` env vars (see
``config/ttl/reward/search_agent_lasj.yaml``), which are independent of
the ``AGENT_LLM_*`` vars the rollout used. So you can re-grade
gpt-oss / glm / qwen rollouts with a stronger, fairer judge without
re-running anything.

Example::

    EVAL_JUDGE_MODEL=openai/gpt-5.4-mini \\
    EVAL_JUDGE_BASE_URL=https://openrouter.ai/api/v1 \\
    EVAL_JUDGE_PROVIDER=openrouter \\
    EVAL_JUDGE_API_KEY=sk-or-... \\
    python scripts/ttl/rejudge_rollouts.py runs/ttl/TTL_v2_5_oss/eval/final_test
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import datetime as _dt
import json
import os
import shutil
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT / "third_party" / "ace"))

try:
    from dotenv import load_dotenv  # type: ignore[import-not-found]

    # override=False → real exported EVAL_JUDGE_* env wins over .env.
    load_dotenv(_PROJECT_ROOT / ".env", override=False)
except Exception:
    pass

from search_ttl.config import compose  # noqa: E402
from search_ttl.rollout.schemas import QueryRecord, Trajectory  # noqa: E402
from search_ttl.trainer.registry import build_reward, import_default_plugins  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("eval_dir", type=Path,
                    help="Eval dir containing rollouts.jsonl "
                         "(e.g. runs/ttl/TTL_v2_5_oss/eval/final_test).")
    ap.add_argument("--reward", default="search_agent_lasj",
                    help="Reward kind (config/ttl/reward/<name>.yaml). "
                         "Default: search_agent_lasj (judge-only).")
    ap.add_argument("--benchmark", default="bcp")
    ap.add_argument("--retriever", default="qwen3-8b")
    ap.add_argument("--concurrency", type=int, default=None,
                    help="Override reward.concurrency (judge fan-out).")
    ap.add_argument("--backup-name", default=None,
                    help="Backup subfolder name. Default: "
                         "rejudge_backup_<UTC timestamp>.")
    ap.add_argument("-o", "--override", nargs="*", default=[],
                    help="Extra OmegaConf dotlist overrides "
                         "(e.g. reward.judge.temperature=0.0).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Reconstruct + report, but do NOT move files or "
                         "call the judge.")
    return ap.parse_args()


# ── rollouts.jsonl → Trajectory ──────────────────────────────────────────
#
# Inverse of utils.persistence.trajectory_to_bcp_record. The BCP record
# does not persist Trajectory.queries / .metrics, so we rebuild
# ``queries`` from the step list (needed for the reward's rule-based
# efficiency: redundant_query_count + novel_doc_ratio) and reconstruct a
# best-effort ``metrics`` (search_count is exact from tool_call_counts;
# turns is approximated from the step list since the per-step turn
# counter is not in rollouts.jsonl). The judge itself only consumes
# question / target / extracted_answer / raw_response, all of which ARE
# persisted verbatim — so the verdict is exact, only efficiency
# n_turn is approximate.


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
            txt = " ".join(map(str, chunks)) if isinstance(chunks, list) else str(chunks)
            if txt.strip():
                last_thought = txt.strip()[:600]
        elif t == "tool_call":
            args = s.get("arguments") or {}
            q = args.get("query") or args.get("q") or ""
            n_results = 0
            top_ids: list[str] = []
            # next tool_result step holds what this query returned
            for nxt in steps[i + 1:]:
                if isinstance(nxt, dict) and nxt.get("type") == "tool_result":
                    out = nxt.get("output")
                    if isinstance(out, list):
                        n_results = len(out)
                        top_ids = [
                            str(h.get("docid")) for h in out
                            if isinstance(h, dict) and h.get("docid") is not None
                        ][:10]
                    break
                if isinstance(nxt, dict) and nxt.get("type") == "tool_call":
                    break
            idx += 1
            queries.append(QueryRecord(
                idx=idx, query=str(q), preceding_thought=last_thought,
                n_results=n_results, top_doc_ids=top_ids,
            ))
            last_thought = ""
    return queries


def _record_to_trajectory(rec: dict) -> Trajectory:
    steps = rec.get("result") or []
    golds = rec.get("golden_answers") or []
    target = str(golds[0]) if golds else ""
    counts = rec.get("tool_call_counts") or {}
    n_tool_calls = sum(1 for s in steps
                       if isinstance(s, dict) and s.get("type") == "tool_call")
    meta = rec.get("metadata") or {}
    return Trajectory(
        qid=str(rec.get("query_id", "")),
        question=str(rec.get("question", "")),
        target=target,
        experience_snapshot="",
        experience_hash=str((meta.get("ttl") or {}).get("experience_hash", "")),
        steps=steps,
        queries=_rebuild_queries(steps),
        final_answer=rec.get("prediction_exact") or None,
        extracted_answer=rec.get("prediction_exact") or None,
        status=str(rec.get("status", "completed")),
        retrieved_docids=list(rec.get("retrieved_docids") or []),
        tool_call_counts=dict(counts),
        raw_response=str(rec.get("prediction_raw", "")),
        # search_count exact; turns approximated from steps (per-step
        # counter not persisted in rollouts.jsonl).
        metrics={"search_count": sum(counts.values()), "turns": n_tool_calls},
        metadata_ttl=dict(meta.get("ttl") or {}),
        r_idx=int(rec.get("r_idx", 0) or 0),
        step=-1,
    )


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
    trajs = [_record_to_trajectory(r) for r in records]
    print(f"[rejudge] loaded {len(trajs)} rollouts from {rollouts_path}")

    overrides = list(args.override)
    if args.concurrency is not None:
        overrides.append(f"reward.concurrency={args.concurrency}")
    # Reward-only compose: reflector/curator/experience layers are
    # merged for a well-formed cfg but never built. validate() is NOT
    # called (no data splits / MCP / agent needed for re-judging).
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

    jcfg = cfg.reward.get("judge", {}) or {}
    api_key = str(jcfg.get("api_key", ""))
    print("[rejudge] judge config (from EVAL_JUDGE_* / reward yaml):")
    print(f"           model      = {jcfg.get('model')}")
    print(f"           base_url   = {jcfg.get('base_url')}")
    print(f"           provider   = {jcfg.get('llm_provider')}")
    print(f"           api_key    = {'set (' + api_key[:6] + '…)' if api_key and api_key != 'EMPTY' else api_key!r}")
    print(f"           template   = {jcfg.get('template')}  temp={jcfg.get('temperature')}")
    print(f"           reward.kind={cfg.reward.kind}  concurrency={cfg.reward.get('concurrency')}")
    if str(jcfg.get("llm_provider")) == "openrouter":
        print(f"           OPENROUTER_PROVIDER={os.environ.get('OPENROUTER_PROVIDER')!r} "
              f"FALLBACKS={os.environ.get('OPENROUTER_PROVIDER_FALLBACKS')!r}")

    metrics_path = eval_dir / "metrics.json"
    rewards_path = eval_dir / "rewards.jsonl"
    old_metrics: dict | None = None
    if metrics_path.exists():
        old_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    if args.dry_run:
        em = sum(
            1 for t in trajs
            if (t.extracted_answer or "").strip().lower()
            == (t.target or "").strip().lower() and t.target
        )
        print(f"[dry-run] would back up: "
              f"{[p.name for p in (metrics_path, rewards_path) if p.exists()]}")
        print(f"[dry-run] {len(trajs)} trajectories; "
              f"{em} would short-circuit on exact-match, "
              f"{len(trajs) - em} would hit the judge LLM.")
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
            json.dumps({
                "rejudged_at_utc": ts,
                "eval_dir": str(eval_dir),
                "n_rollouts": len(trajs),
                "old_judge_model": (old_metrics or {}).get("agent_model"),
                "new_judge_model": jcfg.get("model"),
                "new_judge_base_url": jcfg.get("base_url"),
                "new_judge_provider": jcfg.get("llm_provider"),
                "reward_kind": str(cfg.reward.kind),
                "backed_up": [p.name for p in to_backup],
            }, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"[rejudge] moved {[p.name for p in to_backup]} → {backup_dir}")
    else:
        print("[rejudge] no existing metrics.json / rewards.jsonl to back up")

    # ── 2. score with the (new) judge ───────────────────────────────────
    reward_mgr = build_reward(cfg.reward)
    print(f"[rejudge] scoring {len(trajs)} trajectories ...")
    rewards = await reward_mgr.score(trajs)

    n_total = len(rewards)
    n_correct = sum(1 for rw in rewards if rw.correctness)
    accuracy = n_correct / n_total if n_total else 0.0

    # ── 3. write fresh rewards.jsonl (same shape as Evaluator) ───────────
    with rewards_path.open("w", encoding="utf-8") as f:
        for rw in rewards:
            f.write(json.dumps(dataclasses.asdict(rw),
                               ensure_ascii=False, default=str))
            f.write("\n")

    # ── 4. write fresh metrics.json — preserve everything, overwrite
    #       ONLY n_correct + accuracy (n kept in sync defensively) ───────
    new_metrics = dict(old_metrics) if old_metrics else {}
    new_metrics.setdefault("benchmark", str(cfg.data.benchmark))
    new_metrics.setdefault("n", n_total)
    new_metrics["n_correct"] = n_correct
    new_metrics["accuracy"] = accuracy
    metrics_path.write_text(
        json.dumps(new_metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    old_acc = (old_metrics or {}).get("accuracy")
    delta = f"  (was {old_acc:.4f}, Δ={accuracy - old_acc:+.4f})" \
        if isinstance(old_acc, (int, float)) else ""
    print(f"[rejudge] DONE — accuracy={accuracy:.4f} "
          f"({n_correct}/{n_total}){delta}")
    print(f"[rejudge] wrote {rewards_path}")
    print(f"[rejudge] wrote {metrics_path} (only n_correct + accuracy changed)")


def main() -> None:
    asyncio.run(amain(parse_args()))


if __name__ == "__main__":
    main()
