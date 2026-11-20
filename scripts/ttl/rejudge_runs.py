"""Re-judge an ACE / HiExp run directory's per-query results with a (different) judge.

This is the ``run_*.json``-per-query analogue of
``scripts/ttl/rejudge_rollouts.py`` (which consumes a single
``rollouts.jsonl``). ACE and HiExp both persist one ``run_<query_id>.json``
per query in the SAME BCP-record shape as a ``rollouts.jsonl`` line
(``question`` / ``result`` / ``tool_call_counts`` / ``prediction_raw`` /
``prediction_exact`` / ``metadata``), so the Trajectory reconstruction is
reused verbatim from ``rejudge_rollouts`` — only the judge model changes,
the agent's prediction is reused as-is (NO rollout, NO MCP, NO agent),
keeping this apples-to-apples with the original eval.

Given a run dir holding ``run_*.json`` this:

  1. backfills golden answers:
       * HiExp run files already embed ``golden_answers`` → used as-is,
       * ACE run files do NOT → backfilled by ``query_id`` from the BCP
         decrypted jsonl via the same loader the eval harness uses
         (``hiexp_prompt._shared.load_browsecomp_plus_jsonl``),
  2. rebuilds ``Trajectory`` objects (reusing
     ``rejudge_rollouts._record_to_trajectory`` — identical to the TTL
     re-judge path),
  3. moves any existing ``metrics.json`` / ``rewards.jsonl`` into a
     timestamped backup subfolder of the run dir (the ``run_*.json``
     files — incl. HiExp's embedded ``scoring`` block — are left
     untouched, exactly like ``rejudge_rollouts`` leaves
     ``rollouts.jsonl`` alone),
  4. runs them through the reward manager exactly like ``Evaluator``,
  5. writes a fresh ``rewards.jsonl`` and a ``metrics.json``: if one
     existed (HiExp) every field is preserved verbatim EXCEPT ``n`` +
     ``n_correct`` + ``accuracy``, which always describe the run files
     scored by this invocation; if none existed (ACE) a minimal one is
     synthesised.

The judge model / base_url / api_key / provider come from the DEDICATED
``EVAL_JUDGE_*`` env vars (see ``config/ttl/reward/search_agent_lasj.yaml``),
independent of the ``AGENT_LLM_*`` the original rollout used.

Example::

    EVAL_JUDGE_MODEL=openai/gpt-5.4-mini \\
    EVAL_JUDGE_BASE_URL=https://openrouter.ai/api/v1 \\
    EVAL_JUDGE_PROVIDER=openrouter \\
    EVAL_JUDGE_API_KEY=sk-or-... \\
    python scripts/ttl/rejudge_runs.py \\
        runs/ace/gptoss_r8b_630/ace_offline_gt/runs/final_test
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
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for rejudge_rollouts

try:
    from dotenv import load_dotenv  # type: ignore[import-not-found]

    # override=False → real exported EVAL_JUDGE_* env wins over .env.
    load_dotenv(_PROJECT_ROOT / ".env", override=False)
except Exception:
    pass

from search_ttl.config import compose  # noqa: E402
from search_ttl.trainer.registry import build_reward, import_default_plugins  # noqa: E402

# Reuse the EXACT rollouts.jsonl → Trajectory reconstruction so re-judging
# an ACE/HiExp run dir is byte-for-byte the same path as re-judging a TTL
# rollouts.jsonl (same query rebuild, same metrics approximation).
from rejudge_rollouts import _record_to_trajectory  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path,
                    help="Dir containing run_*.json (ACE: "
                         "runs/.../runs/final_test ; HiExp: the eval "
                         "output dir). If the dir itself has none but "
                         "runs/final_test does, that subdir is used.")
    ap.add_argument("--reward", default="search_agent_lasj",
                    help="Reward kind (config/ttl/reward/<name>.yaml). "
                         "Default: search_agent_lasj (judge-only).")
    ap.add_argument("--benchmark", default="bcp")
    ap.add_argument("--retriever", default="qwen3-8b")
    ap.add_argument("--decrypted-jsonl", default=None,
                    help="Override the BCP decrypted jsonl used to "
                         "backfill ACE golden answers. Default: "
                         "cfg.data.decrypted_jsonl.")
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


def _resolve_run_dir(run_dir: Path) -> Path:
    """Return a dir that actually holds run_*.json.

    Convenience for the two paths the user passes: if ``run_dir`` itself
    has no ``run_*.json`` but ``run_dir/runs/final_test`` does, descend
    into it (ACE's parent save-dir → its final_test run dir).
    """
    if list(run_dir.glob("run_*.json")):
        return run_dir
    cand = run_dir / "runs" / "final_test"
    if cand.is_dir() and list(cand.glob("run_*.json")):
        print(f"[rejudge] {run_dir} has no run_*.json; using {cand}")
        return cand
    return run_dir


def _qid_sort_key(p: Path):
    stem = p.stem[len("run_"):] if p.stem.startswith("run_") else p.stem
    try:
        return (0, int(stem))
    except ValueError:
        return (1, stem)


def _load_run_files(run_dir: Path) -> list[dict]:
    files = sorted(run_dir.glob("run_*.json"), key=_qid_sort_key)
    out: list[dict] = []
    for fp in files:
        with fp.open("r", encoding="utf-8") as f:
            out.append(json.load(f))
    return out


def _backfill_golden(records: list[dict], decrypted_jsonl: Path) -> int:
    """Fill ``golden_answers`` on records missing it (ACE), by query_id.

    Returns the number of records backfilled. HiExp records already carry
    ``golden_answers`` and are left untouched.
    """
    need = {
        str(r.get("query_id", "")): r
        for r in records
        if not r.get("golden_answers") and r.get("query_id") is not None
    }
    if not need:
        return 0

    from hiexp_prompt._shared import load_browsecomp_plus_jsonl  # noqa: E402

    print(f"[rejudge] backfilling {len(need)} golden answers by query_id "
          f"from {decrypted_jsonl} (this reads the BCP jsonl once) ...")
    examples = load_browsecomp_plus_jsonl(
        decrypted_jsonl, only_query_ids=set(need)
    )
    qid_to_gold = {
        ex.query_id: ex.golden_answers for ex in examples if ex.golden_answers
    }
    n = 0
    for qid, rec in need.items():
        gold = qid_to_gold.get(qid)
        if gold:
            rec["golden_answers"] = list(gold)
            n += 1
    missing = [q for q in need if q not in qid_to_gold]
    if missing:
        print(f"[rejudge][warn] {len(missing)} query_ids had no golden "
              f"answer in the jsonl (will score as target=''): "
              f"{missing[:10]}{' …' if len(missing) > 10 else ''}")
    return n


def _find_old_ace_accuracy(run_dir: Path) -> float | None:
    """Best-effort: ACE's canonical accuracy lives in a sibling
    ``ace_run_*/final_test_results.json`` (NOT a metrics.json). Read-only,
    purely for the printed delta."""
    for base in [run_dir, *run_dir.parents[:4]]:
        for frj in base.glob("ace_run_*/final_test_results.json"):
            try:
                tr = json.loads(frj.read_text(encoding="utf-8")).get(
                    "test_results", {})
                acc = tr.get("accuracy")
                if isinstance(acc, (int, float)):
                    return float(acc)
            except Exception:
                pass
    return None


async def amain(args: argparse.Namespace) -> None:
    run_dir = _resolve_run_dir(args.run_dir)
    records = _load_run_files(run_dir)
    if not records:
        print(f"[fatal] no run_*.json in {run_dir}", file=sys.stderr)
        sys.exit(2)
    print(f"[rejudge] loaded {len(records)} run files from {run_dir}")

    overrides = list(args.override)
    if args.concurrency is not None:
        overrides.append(f"reward.concurrency={args.concurrency}")
    # Reward-only compose (same as rejudge_rollouts): other layers merged
    # for a well-formed cfg but never built; validate() NOT called.
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

    decrypted = Path(args.decrypted_jsonl) if args.decrypted_jsonl \
        else _PROJECT_ROOT / str(cfg.data.decrypted_jsonl)
    n_backfilled = _backfill_golden(records, decrypted)
    if n_backfilled:
        print(f"[rejudge] backfilled golden answers for {n_backfilled} "
              f"records (ACE); HiExp records use their embedded ones")

    trajs = [_record_to_trajectory(r) for r in records]

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

    metrics_path = run_dir / "metrics.json"
    rewards_path = run_dir / "rewards.jsonl"
    old_metrics: dict | None = None
    if metrics_path.exists():
        old_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    if args.dry_run:
        em = sum(
            1 for t in trajs
            if (t.extracted_answer or "").strip().lower()
            == (t.target or "").strip().lower() and t.target
        )
        no_gold = sum(1 for t in trajs if not t.target)
        print(f"[dry-run] would back up: "
              f"{[p.name for p in (metrics_path, rewards_path) if p.exists()]}")
        print(f"[dry-run] {len(trajs)} trajectories; {em} would "
              f"short-circuit on exact-match, {len(trajs) - em} would hit "
              f"the judge LLM; {no_gold} have no golden answer.")
        return

    # ── 1. back up + remove existing metrics.json / rewards.jsonl ────────
    to_backup = [p for p in (metrics_path, rewards_path) if p.exists()]
    if to_backup:
        ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup_dir = run_dir / (args.backup_name or f"rejudge_backup_{ts}")
        backup_dir.mkdir(parents=True, exist_ok=True)
        for p in to_backup:
            shutil.move(str(p), str(backup_dir / p.name))
        (backup_dir / "rejudge_info.json").write_text(
            json.dumps({
                "rejudged_at_utc": ts,
                "run_dir": str(run_dir),
                "n_runs": len(trajs),
                "n_golden_backfilled": n_backfilled,
                "old_judge_model": (old_metrics or {}).get("agent_model")
                or (old_metrics or {}).get("llm_model"),
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

    # ── 4. write fresh metrics.json — preserve everything (HiExp),
    #       overwrite ONLY n + n_correct + accuracy; synthesise if none (ACE) ─
    # ``n`` is assigned, not setdefault-ed: it must describe what THIS
    # re-judge actually scored, same as n_correct/accuracy. An old
    # metrics.json carries the run count from whenever it was written, so
    # preserving it silently desyncs the three the moment run_*.json files
    # are added to the dir (e.g. after re-running the queries that had no
    # run file) — you get accuracy = n_correct/630 sitting next to n=619.
    new_metrics = dict(old_metrics) if old_metrics else {}
    new_metrics.setdefault("benchmark", str(cfg.data.benchmark))
    new_metrics.setdefault("retriever", args.retriever)
    new_metrics["n"] = n_total
    new_metrics["n_correct"] = n_correct
    new_metrics["accuracy"] = accuracy
    metrics_path.write_text(
        json.dumps(new_metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    old_acc = (old_metrics or {}).get("accuracy")
    if not isinstance(old_acc, (int, float)):
        old_acc = _find_old_ace_accuracy(run_dir)  # ACE: from final_test_results.json
    delta = f"  (was {old_acc:.4f}, Δ={accuracy - old_acc:+.4f})" \
        if isinstance(old_acc, (int, float)) else ""
    print(f"[rejudge] DONE — accuracy={accuracy:.4f} "
          f"({n_correct}/{n_total}){delta}")
    print(f"[rejudge] wrote {rewards_path}")
    print(f"[rejudge] wrote {metrics_path} "
          f"({'only n_correct + accuracy changed' if old_metrics else 'synthesised'})")


def main() -> None:
    asyncio.run(amain(parse_args()))


if __name__ == "__main__":
    main()
