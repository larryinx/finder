"""Run Evaluator.run_split with an experience markdown loaded from disk.

Use this to score a frozen ``best.md`` / ``final.md`` / arbitrary
checkpoint on any split (typically ``test``) without touching the
trainer or the curator. Writes a self-contained eval directory next to
the run, identical in shape to the per-step eval directories the trainer
produces.

Example::

    python scripts/ttl/eval_experience.py \\
        --experience-path runs/ttl/TTL_v1/experience/best.md \\
        --split test \\
        --tag best_test
"""

from __future__ import annotations

import argparse
import asyncio
import os
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

from omegaconf import OmegaConf  # noqa: E402

from search_ttl.agent.search_agent import SearchAgentConfig  # noqa: E402
from search_ttl.agent.tools import probe_mcp  # noqa: E402
from search_ttl.config import compose  # noqa: E402
from search_ttl.config.compose import validate as validate_cfg  # noqa: E402
from search_ttl.evaluator.runner import Evaluator  # noqa: E402
from search_ttl.rollout.manager import RolloutManager  # noqa: E402
from search_ttl.trainer.registry import build_reward, import_default_plugins  # noqa: E402
from search_ttl.utils.logging import banner, good, info, kv_block  # noqa: E402
from search_ttl.utils.tracing import finish as finish_tracing, init_weave  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Eval a frozen experience playbook.")
    ap.add_argument("--experience-path", type=Path, required=True,
                    help="Path to experience markdown (e.g. .../best.md).")
    ap.add_argument("--split", default="test",
                    help="Split key under data.splits to evaluate "
                    "(test | val | train). Default: test.")
    ap.add_argument("--tag", default=None,
                    help="Subdir under <save-path>/eval/ for outputs. "
                    "Default: <split>_<experience-stem>.")
    ap.add_argument("--save-path", default=None,
                    help="Eval-only run dir. Default: parent of <experience-path>/.. "
                    "(i.e. the original training run's root).")
    ap.add_argument("--benchmark", default="bcp")
    ap.add_argument("--retriever", default="qwen3-8b")
    ap.add_argument("--reward", default="search_agent_lasj")
    ap.add_argument("--reflector", default="querychain_staged")
    ap.add_argument("--curator", default="react_staged")
    ap.add_argument("--experience", default="staged_playbook")
    ap.add_argument("--resume", action="store_true",
                    help="Reuse an existing eval/<tag>/rollouts.jsonl: keep "
                    "rollouts that finished cleanly and re-run only the "
                    "failed (status=error) or missing questions.")
    ap.add_argument("-o", "--override", nargs="*", default=[],
                    help="OmegaConf dotlist overrides (e.g. eval.concurrency=32).")
    return ap.parse_args()


async def amain(args: argparse.Namespace) -> None:
    if not args.experience_path.exists():
        print(f"[fatal] experience-path not found: {args.experience_path}",
              file=sys.stderr)
        sys.exit(2)

    # Default save-path = the training run's root, so eval/<tag>/ lands
    # alongside the existing per-step eval directories.
    save_path = (
        Path(args.save_path)
        if args.save_path
        else args.experience_path.resolve().parents[1]
    )
    tag = args.tag or f"{args.split}_{args.experience_path.stem}"

    overrides = list(args.override) + [
        f"trainer.save_path={save_path}",
        # eval-only: never train, never trigger periodic eval
        "trainer.total_training_steps=0",
        "trainer.total_epochs=0",
        "eval.eval_every_n_steps=0",
        # Disable best-tracking (we're scoring an existing snapshot, not improving on it)
        "eval.save_best=false",
    ]
    cfg = compose(
        benchmark=args.benchmark,
        retriever=args.retriever,
        reward=args.reward,
        reflector=args.reflector,
        curator=args.curator,
        experience=args.experience,
        overrides=overrides,
    )

    # Override eval_split + final_eval_split via the override mechanism since
    # both downstream paths read from cfg.eval.eval_split / data.splits[...]
    cfg.eval.eval_split = args.split

    # We don't care about the train file existing (eval-only doesn't read it),
    # but compose's validator checks all splits. Patch the validator's view
    # by removing the train split if it's missing — keeps the check aligned
    # with what we actually use.
    if not Path(cfg.data.splits.train).exists():
        cfg.data.splits.train = str(args.experience_path)  # any existing path
    validate_cfg(cfg)

    os.environ.setdefault(
        "OPENAI_API_KEY",
        os.environ.get("EVAL_JUDGE_API_KEY")
        or os.environ.get("AGENT_LLM_API_KEY")
        or "EMPTY",
    )
    import_default_plugins()

    tool_provider = str(cfg.agent.get("tool_provider", "mcp"))
    if tool_provider == "mcp":
        tool_summary = (
            f"mcp:{cfg.agent.mcp_url} "
            f"(transport={cfg.agent.get('mcp_transport','streamable_http')})"
        )
    elif tool_provider == "serper":
        s = cfg.agent.get("serper", {}) or {}
        _bk = s.get("backup_api_keys") or s.get("backup_api_key") or []
        _bk = [_bk] if isinstance(_bk, str) else list(_bk)
        n_keys = len([k for k in [s.get("api_key"), *_bk] if k])
        tool_summary = (
            f"serper:k={s.get('num_results', 10)} "
            f"gl={s.get('gl','us')} hl={s.get('hl','en')} api_keys={n_keys}"
        )
    elif tool_provider == "ddg":
        d = cfg.agent.get("ddg", {}) or {}
        tool_summary = (
            f"ddg:k={d.get('max_results', 5)} "
            f"region={d.get('region','wt-wt')} backend={d.get('backend','auto')}"
        )
    else:
        tool_summary = f"{tool_provider}:<unknown provider>"

    banner(f"TTL-Search eval — {args.experience_path.name}",
           f"split={args.split} → {save_path}/eval/{tag}/")
    kv_block("config", {
        "experience_path": str(args.experience_path),
        "split": args.split,
        "split_path": str(cfg.data.splits[args.split]),
        "tag": tag,
        "save_path": str(save_path),
        "agent": f"{cfg.agent.llm_model} @ {cfg.agent.llm_base_url}",
        "tool": tool_summary,
        "user_prompt_template": cfg.agent.get("user_prompt_template", "bcp"),
        "rollout.concurrency": cfg.rollout.concurrency,
        "eval.concurrency": cfg.eval.get("concurrency"),
    })

    # Probe MCP up front so a dead retriever fails loud. Serper has
    # no equivalent probe — a bad key surfaces on the first call.
    if tool_provider == "mcp":
        info(f"probing MCP at {cfg.agent.mcp_url} ...")
        tools = await probe_mcp(
            mcp_url=cfg.agent.mcp_url,
            transport=str(cfg.agent.get("mcp_transport", "streamable_http")),
        )
        good(f"MCP OK — tools: {tools}")
    else:
        info(f"tool_provider={tool_provider} — skipping MCP probe")

    # Build the two managers Evaluator needs (no trainer / curator involved).
    agent_cfg = SearchAgentConfig.from_omegaconf(cfg.agent)
    rollout_mgr = RolloutManager(
        agent_cfg,
        concurrency=int(cfg.rollout.concurrency),
        seed=int(cfg.rollout.seed),
    )
    reward_mgr = build_reward(cfg.reward)

    init_weave(
        run_name=f"ttl-eval-{tag}",
        config=OmegaConf.to_container(cfg, resolve=True),
        tags=[
            f"benchmark:{cfg.data.benchmark}",
            f"split:{args.split}",
            f"tag:{tag}",
            "ttl_eval",
        ],
        job_type="ttl_eval",
        phase="ttl_eval",
    )

    # Read the playbook directly off disk — Evaluator only needs the
    # rendered string, not a live Experience instance.
    experience_text = args.experience_path.read_text(encoding="utf-8")
    info(f"loaded experience: {len(experience_text)} chars")

    evaluator = Evaluator(cfg, rollout_mgr, reward_mgr)
    result = await evaluator.run_split(
        experience_text,
        split_path=str(cfg.data.splits[args.split]),
        decrypted_jsonl=str(cfg.data.decrypted_jsonl),
        tag=tag,
        resume=args.resume,
    )
    good(
        f"eval done — accuracy={result.accuracy:.3f} ({result.n_correct}/{result.n_total}) · "
        f"avg n_search={result.avg_n_search:.1f} · avg n_turn={result.avg_n_turn:.1f}"
    )
    info(f"metrics → {result.metrics_path}")
    finish_tracing()


def main() -> None:
    asyncio.run(amain(parse_args()))


if __name__ == "__main__":
    main()
