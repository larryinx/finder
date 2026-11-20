"""TTL-Search entry point.

Usage:
    python scripts/ttl/run.py --benchmark bcp --retriever qwen3-8b --mode smoke \\
        --save-path runs/ttl/smoke_v0 \\
        -o data.train_batch_size=2 rollout.n=2 \
           actor.mini_batch_size=4 actor.micro_batch_size=2

Composition order: base → benchmark → tools → reward → reflector →
curator → experience → mode → CLI overrides. (``--retriever`` is
a legacy CLI name that resolves under ``config/ttl/tools/``.)
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Import path side-effects: install project + ace_react + hiexp into sys.path.
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT / "third_party" / "ace"))

# Load .env so env vars resolve in YAML before we instantiate anything.
try:
    from dotenv import load_dotenv  # type: ignore[import-not-found]

    load_dotenv(_PROJECT_ROOT / ".env", override=False)
except Exception:
    pass

from omegaconf import OmegaConf  # noqa: E402

from search_ttl.agent.tools import probe_mcp  # noqa: E402
from search_ttl.config import compose, list_options  # noqa: E402
from search_ttl.config.compose import validate as validate_cfg  # noqa: E402
from search_ttl.rollout.schemas import QuestionRecord  # noqa: E402
from search_ttl.trainer.registry import import_default_plugins  # noqa: E402
from search_ttl.trainer.ttl_trainer import TTLTrainer  # noqa: E402
from search_ttl.utils.tracing import finish as finish_tracing, init_weave  # noqa: E402


def parse_args() -> argparse.Namespace:
    bench = " | ".join(list_options("benchmark")) or "(none)"
    retr = " | ".join(list_options("tools")) or "(none)"
    rew = " | ".join(list_options("reward")) or "(none)"
    refl = " | ".join(list_options("reflector")) or "(none)"
    opt_ = " | ".join(list_options("curator")) or "(none)"
    exp_ = " | ".join(list_options("experience")) or "(none)"
    mode = " | ".join(list_options("mode")) or "(none)"
    ap = argparse.ArgumentParser(description="Run TTL-Search.")
    ap.add_argument("--benchmark", default="bcp", help=f"benchmark/<x>.yaml: {bench}")
    ap.add_argument("--retriever", default="qwen3-8b", help=f"retriever/<x>.yaml: {retr}")
    ap.add_argument("--reward", default="search_agent_lasj", help=f"reward/<x>.yaml: {rew}")
    ap.add_argument("--reflector", default="querychain_staged", help=f"reflector/<x>.yaml: {refl}")
    ap.add_argument("--curator", default="react_staged", help=f"curator/<x>.yaml: {opt_}")
    ap.add_argument(
        "--experience", default="staged_playbook", help=f"experience/<x>.yaml: {exp_}"
    )
    ap.add_argument("--mode", default=None, help=f"mode/<x>.yaml: {mode}")
    ap.add_argument("--save-path", required=True, help="run output dir")
    ap.add_argument(
        "-o",
        "--override",
        nargs="*",
        default=[],
        help="OmegaConf dotlist overrides applied after YAML merge",
    )
    return ap.parse_args()


async def main_async(cfg) -> None:
    # Probe the search backend early — saves 5 min of opaque failures
    # down the line. Only MCP has a meaningful probe; other providers
    # (e.g. serper) surface bad credentials on the first real call.
    tool_provider = str(cfg.agent.get("tool_provider", "mcp"))
    if tool_provider == "mcp":
        transport = str(cfg.agent.get("mcp_transport", "streamable_http"))
        print(f"[probe] checking MCP at {cfg.agent.mcp_url} (transport={transport}) ...")
        names = await probe_mcp(mcp_url=cfg.agent.mcp_url, transport=transport)
        print(f"[probe] MCP OK — tools: {names}")
    else:
        print(f"[probe] tool_provider={tool_provider} — skipping MCP probe")

    # Load training questions through the benchmark's registered loader.
    from search_ttl.evaluator.loaders import load_split

    rows = load_split(
        str(cfg.data.benchmark), cfg.data.splits.train,
        decrypted_jsonl=cfg.data.decrypted_jsonl,
    )
    questions = [QuestionRecord.from_bcp_dict(r) for r in rows]
    train_bs = int(cfg.data.train_batch_size)
    G = int(cfg.rollout.n)
    print(
        f"[run] mode={cfg.get('mode', 'train')} train={len(questions)}"
        f" total_epochs={cfg.trainer.total_epochs}"
        f" total_training_steps={cfg.trainer.total_training_steps}"
        f" data.train_batch_size={train_bs} rollout.n={G}"
        f" actor.mini_batch_size={cfg.actor.mini_batch_size}"
        f" actor.micro_batch_size={cfg.actor.micro_batch_size}"
    )

    init_weave(
        run_name=f"ttl-{cfg.exp_name}-{cfg.get('mode', 'train')}",
        config=OmegaConf.to_container(cfg, resolve=True),
        tags=list(cfg.tracing.get("tags", [])) + [f"benchmark:{cfg.data.benchmark}"],
        job_type="ttl_train",
        phase="ttl_train",
    )

    trainer = TTLTrainer(cfg, train_questions=questions)
    await trainer.fit()
    finish_tracing()


def main() -> None:
    args = parse_args()
    # Mirror EVAL_JUDGE_API_KEY → OPENAI_API_KEY for libs that hard-require it.
    os.environ.setdefault(
        "OPENAI_API_KEY",
        os.environ.get("EVAL_JUDGE_API_KEY")
        or os.environ.get("AGENT_LLM_API_KEY")
        or "EMPTY",
    )

    import_default_plugins()
    overrides = list(args.override) + [f"trainer.save_path={args.save_path}"]
    cfg = compose(
        benchmark=args.benchmark,
        retriever=args.retriever,
        reward=args.reward,
        reflector=args.reflector,
        curator=args.curator,
        experience=args.experience,
        mode=args.mode,
        overrides=overrides,
    )
    validate_cfg(cfg)
    asyncio.run(main_async(cfg))


if __name__ == "__main__":
    main()
