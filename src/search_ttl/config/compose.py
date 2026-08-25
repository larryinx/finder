"""YAML composition — base + benchmark + tools + reward + reflector + curator + experience + mode + CLI overrides.

The ``tools`` group used to be called ``retriever``; the CLI flag
remains ``--retriever`` for backwards compatibility (every shell
under ``scripts/experiments/ttl/`` still passes ``--retriever
qwen3-8b``). Internally we resolve names under
``config/ttl/tools/<name>.yaml``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from omegaconf import DictConfig, OmegaConf

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_ROOT = _PROJECT_ROOT / "config" / "ttl"


def _load(path: Path) -> DictConfig:
    return OmegaConf.load(path) if path.exists() else OmegaConf.create({})


def list_options(group: str) -> list[str]:
    """List YAML stem names under ``config/ttl/<group>/``."""
    folder = CONFIG_ROOT / group
    if not folder.is_dir():
        return []
    return sorted(p.stem for p in folder.glob("*.yaml"))


def compose(
    *,
    benchmark: str = "bcp",
    retriever: str = "qwen3-8b",
    reward: str | None = "search_agent_lasj",
    reflector: str | None = "querychain_staged",
    curator: str | None = "react_staged",
    experience: str | None = "staged_playbook",
    mode: str | None = None,
    overrides: Iterable[str] = (),
) -> DictConfig:
    """Build the run config by merging YAML layers + CLI overrides.

    Layer order (later wins):
        base → benchmark → tools → reward → reflector → curator →
        experience → mode → CLI overrides

    ``retriever`` kwarg name kept for back-compat — it resolves under
    ``config/ttl/tools/``.
    """
    layers: list[DictConfig] = [_load(CONFIG_ROOT / "base.yaml")]
    for group, name in (
        ("benchmark", benchmark),
        ("tools", retriever),
        ("reward", reward),
        ("reflector", reflector),
        ("curator", curator),
        ("experience", experience),
        ("mode", mode),
    ):
        if not name:
            continue
        layers.append(_load(CONFIG_ROOT / group / f"{name}.yaml"))

    cfg = OmegaConf.merge(*layers)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    OmegaConf.resolve(cfg)
    return cfg


def validate(cfg: DictConfig) -> None:
    """Fail fast on common misconfigurations.

    Validates:
      - registry kinds resolve (deferred to ``build_*`` calls)
      - per-step edit budgets are consistent
      - data files exist
      - env-resolved fields are non-empty
      - actor mini/micro batch divisibility
    """
    c = cfg.curator.conservativeness
    if int(c.max_total_edits_per_step) < max(
        int(c.max_add_per_step), int(c.max_modify_per_step), int(c.max_delete_per_step)
    ):
        raise ValueError(
            "curator.conservativeness.max_total_edits_per_step must be ≥ max of the per-op caps"
        )

    # ── batch sizing constraints ───────────────────────────────
    G = int(cfg.rollout.n)
    micro = int(cfg.actor.micro_batch_size)
    mini = int(cfg.actor.mini_batch_size)
    train_bs = int(cfg.data.train_batch_size)

    if G <= 0 or micro <= 0 or mini <= 0 or train_bs <= 0:
        raise ValueError(
            "data.train_batch_size, rollout.n, actor.mini_batch_size, "
            "actor.micro_batch_size must all be positive"
        )
    if micro % G != 0:
        raise ValueError(
            f"actor.micro_batch_size ({micro}) must be a multiple of "
            f"rollout.n ({G}) so that reward+reflector chunks land on task "
            "boundaries (reflector forms peer summaries within one task's "
            "G trajectories)."
        )
    if mini % micro != 0:
        raise ValueError(
            f"actor.mini_batch_size ({mini}) must be a multiple of "
            f"actor.micro_batch_size ({micro})."
        )
    total_per_step = train_bs * G
    if mini > total_per_step:
        raise ValueError(
            f"actor.mini_batch_size ({mini}) cannot exceed "
            f"data.train_batch_size × rollout.n ({total_per_step}); "
            "lower mini_batch or raise train_batch_size."
        )

    if int(cfg.rollout.concurrency) <= 0:
        raise ValueError("rollout.concurrency must be positive")

    # ── data files ────────────────────────────────────────────────────────
    for split_name in ("train", "val", "test"):
        path = cfg.data.splits.get(split_name)
        if path and not Path(path).exists():
            raise FileNotFoundError(f"data.splits.{split_name} = {path} does not exist")
    if not Path(cfg.data.decrypted_jsonl).exists():
        raise FileNotFoundError(
            f"data.decrypted_jsonl = {cfg.data.decrypted_jsonl} does not exist"
        )

    # ── env-resolved fields ───────────────────────────────────────────────
    required_fields = ["agent.llm_base_url"]
    tool_provider = str(cfg.agent.get("tool_provider", "mcp"))
    if tool_provider == "mcp":
        required_fields.append("agent.mcp_url")
    elif tool_provider == "serper":
        # Serper key is read at agent build time, but fail fast here
        # so a missing key doesn't surface mid-rollout.
        serper = cfg.agent.get("serper", {}) or {}
        if not serper.get("api_key"):
            raise ValueError(
                "agent.serper.api_key is required when tool_provider=serper "
                "(set SERPER_API_KEY in your environment)"
            )
    for path_field in required_fields:
        section, key = path_field.split(".")
        val = cfg[section].get(key)
        if not val:
            raise ValueError(f"{path_field} is required (set via env or override)")
