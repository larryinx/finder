"""Compose run configurations from YAML + CLI overrides.

The config tree under ``config/`` is split into four axes:

* ``base.yaml``           — invariants for the whole agent family
* ``benchmark/<name>.yaml`` — per-benchmark deltas (data paths, search
  budgets, scorer kind, question template)
* ``retriever/<name>.yaml`` — per-retriever metadata used mostly for
  output-dir templating
* ``mode/<name>.yaml``    — ACE-only; offline vs online controller knobs

A run picks one entry per axis. This module merges them in a fixed
priority (later wins) and applies CLI dotlist overrides on top:

    base < benchmark < retriever < mode < CLI

Resolution is deliberately simple: no defaults-list, no inheritance
chains, no Hydra. Two reasons: (a) the axes are small and orthogonal,
(b) shell wrappers are easier to read when the composition is explicit
in the CLI flags.

Endpoint settings (``AGENT_LLM_MODEL`` / ``AGENT_LLM_BASE_URL`` /
``AGENT_LLM_API_KEY`` / ``AGENT_MCP_URL`` / optional ``EVAL_JUDGE_*``)
stay in ``.env`` — they're per-machine and shouldn't be checked into
config YAMLs that get shared across hosts. See
:meth:`HiExpConfig.from_omegaconf`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Literal

from omegaconf import DictConfig, OmegaConf

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = _PROJECT_ROOT / "config"

Family = Literal["ace", "hiexp_prompt"]


def compose(
    *,
    family: Family,
    benchmark: str,
    retriever: str | None = None,
    mode: str | None = None,
    overrides: Iterable[str] | None = None,
) -> DictConfig:
    """Merge ``base + benchmark + retriever + mode`` for ``family``.

    Args:
        family: ``"ace"`` or ``"hiexp_prompt"``. Picks the subdir.
        benchmark: stem under ``config/<family>/benchmark/``.
        retriever: stem under ``config/<family>/retriever/`` (optional;
            required when the run hits an MCP retriever, which is most of
            them — but skippable for tests / dry-runs).
        mode: stem under ``config/<family>/mode/`` (ACE only).
        overrides: dotlist (``a.b.c=value``) applied last.

    Returns:
        A merged :class:`DictConfig`. Missing keys raise on access.
    """
    root = CONFIG_ROOT / family
    cfg = OmegaConf.load(root / "base.yaml")
    cfg = OmegaConf.merge(cfg, OmegaConf.load(root / "benchmark" / f"{benchmark}.yaml"))
    if retriever is not None:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(root / "retriever" / f"{retriever}.yaml"))
    if mode is not None:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(root / "mode" / f"{mode}.yaml"))
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    return cfg  # type: ignore[return-value]


def list_options(family: Family, axis: Literal["benchmark", "retriever", "mode"]) -> list[str]:
    """Stems of available YAMLs under ``config/<family>/<axis>/`` (no suffix)."""
    d = CONFIG_ROOT / family / axis
    if not d.exists():
        return []
    return sorted(p.stem for p in d.glob("*.yaml"))


__all__ = ["compose", "list_options", "CONFIG_ROOT", "Family"]
