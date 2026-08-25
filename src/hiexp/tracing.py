"""Weave/W&B observability for HiExp + ACE.

This module is the single entry point for tracing. Every CLI script
calls :func:`init_weave` near the top of ``main()``; idempotent, so
repeated calls (across phase scripts spawned by build_hek's subprocess
fan-out) are safe.

Environment variables read here:

================================  ===================================================
``WANDB_API_KEY``                 W&B authentication. Required to log anything.
``WANDB_ENTITY``                  W&B entity (team/user). Standard W&B convention.
``WANDB_TEAM``                    Alias for ``WANDB_ENTITY`` if the latter isn't set.
``WANDB_PROJECT``                 W&B project. ``weave.init`` uses ``entity/project``.
``WEAVE_DISABLED``                Native weave gate. When ``true`` / ``1``, no traces
                                  go out (op decorators become no-ops). Also
                                  consulted here so we skip the wandb run too.
``WEAVE_CAPTURE_CODE``            Native weave gate. When ``false`` / ``0``, op
                                  source code isn't sent to W&B. Read by weave.
``WEAVE_DISABLED_<PHASE>``        Per-phase override of the global gate (uppercase
                                  phase name). Set e.g.
                                  ``WEAVE_DISABLED_HIEXP_ROLLOUT=1`` to skip
                                  ``sample_rollouts.py`` only. The orchestrator,
                                  eval, and ACE phases stay traced. Defined
                                  phases: ``hiexp_rollout`` /
                                  ``hiexp_eval`` / ``hek_build`` /
                                  ``ace_offline`` / ``ace_online``.
================================  ===================================================

Three things we expose:

1. :func:`init_weave` — idempotent ``weave.init`` + ``wandb.init`` (the
   latter is needed to log Artifacts; weave alone gives us the Trace UI
   but artifact uploads need a wandb run handle).
2. :func:`is_enabled` — runtime check (``WEAVE_DISABLED`` not set AND
   we successfully initialized).
3. :func:`log_artifact` / :func:`log_playbook_checkpoint` /
   :func:`log_hek_artifacts` — helpers for uploading the run-level
   artifacts the docs describe (HEK e1/e2 jsonl, ACE playbook
   checkpoints).

Auto-instrumentation: once ``weave.init`` is called, weave's patcher
hooks into LangChain / OpenAI / litellm SDKs automatically, so every
LLM round-trip emits a trace without any decorator on our side. We
add explicit ``@weave.op`` decorators in a few orchestration places
(``HiExpAgent.arun``, ``ReActGenerator.generate``, ...) so the trace
hierarchy in the UI groups the sub-calls under a meaningful parent.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# Module-level state — single weave client + wandb run per process.
_INITIALIZED: bool = False
_WANDB_RUN: Any = None  # wandb.sdk.wandb_run.Run | None
_WEAVE_CLIENT: Any = None


def _is_truthy(s: str | None) -> bool:
    if not s:
        return False
    return s.strip().lower() in {"1", "true", "yes", "y", "on"}


def _resolve_entity() -> str | None:
    """``WANDB_ENTITY`` (standard) takes precedence; ``WANDB_TEAM`` is an alias."""
    return os.getenv("WANDB_ENTITY") or os.getenv("WANDB_TEAM")


def is_disabled(phase: str | None = None) -> bool:
    """Whether tracing is gated off.

    Two layers, OR'd together:

    1. Global: ``WEAVE_DISABLED`` truthy disables everything.
    2. Per-phase: when ``phase`` is given (e.g. ``"hiexp_rollout"``),
       also check ``WEAVE_DISABLED_HIEXP_ROLLOUT``. This lets you
       disable a noisy phase (rollouts) while keeping the surrounding
       run alive (the orchestrator that uploads the artifact at the
       end of HEK build).
    """
    if _is_truthy(os.getenv("WEAVE_DISABLED")):
        return True
    if phase:
        if _is_truthy(os.getenv(f"WEAVE_DISABLED_{phase.upper()}")):
            return True
    return False


def is_enabled() -> bool:
    """Whether tracing is initialized and active."""
    return _INITIALIZED and not is_disabled()


def init_weave(
    *,
    run_name: str | None = None,
    config: dict[str, Any] | None = None,
    tags: list[str] | None = None,
    job_type: str | None = None,
    phase: str | None = None,
) -> Any:
    """Idempotently initialize weave + wandb for the current process.

    Args:
        run_name: passed to ``wandb.init(name=...)``.
        config: passed to ``wandb.init(config=...)``. Stash the YAML
            cfg here (``OmegaConf.to_container(cfg, resolve=True)``).
        tags: passed to ``wandb.init(tags=...)``.
        job_type: passed to ``wandb.init(job_type=...)``.
        phase: per-phase gate name, lowercase. Resolves to
            ``WEAVE_DISABLED_<PHASE>`` env var (uppercase). Set this
            in each entry point (``hiexp_rollout`` /
            ``hiexp_eval`` / ``ace_offline`` / ``ace_online`` /
            ``hek_build``) so users can selectively disable noisy
            phases. Defaults to None (only the global gate applies).

    Returns:
        The weave client, or ``None`` when the gate is on or W&B isn't
        configured.
    """
    global _INITIALIZED, _WANDB_RUN, _WEAVE_CLIENT

    if _INITIALIZED:
        return _WEAVE_CLIENT

    if is_disabled(phase):
        gate = (
            f"WEAVE_DISABLED_{phase.upper()}"
            if phase and _is_truthy(os.getenv(f"WEAVE_DISABLED_{phase.upper()}"))
            else "WEAVE_DISABLED"
        )
        logger.info("[tracing] %s set — skipping init for phase=%s", gate, phase)
        _INITIALIZED = True  # mark so we don't keep re-checking
        return None

    entity = _resolve_entity()
    project = os.getenv("WANDB_PROJECT")
    api_key = os.getenv("WANDB_API_KEY")
    if not entity or not project:
        logger.warning(
            "[tracing] WANDB_ENTITY/WANDB_TEAM and WANDB_PROJECT must both be set "
            "to enable Weave tracing — proceeding without tracing."
        )
        _INITIALIZED = True
        return None
    if not api_key:
        logger.warning(
            "[tracing] WANDB_API_KEY not set — proceeding without tracing."
        )
        _INITIALIZED = True
        return None

    # Initialize wandb first so weave traces are linked to a run handle
    # we can log artifacts against. ``weave.init`` doesn't create a
    # wandb.Run; we want both for artifact upload.
    import wandb  # imported lazily to avoid pulling wandb when disabled

    try:
        _WANDB_RUN = wandb.init(
            entity=entity,
            project=project,
            name=run_name,
            config=dict(config or {}),
            tags=list(tags or []),
            job_type=job_type,
            reinit=False,
            resume="allow",
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("[tracing] wandb.init failed (%s) — running without tracing", exc)
        _INITIALIZED = True
        return None

    import weave

    try:
        _WEAVE_CLIENT = weave.init(f"{entity}/{project}")
    except Exception as exc:  # noqa: BLE001
        logger.warning("[tracing] weave.init failed (%s) — wandb run still active", exc)
        _WEAVE_CLIENT = None

    _INITIALIZED = True
    logger.info(
        "[tracing] initialized weave=%s wandb_run=%s",
        bool(_WEAVE_CLIENT),
        getattr(_WANDB_RUN, "name", None),
    )
    return _WEAVE_CLIENT


def get_run() -> Any:
    """Return the active ``wandb.Run`` (or ``None``)."""
    return _WANDB_RUN


def get_client() -> Any:
    """Return the active weave client (or ``None``)."""
    return _WEAVE_CLIENT


def log_artifact(
    name: str,
    *,
    type: str,
    files: list[str | Path] | None = None,
    metadata: dict[str, Any] | None = None,
    aliases: list[str] | None = None,
) -> Any:
    """Upload one or more files as a versioned ``wandb.Artifact``.

    No-op when tracing is off.

    Args:
        name: artifact name (e.g. ``"hek-r8b"``, ``"ace-playbook"``).
            Uniqueness is per (entity, project, name).
        type: artifact type (e.g. ``"dataset"``, ``"ace-playbook"``).
            Used for filtering in the UI.
        files: paths to add. Directories are added recursively.
        metadata: arbitrary dict. Useful for stashing the spec / config.
        aliases: extra tags besides the auto ``"latest"`` and version
            number — e.g. ``["step-100"]`` for a checkpoint.

    Returns:
        The logged ``wandb.Artifact``, or ``None``.
    """
    if not is_enabled() or _WANDB_RUN is None:
        return None
    import wandb

    art = wandb.Artifact(name=name, type=type, metadata=metadata or {})
    for f in files or []:
        p = Path(f)
        if not p.exists():
            logger.warning("[tracing] artifact file missing: %s", p)
            continue
        if p.is_dir():
            art.add_dir(str(p))
        else:
            art.add_file(str(p))
    return _WANDB_RUN.log_artifact(art, aliases=aliases or [])


def log_hek_artifacts(
    hek_path: str | Path,
    *,
    name_prefix: str = "hek",
    metadata: dict[str, Any] | None = None,
) -> Any:
    """Upload the e1/e2/(e3) JSONLs from a build_hek output directory.

    Files included if they exist:
      * ``e1.jsonl``
      * ``e2.jsonl``
      * ``e3.jsonl``
      * ``manifest.json`` (if your build_hek writes one)

    The ``rollouts/`` subdir is intentionally **not** included — it can
    be multi-GB. Add it manually if you need it.

    Args:
        hek_path: directory written by ``build_hek.py``.
        name_prefix: artifact name will be ``f"{prefix}-{hek_path.name}"``.
        metadata: extra metadata; counts of e1/e2/e3 entries are added
            automatically.
    """
    if not is_enabled():
        return None
    p = Path(hek_path)
    files: list[Path] = []
    counts: dict[str, int] = {}
    for stem in ("e1", "e2", "e3"):
        f = p / f"{stem}.jsonl"
        if f.exists():
            files.append(f)
            counts[stem] = sum(1 for _ in f.open("r", encoding="utf-8"))
    manifest = p / "manifest.json"
    if manifest.exists():
        files.append(manifest)
    if not files:
        logger.warning("[tracing] no HEK files found under %s — nothing to log", p)
        return None
    md = {**(metadata or {}), "counts": counts, "hek_path": str(p)}
    return log_artifact(
        name=f"{name_prefix}-{p.name}",
        type="hek",
        files=[str(f) for f in files],
        metadata=md,
        aliases=["latest"],
    )


def log_playbook_checkpoint(
    playbook_path: str | Path,
    *,
    step: int | None = None,
    name: str = "ace-playbook",
    metadata: dict[str, Any] | None = None,
) -> Any:
    """Upload one ACE playbook checkpoint as a versioned artifact.

    Each call creates a new version of the same artifact, with an
    optional ``step-N`` alias for traceability. The W&B UI shows
    ``ace-playbook:v0``, ``v1``, ... and the latest gets ``latest``.

    Args:
        playbook_path: file written by ACE's curator (markdown / JSON).
        step: training step / sample index for the checkpoint alias.
        name: artifact name, default ``"ace-playbook"``. Use a
            run-specific name when you want per-run version chains
            instead of one merged chain across runs.
        metadata: extra metadata.
    """
    if not is_enabled():
        return None
    p = Path(playbook_path)
    if not p.exists():
        logger.warning("[tracing] playbook checkpoint missing: %s", p)
        return None
    aliases = [f"step-{step}"] if step is not None else []
    md = {**(metadata or {}), "step": step, "source_path": str(p)}
    return log_artifact(
        name=name,
        type="ace-playbook",
        files=[str(p)],
        metadata=md,
        aliases=aliases,
    )


def finish() -> None:
    """Close the active wandb run (call from script teardown)."""
    global _WANDB_RUN, _WEAVE_CLIENT, _INITIALIZED
    if _WANDB_RUN is not None:
        try:
            _WANDB_RUN.finish()
        except Exception:  # noqa: BLE001
            pass
    _WANDB_RUN = None
    _WEAVE_CLIENT = None
    _INITIALIZED = False


__all__ = [
    "init_weave",
    "is_disabled",
    "is_enabled",
    "get_run",
    "get_client",
    "log_artifact",
    "log_hek_artifacts",
    "log_playbook_checkpoint",
    "finish",
]
