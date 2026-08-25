"""CheckpointManager — verl-compatible naming + resume contract.

- ``trainer.save_freq`` (int) — save every N curator steps; 0/-1 disables.
- ``trainer.resume_mode`` ∈ {auto, disable, resume_path}
- ``trainer.resume_from_path`` — explicit ``global_step_<N>`` folder.

Each checkpoint folder contains:
    experience.md
    sampler_state.json
    trainer_state.json
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LOG = logging.getLogger(__name__)


_CKPT_NAME_RE = re.compile(r"^global_step_(\d+)$")


@dataclass
class ResumeBundle:
    global_step: int
    experience_text: str
    sampler_state: dict
    trainer_state: dict
    config_hash: str
    path: Path


class CheckpointManager:
    def __init__(
        self,
        save_root: Path,
        *,
        save_freq: int = 5,
        keep_last_n: int = -1,
    ) -> None:
        self.save_root = Path(save_root)
        self.save_freq = int(save_freq)
        self.keep_last_n = int(keep_last_n)

    def _ckpt_dir(self, step: int) -> Path:
        return self.save_root / "checkpoints" / f"global_step_{step}"

    def maybe_save(
        self,
        opt_step: int,
        experience_text: str,
        sampler_state: dict,
        trainer_state: dict,
        *,
        is_last_step: bool = False,
    ) -> Path | None:
        if self.save_freq <= 0:
            if not is_last_step:
                return None
        elif not is_last_step and opt_step % self.save_freq != 0:
            return None

        d = self._ckpt_dir(opt_step)
        d.mkdir(parents=True, exist_ok=True)
        (d / "experience.md").write_text(experience_text, encoding="utf-8")
        (d / "sampler_state.json").write_text(
            json.dumps(sampler_state, indent=2), encoding="utf-8"
        )
        (d / "trainer_state.json").write_text(
            json.dumps(trainer_state, indent=2), encoding="utf-8"
        )
        try:
            from search_ttl.utils.logging import good

            good(f"checkpoint saved → {d}")
        except Exception:  # noqa: BLE001
            LOG.info("[ckpt] saved global_step_%d → %s", opt_step, d)
        if self.keep_last_n > 0:
            self._gc()
        return d

    def maybe_resume(
        self, mode: str, resume_from_path: str | None
    ) -> ResumeBundle | None:
        mode = (mode or "auto").lower()
        if mode == "disable":
            return None

        if mode == "resume_path":
            if not resume_from_path:
                raise ValueError(
                    "trainer.resume_mode=resume_path requires trainer.resume_from_path"
                )
            target = Path(resume_from_path)
            if not target.is_absolute():
                target = Path.cwd() / target
        elif mode == "auto":
            target = self._find_latest()
            if target is None:
                LOG.info("[ckpt] no prior checkpoint; starting from scratch")
                return None
        else:
            raise ValueError(f"unknown resume_mode: {mode!r}")

        if not target.exists():
            LOG.warning("[ckpt] resume target missing: %s", target)
            return None
        return self._load(target)

    def _find_latest(self) -> Path | None:
        root = self.save_root / "checkpoints"
        if not root.is_dir():
            return None
        candidates: list[tuple[int, Path]] = []
        for child in root.iterdir():
            m = _CKPT_NAME_RE.match(child.name)
            if m and child.is_dir():
                candidates.append((int(m.group(1)), child))
        if not candidates:
            return None
        candidates.sort(key=lambda p: p[0])
        return candidates[-1][1]

    def _load(self, target: Path) -> ResumeBundle:
        m = _CKPT_NAME_RE.match(target.name)
        if not m:
            raise ValueError(
                f"checkpoint folder must be named global_step_<N>; got {target.name!r}"
            )
        step = int(m.group(1))
        experience = (target / "experience.md").read_text(encoding="utf-8")
        sampler_state = json.loads((target / "sampler_state.json").read_text(encoding="utf-8"))
        trainer_state = json.loads((target / "trainer_state.json").read_text(encoding="utf-8"))
        return ResumeBundle(
            global_step=step,
            experience_text=experience,
            sampler_state=sampler_state,
            trainer_state=trainer_state,
            config_hash=trainer_state.get("config_hash", ""),
            path=target,
        )

    def _gc(self) -> None:
        root = self.save_root / "checkpoints"
        if not root.is_dir():
            return
        kept: list[tuple[int, Path]] = []
        for child in root.iterdir():
            m = _CKPT_NAME_RE.match(child.name)
            if m:
                kept.append((int(m.group(1)), child))
        kept.sort(key=lambda p: p[0])
        while len(kept) > self.keep_last_n:
            _, p = kept.pop(0)
            shutil.rmtree(p, ignore_errors=True)
            LOG.info("[ckpt] gc removed %s", p)
