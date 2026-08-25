"""CheckpointManager save / resume."""

from pathlib import Path

import pytest

from search_ttl.trainer.checkpoint import CheckpointManager


def test_save_freq_zero_only_saves_last(tmp_path: Path):
    mgr = CheckpointManager(tmp_path, save_freq=0)
    assert mgr.maybe_save(3, "exp", {}, {"global_step": 3}) is None
    saved = mgr.maybe_save(5, "exp", {}, {"global_step": 5}, is_last_step=True)
    assert saved is not None
    assert saved.name == "global_step_5"


def test_save_freq_periodic(tmp_path: Path):
    mgr = CheckpointManager(tmp_path, save_freq=2)
    assert mgr.maybe_save(1, "x", {}, {}) is None
    p = mgr.maybe_save(2, "x", {}, {})
    assert p is not None and p.name == "global_step_2"
    p = mgr.maybe_save(4, "x", {}, {})
    assert p is not None


def test_resume_auto_picks_latest(tmp_path: Path):
    mgr = CheckpointManager(tmp_path, save_freq=1)
    mgr.maybe_save(1, "exp1", {"seed": 1}, {"global_step": 1, "config_hash": "h"})
    mgr.maybe_save(3, "exp3", {"seed": 1}, {"global_step": 3, "config_hash": "h"})
    bundle = mgr.maybe_resume("auto", None)
    assert bundle is not None
    assert bundle.global_step == 3
    assert bundle.experience_text == "exp3"


def test_resume_disable_returns_none(tmp_path: Path):
    mgr = CheckpointManager(tmp_path, save_freq=1)
    mgr.maybe_save(1, "exp1", {}, {})
    assert mgr.maybe_resume("disable", None) is None


def test_resume_path_requires_path(tmp_path: Path):
    mgr = CheckpointManager(tmp_path, save_freq=1)
    with pytest.raises(ValueError):
        mgr.maybe_resume("resume_path", None)


def test_resume_path_explicit(tmp_path: Path):
    mgr = CheckpointManager(tmp_path, save_freq=1)
    p = mgr.maybe_save(7, "explicit", {}, {"global_step": 7})
    bundle = mgr.maybe_resume("resume_path", str(p))
    assert bundle and bundle.global_step == 7


def test_keep_last_n_evicts_older(tmp_path: Path):
    mgr = CheckpointManager(tmp_path, save_freq=1, keep_last_n=2)
    for s in (1, 2, 3, 4):
        mgr.maybe_save(s, "x", {}, {})
    remaining = sorted(p.name for p in (tmp_path / "checkpoints").iterdir())
    assert remaining == ["global_step_3", "global_step_4"]
