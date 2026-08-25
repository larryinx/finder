"""Config composition + validate() rules."""

import pytest
from omegaconf import OmegaConf

from search_ttl.config import compose, list_options
from search_ttl.config.compose import validate as validate_cfg


def _smoke(**ovr):
    args = ["trainer.save_path=/tmp/_x"] + [f"{k}={v}" for k, v in ovr.items()]
    return compose(mode="smoke", overrides=args)


def test_lists_kinds():
    assert "bcp" in list_options("benchmark")
    assert "smoke" in list_options("mode")


def test_compose_smoke_has_required_fields():
    cfg = _smoke()
    # Defaults match `scripts/ttl/run.py`'s arg defaults.
    assert cfg.curator.kind == "react_staged"
    assert cfg.reflector.kind == "querychain_staged"
    assert cfg.reward.kind == "search_agent_lasj"
    assert cfg.experience.kind == "staged_playbook"
    assert cfg.trainer.total_training_steps == 2
    assert cfg.rollout.n == 2
    assert cfg.data.train_batch_size == 2


def test_overrides_apply_last():
    cfg = _smoke(**{"rollout.n": 4, "data.train_batch_size": 8,
                    "actor.micro_batch_size": 4,
                    "actor.mini_batch_size": 32})
    assert cfg.rollout.n == 4
    assert cfg.data.train_batch_size == 8


def test_default_balanced_config_validates():
    """Top-level base.yaml defaults pass validate (skipping data-file existence)."""
    cfg = compose(overrides=["trainer.save_path=/tmp/_x"])
    # Skip file-existence checks by patching paths to /dev/null-equivalents
    # — we only test the batch-divisibility branch here.
    cfg.data.splits.train = "/tmp/__notreal__.tsv"
    cfg.data.splits.val = "/tmp/__notreal__.tsv"
    cfg.data.splits.test = "/tmp/__notreal__.tsv"
    cfg.data.decrypted_jsonl = "/tmp/__notreal__.jsonl"
    # Only the file-existence check should fire — divisibility passes.
    with pytest.raises(FileNotFoundError):
        validate_cfg(cfg)


def test_micro_must_be_multiple_of_rollout_n():
    cfg = _smoke(**{"rollout.n": 4, "actor.micro_batch_size": 3,
                    "actor.mini_batch_size": 12,
                    "data.train_batch_size": 4})
    cfg.data.splits.train = cfg.data.splits.val = cfg.data.splits.test = (
        "/tmp/__notreal__.tsv"
    )
    cfg.data.decrypted_jsonl = "/tmp/__notreal__.jsonl"
    with pytest.raises(ValueError, match="micro_batch_size"):
        validate_cfg(cfg)


def test_mini_must_be_multiple_of_micro():
    cfg = _smoke(**{"rollout.n": 2, "actor.micro_batch_size": 2,
                    "actor.mini_batch_size": 5,
                    "data.train_batch_size": 4})
    cfg.data.splits.train = cfg.data.splits.val = cfg.data.splits.test = (
        "/tmp/__notreal__.tsv"
    )
    cfg.data.decrypted_jsonl = "/tmp/__notreal__.jsonl"
    with pytest.raises(ValueError, match="mini_batch_size"):
        validate_cfg(cfg)


def test_mini_cannot_exceed_total_per_step():
    cfg = _smoke(**{"rollout.n": 2, "actor.micro_batch_size": 2,
                    "actor.mini_batch_size": 100,
                    "data.train_batch_size": 4})
    cfg.data.splits.train = cfg.data.splits.val = cfg.data.splits.test = (
        "/tmp/__notreal__.tsv"
    )
    cfg.data.decrypted_jsonl = "/tmp/__notreal__.jsonl"
    with pytest.raises(ValueError, match="cannot exceed"):
        validate_cfg(cfg)
