from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from hiexp import HiExpConfig


def test_defaults_match_paper():
    cfg = HiExpConfig()
    # Paper §4.1
    assert cfg.temperature == 1.0
    assert cfg.top_p == 0.95
    assert cfg.max_response_tokens == 8192
    # Paper §A
    assert cfg.hek_embedding_model == "intfloat/multilingual-e5-base"
    assert cfg.e1_threshold == 0.8
    assert cfg.e2_top_k == 5
    assert cfg.seed == 42
    # Paper §4.1 base model
    assert cfg.llm_model == "Qwen/Qwen2.5-7B-Instruct"


def test_validate_requires_mcp_url():
    cfg = HiExpConfig(mcp_url=None)
    with pytest.raises(ValueError, match="AGENT_MCP_URL"):
        cfg.validate()


def test_validate_threshold_range():
    cfg = HiExpConfig(mcp_url="http://x", e1_threshold=1.5)
    with pytest.raises(ValueError, match="threshold"):
        cfg.validate()


def test_validate_hek_path_required_when_enabled():
    cfg = HiExpConfig(mcp_url="http://x", enable_hek=True, hek_path=None)
    with pytest.raises(ValueError, match="hek_path"):
        cfg.validate()


def test_validate_hek_disabled_skips_hek_path():
    cfg = HiExpConfig(mcp_url="http://x", enable_hek=False, hek_path=None)
    cfg.validate()  # should not raise


def test_include_tool_call_template_default_is_true():
    cfg = HiExpConfig()
    assert cfg.include_tool_call_template is True


def test_from_omegaconf_reads_yaml(monkeypatch, tmp_path):
    """YAML drives algorithmic settings; env drives endpoint fields."""
    monkeypatch.setenv("AGENT_LLM_MODEL", "test/model")
    monkeypatch.setenv("AGENT_MCP_URL", "http://localhost:9999/mcp")
    yaml_cfg = OmegaConf.create(
        {
            "llm": {"temperature": 0.5, "enable_thinking": True},
            "agent": {"max_search_limit": 25},
            "hek": {"e1_top_k": 7, "use_e3_instead_of_e2": True},
        }
    )
    cfg = HiExpConfig.from_omegaconf(
        yaml_cfg, dotenv_path=tmp_path / "missing.env"
    )
    assert cfg.temperature == 0.5
    assert cfg.enable_thinking is True
    assert cfg.max_search_limit == 25
    assert cfg.e1_top_k == 7
    assert cfg.use_e3_instead_of_e2 is True
    assert cfg.llm_model == "test/model"
    assert cfg.mcp_url == "http://localhost:9999/mcp"


def test_from_omegaconf_kwargs_beat_yaml(monkeypatch, tmp_path):
    """Explicit kwargs win over YAML and env."""
    monkeypatch.setenv("AGENT_LLM_MODEL", "env/model")
    monkeypatch.setenv("AGENT_MCP_URL", "http://x/mcp")
    yaml_cfg = OmegaConf.create({"llm": {"temperature": 0.5}})
    cfg = HiExpConfig.from_omegaconf(
        yaml_cfg,
        dotenv_path=tmp_path / "missing.env",
        temperature=0.1,
        llm_model="kwarg/model",
    )
    assert cfg.temperature == 0.1
    assert cfg.llm_model == "kwarg/model"


def test_from_omegaconf_empty_cfg_falls_back_to_defaults(monkeypatch, tmp_path):
    """Empty cfg = dataclass defaults for algo + env for endpoints."""
    monkeypatch.setenv("AGENT_MCP_URL", "http://localhost/mcp")
    cfg = HiExpConfig.from_omegaconf(
        OmegaConf.create({}), dotenv_path=tmp_path / "missing.env"
    )
    assert cfg.temperature == 1.0  # dataclass default
    assert cfg.max_search_limit == 10  # dataclass default
    assert cfg.mcp_url == "http://localhost/mcp"  # from env
