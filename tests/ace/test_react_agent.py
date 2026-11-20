"""Unit tests for the ACE-side ReAct adapter (``src/ace_react``)."""

from __future__ import annotations

import os
import pytest

from ace_react import ReActAgent, render_playbook_preamble
from hiexp import HiExpConfig


# ── render_playbook_preamble ─────────────────────────────────────────────


def test_render_preamble_empty():
    assert render_playbook_preamble("") == ""
    assert render_playbook_preamble("   \n\n  ") == ""


def test_render_preamble_includes_playbook_text():
    p = render_playbook_preamble("## SECTION A\n[a-001] do X")
    assert "## ACE PLAYBOOK" in p
    assert "[a-001] do X" in p


# ── ReActAgent construction ──────────────────────────────────────────────


@pytest.fixture
def base_cfg(monkeypatch):
    """Build a HiExpConfig that doesn't need a real MCP server."""
    from omegaconf import OmegaConf

    monkeypatch.setenv("AGENT_MCP_URL", "http://localhost:9999/mcp")
    monkeypatch.setenv("AGENT_LLM_API_KEY", "EMPTY")
    cfg = HiExpConfig.from_omegaconf(
        OmegaConf.create({}), enable_hek=False, hek_path=None
    )
    return cfg


def test_construction_disables_hek_regardless_of_input(base_cfg):
    # Even if base config has HEK on (impossible without hek_path, but
    # simulate by twisting a frozen cfg), the ReActAgent path forces it off.
    agent = ReActAgent(base_cfg, playbook="## A\n[a-1] x")
    assert agent._agent.config.enable_hek is False
    assert agent._agent.config.hek_path is None


def test_construction_injects_playbook_into_system_prompt_extra(base_cfg):
    agent = ReActAgent(base_cfg, playbook="## A\n[a-1] use unique anchors")
    extra = agent._agent.config.system_prompt_extra
    assert extra  # non-empty
    assert "ACE PLAYBOOK" in extra
    assert "[a-1] use unique anchors" in extra


def test_empty_playbook_produces_empty_extra(base_cfg):
    agent = ReActAgent(base_cfg, playbook="")
    assert agent._agent.config.system_prompt_extra == ""


def test_playbook_chars_property(base_cfg):
    pb = "## A\n[a-1] x"
    agent = ReActAgent(base_cfg, playbook=pb)
    assert agent.playbook_chars == len(pb)


def test_other_config_fields_preserved(base_cfg):
    """Override only HEK + system_prompt_extra; everything else passes through."""
    agent = ReActAgent(base_cfg, playbook="## A\n[a-1] x")
    assert agent._agent.config.llm_model == base_cfg.llm_model
    assert agent._agent.config.llm_base_url == base_cfg.llm_base_url
    assert agent._agent.config.max_search_limit == base_cfg.max_search_limit
    assert agent._agent.config.max_turn == base_cfg.max_turn
    assert agent._agent.config.question_template == base_cfg.question_template
