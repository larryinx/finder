"""Pin the shared ``utils/llm_builder.build_chat_llm`` contract and
verify all four ReAct entry points (agent, both reflectors, curator)
expose ``llm_api_flavor`` + ``preserve_thinking``.

The historical bug this guards against: the reflector and curator
defaulted to plain ``ChatOpenAI``, so flipping ``enable_thinking: true``
in their YAML silently produced a multi-turn ReAct loop that forgot
its own reasoning at every turn — only the agent had HiExp's
``ReasoningPreservingChatOpenAI`` wired up.
"""

from __future__ import annotations

from pathlib import Path

from search_ttl.utils.llm_builder import build_chat_llm


# ── build_chat_llm contract ───────────────────────────────────────────────


def test_chat_completions_flavor_returns_reasoning_preserving_client():
    """The default flavor wraps with HiExp's ``ReasoningPreservingChatOpenAI``
    so reasoning_content is re-injected on every request."""
    from hiexp.utils.reasoning_chat_openai import ReasoningPreservingChatOpenAI

    llm = build_chat_llm(
        model="x", base_url="http://x", api_key="x",
        llm_api_flavor="chat_completions",
        enable_thinking=True, preserve_thinking=True,
    )
    assert isinstance(llm, ReasoningPreservingChatOpenAI)
    # The flag that actually re-injects reasoning on the next request.
    assert llm.inject_reasoning_on_request is True


def test_chat_completions_thinking_off_disables_reasoning_injection():
    """``preserve_thinking=True`` alone is not enough — ``enable_thinking``
    must also be on, otherwise the request-side injection is pointless
    (there's no reasoning_content to inject)."""
    from hiexp.utils.reasoning_chat_openai import ReasoningPreservingChatOpenAI

    llm = build_chat_llm(
        model="x", base_url="http://x", api_key="x",
        enable_thinking=False, preserve_thinking=True,
    )
    assert isinstance(llm, ReasoningPreservingChatOpenAI)
    assert llm.inject_reasoning_on_request is False


def test_chat_completions_preserve_off_disables_injection_even_with_thinking():
    from hiexp.utils.reasoning_chat_openai import ReasoningPreservingChatOpenAI

    llm = build_chat_llm(
        model="x", base_url="http://x", api_key="x",
        enable_thinking=True, preserve_thinking=False,
    )
    assert isinstance(llm, ReasoningPreservingChatOpenAI)
    assert llm.inject_reasoning_on_request is False


def test_responses_flavor_uses_plain_chat_openai_no_wrapper():
    """Responses API handles reasoning natively via
    ``reasoning={'summary': 'auto'}``; we don't wrap it ourselves."""
    from hiexp.utils.reasoning_chat_openai import ReasoningPreservingChatOpenAI
    from langchain_openai import ChatOpenAI

    llm = build_chat_llm(
        model="x", base_url="http://x", api_key="x",
        llm_api_flavor="responses",
        enable_thinking=True, preserve_thinking=True,
    )
    assert isinstance(llm, ChatOpenAI)
    assert not isinstance(llm, ReasoningPreservingChatOpenAI)


def test_enable_thinking_appears_in_extra_body_chat_template_kwargs():
    """Whether thinking is on/off must reach the server through the
    chat template kwargs (not just be stashed in the wrapper)."""
    llm = build_chat_llm(
        model="x", base_url="http://x", api_key="x",
        enable_thinking=True, preserve_thinking=True,
    )
    extra = llm.extra_body or {}
    ctk = extra.get("chat_template_kwargs") or {}
    assert ctk.get("enable_thinking") is True
    # preserve_thinking is also forwarded so the chat template can
    # mark prior turns' reasoning for re-injection.
    assert ctk.get("preserve_thinking") is True


def test_preserve_thinking_NOT_forwarded_when_thinking_off():
    """Avoid sending ``preserve_thinking=True`` to the server when
    thinking is off — there's no reasoning to preserve and some servers
    reject unknown chat_template_kwargs combinations."""
    llm = build_chat_llm(
        model="x", base_url="http://x", api_key="x",
        enable_thinking=False, preserve_thinking=True,
    )
    ctk = (llm.extra_body or {}).get("chat_template_kwargs") or {}
    assert ctk.get("enable_thinking") is False
    assert "preserve_thinking" not in ctk


# ── from_omegaconf wiring across the four entry points ───────────────────


def _omegaconf(d: dict):
    from omegaconf import OmegaConf
    return OmegaConf.create(d)


def test_reflector_querychain_staged_reads_llm_api_flavor_and_preserve_thinking():
    from search_ttl.reflector.querychain_staged import QuerychainStagedReflector

    cfg = _omegaconf({
        "kind": "querychain_staged",
        "llm_model": "m", "llm_base_url": "http://x", "llm_api_key": "k",
        "mcp_url": "http://mcp",
        "llm_api_flavor": "responses",
        "enable_thinking": True,
        "preserve_thinking": False,
    })
    r = QuerychainStagedReflector.from_omegaconf(cfg)
    assert r.cfg.llm_api_flavor == "responses"
    assert r.cfg.enable_thinking is True
    assert r.cfg.preserve_thinking is False


def test_reflector_querychain_tactic_reads_llm_api_flavor_and_preserve_thinking():
    from search_ttl.reflector.querychain_tactic import QuerychainTacticReflector

    cfg = _omegaconf({
        "kind": "querychain_tactic",
        "llm_model": "m", "llm_base_url": "http://x", "llm_api_key": "k",
        "mcp_url": "http://mcp",
        "llm_api_flavor": "chat_completions",
        "enable_thinking": True,
        "preserve_thinking": True,
    })
    r = QuerychainTacticReflector.from_omegaconf(cfg)
    assert r.cfg.llm_api_flavor == "chat_completions"
    assert r.cfg.enable_thinking is True
    assert r.cfg.preserve_thinking is True


def test_curator_react_staged_reads_llm_api_flavor_and_preserve_thinking():
    from search_ttl.curator.react_staged import ReActStagedCurator

    cfg = _omegaconf({
        "kind": "react_staged",
        "llm_model": "m", "llm_base_url": "http://x", "llm_api_key": "k",
        "llm_api_flavor": "chat_completions",
        "enable_thinking": True,
        "preserve_thinking": True,
    })
    o = ReActStagedCurator.from_omegaconf(cfg)
    assert o.cfg.llm_api_flavor == "chat_completions"
    assert o.cfg.enable_thinking is True
    assert o.cfg.preserve_thinking is True


# ── YAML defaults expose the two knobs ───────────────────────────────────


def test_openrouter_provider_returns_chat_openrouter():
    """``llm_provider='openrouter'`` returns ``ChatOpenRouter`` and
    does NOT wrap with HiExp's reasoning-preserving client (the
    package handles the round-trip natively)."""
    from hiexp.utils.reasoning_chat_openai import ReasoningPreservingChatOpenAI
    from langchain_openrouter import ChatOpenRouter

    llm = build_chat_llm(
        model="anthropic/claude-3.5-sonnet",
        base_url="https://openrouter.ai/api/v1",
        api_key="sk-x",
        llm_provider="openrouter",
        enable_thinking=False,
    )
    assert isinstance(llm, ChatOpenRouter)
    assert not isinstance(llm, ReasoningPreservingChatOpenAI)


def test_openrouter_provider_with_thinking_sets_reasoning_summary_auto():
    """When ``enable_thinking=True`` on the openrouter branch, the
    ``reasoning={'summary': 'auto'}`` constructor kwarg is set; the
    package then asks upstream models for a reasoning summary."""
    llm = build_chat_llm(
        model="anthropic/claude-3.5-sonnet",
        base_url="https://openrouter.ai/api/v1",
        api_key="sk-x",
        llm_provider="openrouter",
        enable_thinking=True,
    )
    # ChatOpenRouter stores the constructor reasoning hint on a known
    # attribute. The package may rename it across versions — we accept
    # either ``reasoning`` or a key inside ``model_kwargs``.
    rv = getattr(llm, "reasoning", None) or (getattr(llm, "model_kwargs", {}) or {}).get("reasoning")
    assert rv == {"summary": "auto"}


def test_openrouter_provider_ignores_preserve_thinking_and_api_flavor():
    """``preserve_thinking`` and ``llm_api_flavor`` are no-ops on the
    openrouter path — assert by smoke-instantiating with the "bad"
    combo and confirming it still builds a ChatOpenRouter (no crash,
    no fallthrough to the openai_compat branches)."""
    from langchain_openrouter import ChatOpenRouter

    llm = build_chat_llm(
        model="x", base_url="https://openrouter.ai/api/v1", api_key="sk-x",
        llm_provider="openrouter",
        llm_api_flavor="responses",     # ignored on this provider
        enable_thinking=True,
        preserve_thinking=True,          # ignored on this provider
    )
    assert isinstance(llm, ChatOpenRouter)


def test_default_provider_is_openai_compat():
    """No explicit ``llm_provider`` argument → openai_compat path."""
    from hiexp.utils.reasoning_chat_openai import ReasoningPreservingChatOpenAI

    llm = build_chat_llm(
        model="x", base_url="http://x", api_key="x",
        enable_thinking=False,
    )
    assert isinstance(llm, ReasoningPreservingChatOpenAI)


# ── from_omegaconf wiring across the four entry points (provider) ────────


def test_search_agent_config_reads_llm_provider():
    from search_ttl.agent.search_agent import SearchAgentConfig

    cfg = _omegaconf({
        "llm_model": "m", "llm_base_url": "http://x", "llm_api_key": "k",
        "llm_provider": "openrouter",
    })
    a = SearchAgentConfig.from_omegaconf(cfg)
    assert a.llm_provider == "openrouter"


def test_search_agent_config_defaults_provider_to_openai_compat():
    from search_ttl.agent.search_agent import SearchAgentConfig

    cfg = _omegaconf({
        "llm_model": "m", "llm_base_url": "http://x", "llm_api_key": "k",
    })
    a = SearchAgentConfig.from_omegaconf(cfg)
    assert a.llm_provider == "openai_compat"


def test_reflectors_and_curators_read_llm_provider():
    from search_ttl.curator.react_staged import ReActStagedCurator
    from search_ttl.curator.react_tactic import ReActTacticCurator
    from search_ttl.reflector.querychain_staged import QuerychainStagedReflector
    from search_ttl.reflector.querychain_tactic import QuerychainTacticReflector

    base = {
        "llm_model": "m", "llm_base_url": "http://x", "llm_api_key": "k",
        "llm_provider": "openrouter",
    }
    managers = [
        QuerychainTacticReflector.from_omegaconf(_omegaconf(
            {**base, "kind": "querychain_tactic", "mcp_url": "http://mcp"}
        )),
        QuerychainStagedReflector.from_omegaconf(_omegaconf(
            {**base, "kind": "querychain_staged", "mcp_url": "http://mcp"}
        )),
        ReActTacticCurator.from_omegaconf(_omegaconf({**base, "kind": "react_tactic"})),
        ReActStagedCurator.from_omegaconf(_omegaconf({**base, "kind": "react_staged"})),
    ]
    for m in managers:
        assert m.cfg.llm_provider == "openrouter"


def test_reward_judge_config_reads_llm_provider():
    from search_ttl.reward.search_agent_lasj import SearchAgentRewardLASJ

    cfg = _omegaconf({
        "kind": "search_agent_lasj",
        "judge": {"model": "m", "base_url": "http://x", "api_key": "k",
                  "llm_provider": "openrouter"},
    })
    r = SearchAgentRewardLASJ.from_omegaconf(cfg)
    assert r.cfg.judge.llm_provider == "openrouter"


def test_reward_judge_defaults_to_openai_compat():
    from search_ttl.reward.search_agent_lasj import SearchAgentRewardLASJ

    cfg = _omegaconf({
        "kind": "search_agent_lasj",
        "judge": {"model": "m", "base_url": "http://x", "api_key": "k"},
    })
    r = SearchAgentRewardLASJ.from_omegaconf(cfg)
    assert r.cfg.judge.llm_provider == "openai_compat"


def test_judge_llm_lazy_routes_through_build_chat_llm(monkeypatch):
    """The Judge should produce a ChatOpenRouter (not a plain
    ChatOpenAI / ReasoningPreservingChatOpenAI) when ``llm_provider=
    'openrouter'`` — confirms the build site uses the shared helper."""
    from langchain_openrouter import ChatOpenRouter

    from search_ttl.reward.judge import Judge, JudgeConfig

    j = Judge(JudgeConfig(
        model="anthropic/claude-3.5-sonnet",
        base_url="https://openrouter.ai/api/v1",
        api_key="sk-x",
        llm_provider="openrouter",
    ))
    assert isinstance(j._llm_lazy(), ChatOpenRouter)


def test_env_var_AGENT_LLM_PROVIDER_drives_reward_judge(monkeypatch):
    """Setting one env var should also flip the reward's judge to
    openrouter (fallback chain ends at AGENT_LLM_PROVIDER)."""
    from search_ttl.config.compose import compose

    monkeypatch.setenv("AGENT_LLM_PROVIDER", "openrouter")
    cfg = compose(overrides=["trainer.save_path=/tmp/_x"], mode="smoke",
                  reward="search_agent_lasj")
    assert cfg.reward.judge.llm_provider == "openrouter"


def test_component_specific_env_var_overrides_reward_provider(monkeypatch):
    """``EVAL_JUDGE_PROVIDER`` wins over ``AGENT_LLM_PROVIDER`` so the
    reward can stay on a different provider than the rollout agent
    (useful when grading on a stronger judge model that lives on a
    different endpoint)."""
    from search_ttl.config.compose import compose

    monkeypatch.setenv("AGENT_LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("EVAL_JUDGE_PROVIDER", "openai_compat")
    cfg = compose(overrides=["trainer.save_path=/tmp/_x"], mode="smoke",
                  reward="search_agent_lasj")
    assert cfg.reward.judge.llm_provider == "openai_compat"
    # Agent + downstream loops still on openrouter via AGENT_LLM_PROVIDER.
    assert cfg.agent.llm_provider == "openrouter"


def test_yaml_configs_expose_thinking_knobs():
    """Lock the YAMLs against silent regression — every config that
    spins a ReAct loop must expose all four knobs (provider + flavor
    + thinking pair). Includes ``base.yaml`` for the agent."""
    paths = [
        "config/ttl/base.yaml",
        "config/ttl/reflector/querychain_tactic.yaml",
        "config/ttl/reflector/querychain_staged.yaml",
        "config/ttl/curator/react_tactic.yaml",
        "config/ttl/curator/react_staged.yaml",
    ]
    for p in paths:
        text = Path(p).read_text()
        assert "llm_provider:" in text, f"{p} missing llm_provider"
        assert "llm_api_flavor:" in text, f"{p} missing llm_api_flavor"
        # base.yaml's agent block doesn't expose preserve_thinking
        # via the explicit key (it lives in code defaults); the others
        # do.
        if "base.yaml" not in p:
            assert "preserve_thinking:" in text, f"{p} missing preserve_thinking"


def test_env_var_AGENT_LLM_PROVIDER_drives_agent_yaml(monkeypatch):
    """``AGENT_LLM_PROVIDER=openrouter`` flows through ``base.yaml``
    into ``cfg.agent.llm_provider``."""
    from search_ttl.config.compose import compose

    monkeypatch.setenv("AGENT_LLM_PROVIDER", "openrouter")
    cfg = compose(overrides=["trainer.save_path=/tmp/_x"], mode="smoke")
    assert cfg.agent.llm_provider == "openrouter"


def test_env_var_AGENT_LLM_PROVIDER_also_drives_reflector_and_curator(monkeypatch):
    """The reflector + curator YAMLs fall back to ``AGENT_LLM_PROVIDER``
    so a single env var moves all four ReAct loops together."""
    from search_ttl.config.compose import compose

    monkeypatch.setenv("AGENT_LLM_PROVIDER", "openrouter")
    cfg = compose(overrides=["trainer.save_path=/tmp/_x"], mode="smoke")
    assert cfg.reflector.llm_provider == "openrouter"
    assert cfg.curator.llm_provider == "openrouter"


def test_component_specific_env_var_overrides_agent_provider(monkeypatch):
    """``REFLECTOR_LLM_PROVIDER`` wins over ``AGENT_LLM_PROVIDER`` for
    the reflector — letting users mix providers (e.g., OSS rollout +
    openrouter reflector)."""
    from search_ttl.config.compose import compose

    monkeypatch.setenv("AGENT_LLM_PROVIDER", "openai_compat")
    monkeypatch.setenv("REFLECTOR_LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("CURATOR_LLM_PROVIDER", "openrouter")
    cfg = compose(overrides=["trainer.save_path=/tmp/_x"], mode="smoke")
    assert cfg.agent.llm_provider == "openai_compat"
    assert cfg.reflector.llm_provider == "openrouter"
    assert cfg.curator.llm_provider == "openrouter"
