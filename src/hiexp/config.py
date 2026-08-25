"""Configuration for HiExpAgent.

Defaults follow the HiExp paper (Hao et al., 2026, *Beyond Stochastic
Exploration*) when the paper specifies a value, otherwise fall back to:
  - the BrowseComp-Plus reference (third_party/BrowseComp-Plus/) for the
    search-server / MCP / eval-runner shape, since BrowseComp-Plus is the
    final eval target;
  - the Search-o1 reference (related/Search-o1/scripts/run_search_o1.py)
    for agent-loop guards (max rounds, doc budgets) the paper does not
    pin down.

Architectural separation enforced here:
  * The agent's search tool comes from an MCP server (``AGENT_MCP_URL``).
    There is no embedded retriever inside the agent.
  * HEK retrieval (E1/E2 experience injection) is HiExp-internal — its
    own embedding model and store, not exposed via MCP.
  * Answer extraction and metric computation live in ``hiexp.evaluation``
    and are NOT invoked by the agent loop.

Environment variables are loaded via python-dotenv from ``.env``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

MCPTransport = Literal["sse", "streamable-http", "stdio"]
QuestionTemplate = Literal["hiexp_boxed", "browsecomp_plus", "raw"]
LlmApiFlavor = Literal["responses", "chat_completions"]
LlmProvider = Literal["openai_compat", "openrouter"]


def _get_env(name: str, default: str | None = None) -> str | None:
    """Return the env var value, or ``default`` if unset / empty."""
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value


@dataclass
class HiExpConfig:
    """Typed configuration for HiExpAgent.

    Construct directly with keyword arguments, or — recommended — compose
    a YAML stack (``config/hiexp_prompt/*`` or ``config/ace/*``) and pass
    it to :meth:`from_omegaconf`. ``from_omegaconf`` reads algorithmic
    settings from the YAML and pulls endpoint fields (``llm_*``,
    ``mcp_url``, ``hek_path``) from environment variables loaded from
    ``.env``.
    """

    # ── LLM ─────────────────────────────────────────────────────────────────
    llm_model: str = "Qwen/Qwen2.5-7B-Instruct"
    """Paper §4.1: experiments primarily based on Qwen2.5-7B-Instruct."""

    llm_base_url: str = "http://localhost:8000/v1"
    """OpenAI-compatible endpoint. Default points at a local vLLM server."""

    llm_api_key: str = "EMPTY"

    temperature: float = 1.0
    """Paper §4.1: ``sampling temperature of 1.0``."""

    top_p: float = 0.95
    """Paper §4.1: ``top-p of 0.95``."""

    max_response_tokens: int = 8192
    """Paper §4.1: ``maximum response length of 8192``."""

    llm_provider: LlmProvider = "openai_compat"
    """Which LangChain integration to use for the agent LLM.

    - ``openai_compat`` (default): ``ChatOpenAI`` against any OpenAI-
      compatible endpoint (vLLM, SGLang, vanilla OpenAI). The
      :class:`~hiexp.utils.reasoning_chat_openai.ReasoningPreservingChatOpenAI`
      subclass handles reasoning round-trip for SGLang-style servers.
    - ``openrouter``: ``langchain_openrouter.ChatOpenRouter`` against
      ``https://openrouter.ai/api/v1``. Reasoning round-trip is built
      in (the package writes ``reasoning`` / ``reasoning_details``
      fields on outgoing assistant messages), so
      ``preserve_thinking`` and our request-side injection are no-ops
      on this path. ``enable_thinking=True`` maps to
      ``reasoning={"summary": "auto"}`` on the SDK constructor.

    Resolved from ``AGENT_LLM_PROVIDER`` (default ``openai_compat``).
    Auto-detection from ``base_url=openrouter.ai`` is *not* applied —
    set this explicitly so behavior is reproducible.
    """

    llm_api_flavor: LlmApiFlavor = "responses"
    """Which OpenAI-compatible endpoint the agent talks to.

    - ``responses`` (default, paper-faithful): hits ``/v1/responses``,
      reasoning models emit structured ``reasoning`` blocks captured by
      our trajectory layer. Requires vLLM 0.19+ or SGLang ≥0.4 with
      Responses API enabled.
    - ``chat_completions``: hits ``/v1/chat/completions``. Use this for
      SGLang or older vLLM that 500s on ``/v1/responses``. Reasoning
      models still surface their thinking via ``reasoning_content`` in
      ``additional_kwargs``, which ``graph.py`` already handles.

    Override with ``AGENT_LLM_API_FLAVOR=chat_completions`` when you
    see every rollout failing with ``InternalServerError`` from a
    server that *does* answer ``/v1/chat/completions`` correctly.
    """

    enable_thinking: bool = False
    """Whether the model emits thinking blocks at all.

    Forwarded as ``extra_body.chat_template_kwargs.enable_thinking`` to
    every LLM call, regardless of API surface (we always use the
    Responses API, but ``chat_template_kwargs`` is honored there too).

    - ``True``: reasoning models (Qwen3, o-series) emit thinking which
      we capture as ``reasoning`` blocks in the trajectory.
    - ``False``: even thinking-capable models skip reasoning, which is
      faster and matches Qwen2.5-7B-Instruct (paper's model).
    """

    preserve_thinking: bool = False
    """Whether prior-turn thinking is fed back into subsequent turns.

    When True (and the chat-completions flavor is in use):

    * ``chat_template_kwargs.preserve_thinking=True`` is sent so
      SGLang/vLLM keep ``<think>`` blocks from prior assistant messages
      in the rendered prompt instead of stripping them.
    * Outgoing assistant messages get a ``reasoning_content`` field
      populated from ``additional_kwargs.reasoning_content`` (the
      response-side capture path already in
      :class:`ReasoningPreservingChatOpenAI`). Verified on Qwen3.6-35B-A3B:
      ``preserve_thinking=True`` + the ``reasoning_content`` field bumps
      ``prompt_tokens`` by exactly the rendered ``<think>`` size; without
      either, the chat template strips them.

    Only effective when ``enable_thinking`` is True.
    """


    system_prompt_extra: str = ""
    """Free-form text appended to the rendered system prompt.

    Empty by default — behavior unchanged. Reserved for downstream packages
    (currently :mod:`ace_react`) that need to inject extra context — such
    as ACE's evolving playbook — into the agent without subclassing the
    prompt rendering pipeline. Not exposed via env vars: this is a runtime
    parameter that programs construct, not a deployment knob.
    """

    include_tool_call_template: bool = True
    """Whether to append paper Table 7's trailing ``<tool_call>{...}</tool_call>``
    format example to the system prompt.

    Default ``True`` is the paper-faithful prompt. Set ``False`` when
    the LLM/server already documents the function schema via
    ``bind_tools`` (which it always does in our setup) and the literal
    example confuses the model's tool-call generation. The body of the
    system prompt — protocol, ``\\boxed{}`` final-answer rule — is
    rendered either way.
    """

    question_template: QuestionTemplate = "hiexp_boxed"
    """Which user-message template to render around the question.

    - ``hiexp_boxed`` (default): paper Table 7 contract → final answer in
      ``\\boxed{...}``. Used for the HiExp multi-hop QA reproduction
      (HotpotQA / 2Wiki / Musique / Bamboogle).
    - ``browsecomp_plus``: BrowseComp-Plus contract → ``Explanation /
      Exact Answer / Confidence`` block (matches
      ``third_party/BrowseComp-Plus/search_agent/prompts.py``).
    - ``raw``: pass the question verbatim (for ablation / debugging).
    """

    # ── Agent loop guards ──────────────────────────────────────────────────
    max_search_limit: int = 10
    """Search-o1 default for multi-hop QA (run_search_o1.py:202)."""

    max_turn: int = 15
    """Search-o1 default for multi-hop QA (run_search_o1.py:203)."""

    # ── Search tool: MCP only ──────────────────────────────────────────────
    mcp_url: str | None = None
    """URL of the MCP retrieval server.

    Mirror of BrowseComp-Plus ``--mcp-url``. The agent has no embedded
    retriever — *all* search calls go through this server, which is
    operated by ``third_party/BrowseComp-Plus/searcher/mcp_server.py`` or
    any compatible deployment. Set via ``AGENT_MCP_URL``.
    """

    mcp_transport: MCPTransport = "sse"
    """MCP transport for ``langchain-mcp-adapters``. Matches BrowseComp-Plus
    server default (``mcp_server.py:55``)."""

    mcp_search_tool_name: str = "search"
    """Name of the MCP search tool we expect (``searcher/tools.py:21``)."""

    mcp_get_document_tool_name: str | None = "get_document"
    """Optional second MCP tool. Set to ``None`` to ignore even if exposed."""

    # ── HEK (Hierarchical Experience Knowledge) — HiExp-internal ───────────
    hek_path: str | None = None
    """Directory containing JSONL + npy files for E1/E2/E3 levels."""

    hek_embedding_model: str = "intfloat/multilingual-e5-base"
    """Paper §A: ``we employ multilingual-e5-base as the retriever``.

    This embedder is used *only* for HEK retrieval. The MCP search server
    has its own (independent) embedding stack inside.
    """

    e1_threshold: float = 0.8
    """Paper §A: ``We apply a 0.8 similarity threshold for case-based experiences (E1)``."""

    e1_top_k: int = 3
    """Paper does not specify; default 3 (see plan §10)."""

    e2_top_k: int = 5
    """Paper §A: ``For strategy-based experiences (E2 or E3), we select the top-5 candidates``."""

    use_e3_instead_of_e2: bool = False
    """Paper Table 2: E2+E1 outperforms E3+E1; default False keeps the strongest config."""

    enable_hek: bool = True
    """If False the agent runs as plain Search-o1 (no E1/E2 injection). Useful for ablation."""

    # ── Misc ────────────────────────────────────────────────────────────────
    seed: int = 42
    """Paper §A fixed random seed."""

    cache_dir: str = "./.hiexp_cache"

    # ── Internal sentinel ──────────────────────────────────────────────────
    _env_loaded: bool = field(default=False, repr=False)

    @classmethod
    def from_omegaconf(
        cls,
        cfg,
        *,
        dotenv_path: str | os.PathLike[str] | None = None,
        **overrides,
    ) -> "HiExpConfig":
        """Build a config from a composed OmegaConf YAML + env (for endpoints).

        YAML drives algorithmic settings (sampling, agent loop, MCP protocol,
        HEK params, seed). Endpoint fields (``llm_*``, ``mcp_url``,
        ``hek_path``) are pulled from environment variables loaded from
        ``.env`` so the same YAML works on different machines without edits.

        Precedence: ``overrides`` kwargs > env > YAML > dataclass defaults.
        """
        if dotenv_path is None:
            dotenv_path = Path(__file__).resolve().parents[2] / ".env"
        load_dotenv(dotenv_path, override=False)

        defaults = cls()

        def _g(node, path: str, fallback):
            cur = node
            for part in path.split("."):
                if cur is None:
                    return fallback
                cur = cur.get(part) if hasattr(cur, "get") else getattr(cur, part, None)
            return fallback if cur is None else cur

        yaml_values = dict(
            # LLM sampling (algorithmic; from YAML)
            temperature=_g(cfg, "llm.temperature", defaults.temperature),
            top_p=_g(cfg, "llm.top_p", defaults.top_p),
            max_response_tokens=_g(cfg, "llm.max_response_tokens", defaults.max_response_tokens),
            llm_api_flavor=_g(cfg, "llm.api_flavor", defaults.llm_api_flavor),
            enable_thinking=_g(cfg, "llm.enable_thinking", defaults.enable_thinking),
            preserve_thinking=_g(cfg, "llm.preserve_thinking", defaults.preserve_thinking),
            include_tool_call_template=_g(
                cfg, "llm.include_tool_call_template", defaults.include_tool_call_template
            ),
            # Agent loop
            max_search_limit=_g(cfg, "agent.max_search_limit", defaults.max_search_limit),
            max_turn=_g(cfg, "agent.max_turn", defaults.max_turn),
            question_template=_g(cfg, "agent.question_template", defaults.question_template),
            # MCP protocol details (URL itself comes from env)
            mcp_transport=_g(cfg, "mcp.transport", defaults.mcp_transport),
            mcp_search_tool_name=_g(cfg, "mcp.search_tool", defaults.mcp_search_tool_name),
            mcp_get_document_tool_name=_g(
                cfg, "mcp.get_document_tool", defaults.mcp_get_document_tool_name
            ),
            # HEK
            hek_embedding_model=_g(cfg, "hek.embedding_model", defaults.hek_embedding_model),
            e1_threshold=_g(cfg, "hek.e1_threshold", defaults.e1_threshold),
            e1_top_k=_g(cfg, "hek.e1_top_k", defaults.e1_top_k),
            e2_top_k=_g(cfg, "hek.e2_top_k", defaults.e2_top_k),
            use_e3_instead_of_e2=_g(cfg, "hek.use_e3_instead_of_e2", defaults.use_e3_instead_of_e2),
            enable_hek=_g(cfg, "hek.enabled", defaults.enable_hek),
            hek_path=_g(cfg, "hek.path", None),
            # Misc
            seed=_g(cfg, "misc.seed", defaults.seed),
            cache_dir=_g(cfg, "misc.cache_dir", defaults.cache_dir),
        )
        # Endpoint fields — env-driven.
        env_values = dict(
            llm_model=_get_env("AGENT_LLM_MODEL", defaults.llm_model),
            llm_base_url=_get_env("AGENT_LLM_BASE_URL", defaults.llm_base_url),
            llm_api_key=_get_env("AGENT_LLM_API_KEY", defaults.llm_api_key),
            llm_provider=_get_env("AGENT_LLM_PROVIDER", defaults.llm_provider),
            mcp_url=_get_env("AGENT_MCP_URL", defaults.mcp_url),
        )
        # HEK path: env wins over YAML (YAML default is null), CLI override
        # wins over env (handled by the **overrides merge below).
        if env_hek := _get_env("AGENT_HEK_PATH"):
            yaml_values["hek_path"] = env_hek

        merged = {**yaml_values, **env_values, **overrides, "_env_loaded": True}
        return cls(**merged)

    def validate(self) -> None:
        if not self.mcp_url:
            raise ValueError("HiExpAgent requires an MCP search server (AGENT_MCP_URL)")
        if self.llm_api_flavor not in ("responses", "chat_completions"):
            raise ValueError(
                f"llm_api_flavor must be 'responses' or 'chat_completions'; "
                f"got {self.llm_api_flavor!r}"
            )
        if self.llm_provider not in ("openai_compat", "openrouter"):
            raise ValueError(
                f"llm_provider must be 'openai_compat' or 'openrouter'; "
                f"got {self.llm_provider!r}"
            )
        if not 0.0 <= self.e1_threshold <= 1.0:
            raise ValueError(f"e1_threshold must be in [0, 1]; got {self.e1_threshold}")
        if self.e1_top_k <= 0 or self.e2_top_k <= 0:
            raise ValueError("top_k values must be positive")
        if self.enable_hek and not self.hek_path:
            raise ValueError("enable_hek=True requires hek_path (AGENT_HEK_PATH)")
