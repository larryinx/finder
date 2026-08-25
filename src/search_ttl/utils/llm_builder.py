"""Shared chat-LLM builder.

The rollout agent, both reflectors, and the curator all spin up a
LangChain chat model with the same four orthogonal toggles:

- ``llm_provider``       — ``openai_compat`` (default; vLLM / SGLang /
  vanilla OpenAI / OSS Responses servers) **or** ``openrouter``
  (``langchain_openrouter.ChatOpenRouter`` against
  ``https://openrouter.ai/api/v1``; reasoning round-trip is built into
  the package — no SGLang chat-template tricks needed).
- ``llm_api_flavor``     — only meaningful on ``openai_compat``:
  ``chat_completions`` (HiExp's ``ReasoningPreservingChatOpenAI``,
  which re-injects the prior turn's ``reasoning_content`` into the
  next request so multi-turn ReAct loops preserve thinking) **or**
  ``responses`` (OpenAI's Responses API via plain ``ChatOpenAI`` with
  ``reasoning={"summary": "auto"}``).
- ``enable_thinking``    — flips the ``enable_thinking`` chat-template
  kwarg the server forwards to Qwen / GLM-style models. On
  ``openrouter`` it maps to the constructor's ``reasoning={'summary':
  'auto'}`` instead — OpenRouter has no chat-template concept.
- ``preserve_thinking``  — only meaningful on ``openai_compat`` +
  ``chat_completions`` when ``enable_thinking`` is also on; turns on
  the request-side reasoning_content injection that makes multi-turn
  ReAct actually "see" its prior thoughts. Ignored on ``openrouter``
  (the package handles the round-trip natively).

Before this helper, only the agent had all four knobs wired; the
reflector + curator hardcoded ``ChatOpenAI`` and thus silently lost
``preserve_thinking`` even when their YAML had ``enable_thinking: true``.
The helper centralises the construction so the four call sites stay in
lockstep.
"""

from __future__ import annotations

import os
from typing import Any

# Provider + API flavor are both ``str`` to keep OmegaConf-friendly typing.
LlmProvider = str  # "openai_compat" | "openrouter"
ApiFlavor = str    # "chat_completions" | "responses"


def _resolve_openrouter_provider_kwarg() -> dict[str, Any] | None:
    """Build the ``openrouter_provider`` constructor kwarg from env, or
    return ``None`` to omit it entirely.

    Reads two env vars:

    - ``OPENROUTER_PROVIDER``           — single provider name, e.g.
      ``Anthropic`` / ``OpenAI`` / ``Google``. Unset/empty → return
      ``None`` (no provider routing override).
    - ``OPENROUTER_PROVIDER_FALLBACKS`` — ``true`` (default) or
      ``false``. When ``true`` we emit ``{"order": [provider]}`` (try
      this provider first, fall back to others if unavailable). When
      ``false`` we emit ``{"only": [provider]}`` (strict — fail if
      this provider is unavailable). See
      https://openrouter.ai/docs/guides/routing/provider-selection.
    """
    provider = (os.environ.get("OPENROUTER_PROVIDER") or "").strip()
    if not provider:
        return None
    fb_raw = (os.environ.get("OPENROUTER_PROVIDER_FALLBACKS") or "true").strip().lower()
    allow_fallbacks = fb_raw not in ("false", "0", "no", "off")
    if allow_fallbacks:
        return {"order": [provider]}
    return {"only": [provider]}


def build_chat_llm(
    *,
    model: str,
    base_url: str,
    api_key: str,
    llm_provider: LlmProvider = "openai_compat",
    llm_api_flavor: ApiFlavor = "chat_completions",
    temperature: float = 1.0,
    top_p: float = 0.95,
    max_tokens: int = 8192,
    enable_thinking: bool = False,
    preserve_thinking: bool = True,
    timeout_s: float = 600.0,
):
    """Construct a chat model with TTL's standard reasoning wiring.

    Branching, in order:

    1. ``llm_provider == 'openrouter'`` → ``ChatOpenRouter``. The
       ``reasoning`` constructor kwarg is set to
       ``{"summary": "auto"}`` when ``enable_thinking`` is on; the
       ``langchain-openrouter`` package handles the reasoning_content
       round-trip on subsequent turns. ``preserve_thinking`` and
       ``llm_api_flavor`` have no analog on this path and are ignored
       (matches HiExp's behavior).
    2. ``llm_provider == 'openai_compat'`` + ``llm_api_flavor ==
       'responses'`` → plain ``ChatOpenAI`` with
       ``output_version='responses/v1'``; the Responses API natively
       round-trips reasoning blocks across turns.
    3. ``llm_provider == 'openai_compat'`` + ``llm_api_flavor ==
       'chat_completions'`` → HiExp's
       ``ReasoningPreservingChatOpenAI`` with
       ``inject_reasoning_on_request`` gated on both
       ``enable_thinking`` AND ``preserve_thinking``.
    """
    # ── OpenRouter ────────────────────────────────────────────────────────
    if llm_provider == "openrouter":
        # Lazy import — the package is optional and only required when
        # this provider is selected.
        from langchain_openrouter import ChatOpenRouter  # type: ignore[import-not-found]

        or_kwargs: dict[str, Any] = dict(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            # ChatOpenRouter.timeout is MILLISECONDS (alias for the SDK's
            # timeout_ms — see langchain_openrouter/chat_models.py:201).
            # Our ``timeout_s`` is seconds (matches ChatOpenAI). Forgetting
            # to convert turns the agent default 600s into 600ms, which
            # times out every reasoning-model request and surfaces as
            # ``[llm_error] ReadTimeout:`` with an empty body.
            timeout=int(timeout_s * 1000),
        )
        if enable_thinking:
            or_kwargs["reasoning"] = {"summary": "auto"}
        # Optional provider-routing override (single provider, optional
        # strict mode). All four ReAct loops in this repo build their
        # LLM through this helper, so a single pair of env vars steers
        # routing for the whole pipeline.
        provider_kwarg = _resolve_openrouter_provider_kwarg()
        if provider_kwarg is not None:
            or_kwargs["openrouter_provider"] = provider_kwarg
        return ChatOpenRouter(**or_kwargs)

    # ── OpenAI-compatible (vLLM / SGLang / OpenAI / OSS Responses) ────────
    from langchain_openai import ChatOpenAI

    chat_template_kwargs: dict[str, Any] = {"enable_thinking": enable_thinking}
    if enable_thinking and preserve_thinking:
        chat_template_kwargs["preserve_thinking"] = True

    common = dict(
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        extra_body={"chat_template_kwargs": chat_template_kwargs},
        timeout=timeout_s,
    )

    if llm_api_flavor == "responses":
        # Install the vLLM Responses-API status patch (idempotent) so
        # assistant message items carry the required ``status`` field
        # in the outgoing ``input`` array. Without this, multi-turn
        # ReAct against vLLM 400s on union validation.
        from search_ttl.utils.responses_api_compat import (  # noqa: F401
            _install_responses_api_status_patch,
        )
        _install_responses_api_status_patch()
        return ChatOpenAI(
            **common, reasoning={"summary": "auto"}, output_version="responses/v1"
        )

    # Default: HiExp's reasoning-preserving wrapper. When
    # ``enable_thinking`` is off this behaves like plain ChatOpenAI —
    # the wrapper is harmless in that case (no reasoning to inject).
    from hiexp.utils.reasoning_chat_openai import (  # type: ignore[import-not-found]
        ReasoningPreservingChatOpenAI,
    )

    return ReasoningPreservingChatOpenAI(
        **common,
        inject_reasoning_on_request=(enable_thinking and preserve_thinking),
    )
