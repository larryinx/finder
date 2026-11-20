"""OpenAI-compatible client builders shared across reward / reflector / curator.

Two builders:

- ``make_chat_model``: thin LangChain ``ChatOpenAI`` factory mirroring
  ``hiexp_prompt._llm_utils.make_chat_model`` so we get the same defaults
  (timeout, retry, structured-output behaviour) for free.
- ``build_no_think_openai_client``: raw ``openai.OpenAI`` client that
  auto-injects ``extra_body.chat_template_kwargs.enable_thinking=False`` on
  every chat completion. Necessary for SGLang-served Qwen3 reasoning
  models when we want structured JSON output (otherwise the entire token
  budget goes to ``reasoning_content`` and ``content`` is None).
"""

from __future__ import annotations

from typing import Any

import openai


def build_no_think_openai_client(base_url: str, api_key: str) -> openai.OpenAI:
    """Wrap an ``openai.OpenAI`` so every call sets ``enable_thinking=False``.

    Use for Reward analyzer / Reflector finalize / Curator — anywhere we
    need clean JSON / structured output from a reasoning model.
    """
    client = openai.OpenAI(api_key=api_key or "EMPTY", base_url=base_url)
    original_create = client.chat.completions.create

    def patched_create(**kw: Any):
        extra = dict(kw.pop("extra_body", {}) or {})
        ctk = dict(extra.get("chat_template_kwargs", {}) or {})
        ctk.setdefault("enable_thinking", False)
        extra["chat_template_kwargs"] = ctk
        response = original_create(extra_body=extra, **kw)
        # Nemotron-3 honors enable_thinking=False but never emits
        # </think>, so SGLang's nemotron_3 reasoning parser files the
        # whole (non-thinking) answer under reasoning_content and
        # content comes back None. Lift it back so callers see it.
        for choice in getattr(response, "choices", None) or []:
            message = choice.message
            if message.content is None:
                reasoning = getattr(message, "reasoning_content", None)
                if reasoning:
                    message.content = reasoning
        return response

    client.chat.completions.create = patched_create  # type: ignore[assignment]
    return client


def make_chat_model_for_reflector(
    *,
    model: str,
    base_url: str,
    api_key: str,
    temperature: float = 0.7,
    enable_thinking: bool = False,
):
    """LangChain chat model used by Reflector ReAct loops.

    We delegate to ``hiexp_prompt._llm_utils.make_chat_model`` so the rest
    of the codebase has one place to tune retry / timeout behaviour. If
    ``enable_thinking=True`` the agent will run reasoning models with
    ``<think>`` blocks — slower but sometimes higher-quality reflection.
    """
    # TODO: Write one in our own
    from hiexp_prompt._llm_utils import make_chat_model

    return make_chat_model(
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        enable_thinking=enable_thinking,
    )
