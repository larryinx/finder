"""Lightweight LLM call + JSON parsing helpers shared by training scripts."""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


class _ReasoningLiftChatOpenAI(ChatOpenAI):
    """ChatOpenAI that rescues answers misfiled under ``reasoning_content``.

    SGLang reasoning parsers key on a closing ``</think>`` tag. Nemotron-3
    honors ``enable_thinking=False`` (answers directly) but then never
    emits the tag, so the ``nemotron_3`` parser files the entire answer
    under ``reasoning_content`` and ``content`` arrives ``None`` — which
    LangChain silently turns into ``""`` and every downstream parse fails
    (the judge grades everything incorrect, distillation yields nothing).
    Lift the answer back into ``content`` before message conversion.
    ``_create_chat_result`` is the funnel for both sync and async invoke.
    """

    def _create_chat_result(self, response, generation_info=None):
        if not isinstance(response, dict):
            for choice in response.choices or []:
                message = choice.message
                if message.content is None:
                    reasoning = getattr(message, "reasoning_content", None)
                    if reasoning:
                        message.content = reasoning
        return super()._create_chat_result(response, generation_info)


def make_chat_model(
    model: str,
    base_url: str,
    api_key: str = "EMPTY",
    temperature: float = 0.3,
    top_p: float = 0.95,
    max_tokens: int = 2048,
    enable_thinking: bool = False,
) -> ChatOpenAI:
    """Build a ChatOpenAI for non-agent calls (distill / cluster / judge).

    ``enable_thinking`` defaults to **False** because every caller of
    this helper produces structured output (JSON for Table 11/12,
    yes/no for the BCP grader). On reasoning-model servers (SGLang
    Qwen3, vLLM with reasoning parsers), letting the model think
    strips ``reasoning_content`` via the OpenAI schema and often
    leaves ``content=""`` when reasoning eats the token budget,
    silently producing parse failures downstream. Pass
    ``enable_thinking=True`` only when the caller really needs CoT.
    """
    return _ReasoningLiftChatOpenAI(
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        extra_body={
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
        },
    )


def chat_once(llm: ChatOpenAI, instruction: str, system: str | None = None) -> str:
    msgs = []
    if system:
        msgs.append(SystemMessage(content=system))
    msgs.append(HumanMessage(content=instruction))
    resp = llm.invoke(msgs)
    return resp.content if isinstance(resp.content, str) else str(resp.content)


async def achat_once(
    llm: ChatOpenAI, instruction: str, system: str | None = None
) -> str:
    """Async equivalent of :func:`chat_once` for batching with ``asyncio.gather``."""
    msgs = []
    if system:
        msgs.append(SystemMessage(content=system))
    msgs.append(HumanMessage(content=instruction))
    resp = await llm.ainvoke(msgs)
    return resp.content if isinstance(resp.content, str) else str(resp.content)


_FENCED = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def _strip_fences(text: str) -> str:
    matches = _FENCED.findall(text)
    return matches[-1] if matches else text


def parse_json_object(text: str) -> dict[str, Any] | None:
    """Best-effort: pull a JSON object out of ``text``.

    Tries the whole string first, then any fenced ``json`` block, then
    a greedy ``{...}`` substring. Returns ``None`` on failure.
    """
    candidates = [text, _strip_fences(text)]
    # Try greedy {...} substring as last resort.
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        candidates.append(m.group(0))
    for cand in candidates:
        cand = cand.strip()
        if not cand:
            continue
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def parse_json_array(text: str) -> list[Any] | None:
    """Same idea as :func:`parse_json_object` but expects a top-level list."""
    candidates = [text, _strip_fences(text)]
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if m:
        candidates.append(m.group(0))
    for cand in candidates:
        cand = cand.strip()
        if not cand:
            continue
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, list):
            return obj
    return None
