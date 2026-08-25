"""ChatOpenAI subclass that preserves reasoning across vendor schemas.

Many OpenAI-compatible servers return reasoning text in a non-standard
field on the message object. Three shapes seen in the wild:

* vLLM/SGLang reasoning parsers, DeepSeek-R1, Qwen3-thinking →
  ``message.reasoning_content`` (string).
* OpenRouter → ``message.reasoning`` (string) and
  ``message.reasoning_details`` (typed list, e.g.
  ``[{"type": "reasoning.text", "text": "..."}]``).

``langchain_openai.ChatOpenAI`` builds messages via ``_convert_dict_to_message``
which only knows the standard OpenAI schema, so all of these are silently
dropped. This subclass overrides ``_create_chat_result`` to coerce
whichever field is present into
``AIMessage.additional_kwargs.reasoning_content``, so ``graph.py:376``
picks it up uniformly and emits ``type="reasoning"`` trajectory items.

Symmetrically, ``_get_request_payload`` is overridden so that — when
``inject_reasoning_on_request`` is set — outgoing assistant messages
carry the captured ``reasoning_content`` back as a non-standard field on
the message dict. Combined with
``chat_template_kwargs.preserve_thinking=True`` this lets SGLang's chat
template re-emit prior-turn ``<think>`` blocks into the rendered prompt
(verified on Qwen3.6-35B-A3B: ``prompt_tokens`` rises by the rendered
``<think>`` size). Without both pieces the prior thinking is silently
stripped between turns.
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI


def _coerce_reasoning(raw: dict) -> str | None:
    """Return reasoning text from a raw message dict, across vendor schemas."""
    rc = raw.get("reasoning_content")
    if isinstance(rc, str) and rc.strip():
        return rc
    r = raw.get("reasoning")
    if isinstance(r, str) and r.strip():
        return r
    details = raw.get("reasoning_details")
    if isinstance(details, list):
        parts = [
            blk.get("text", "")
            for blk in details
            if isinstance(blk, dict) and isinstance(blk.get("text"), str)
        ]
        joined = "\n\n".join(p.strip() for p in parts if p.strip())
        if joined:
            return joined
    return None


def _extract_choice_messages(response: Any) -> list[dict]:
    """Return the raw ``message`` dict for each choice in the response.

    Handles both the dict form (some servers / mocks) and the openai SDK
    object form by calling ``.model_dump()``.
    """
    if isinstance(response, dict):
        choices = response.get("choices") or []
        return [c.get("message") or {} for c in choices]
    try:
        dumped = response.model_dump()
    except Exception:
        return []
    choices = dumped.get("choices") or []
    return [c.get("message") or {} for c in choices]


class ReasoningPreservingChatOpenAI(ChatOpenAI):
    """``ChatOpenAI`` that copies ``reasoning_content`` into ``additional_kwargs``.

    When ``inject_reasoning_on_request`` is True it also writes the
    captured ``reasoning_content`` back into the outgoing assistant
    message dict so the server's chat template (with
    ``preserve_thinking=True``) can rehydrate prior-turn ``<think>``
    blocks. The Responses-API path is left untouched — that flavor has
    its own structured reasoning blocks.
    """

    inject_reasoning_on_request: bool = False

    def _create_chat_result(self, response, generation_info=None):
        result = super()._create_chat_result(response, generation_info=generation_info)

        raw_messages = _extract_choice_messages(response)
        for i, gen in enumerate(result.generations):
            if i >= len(raw_messages):
                break
            raw = raw_messages[i]
            reasoning = _coerce_reasoning(raw)
            if not reasoning or not isinstance(gen.message, AIMessage):
                continue
            ak = dict(getattr(gen.message, "additional_kwargs", {}) or {})
            # Don't clobber a value the base class somehow already set.
            if not ak.get("reasoning_content"):
                ak["reasoning_content"] = reasoning
                gen.message.additional_kwargs = ak
        return result

    def _get_request_payload(
        self,
        input_: LanguageModelInput,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict:
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        if not self.inject_reasoning_on_request:
            return payload
        # Responses-API path stores messages under ``input``; only patch
        # the chat-completions ``messages`` payload here.
        msgs_dict = payload.get("messages")
        if not isinstance(msgs_dict, list):
            return payload
        source = self._convert_input(input_).to_messages()
        if len(source) != len(msgs_dict):
            return payload
        for src, dst in zip(source, msgs_dict):
            if not isinstance(src, AIMessage) or not isinstance(dst, dict):
                continue
            ak = getattr(src, "additional_kwargs", {}) or {}
            rc = ak.get("reasoning_content")
            if isinstance(rc, str) and rc.strip() and "reasoning_content" not in dst:
                dst["reasoning_content"] = rc
        return payload


__all__ = ["ReasoningPreservingChatOpenAI"]
