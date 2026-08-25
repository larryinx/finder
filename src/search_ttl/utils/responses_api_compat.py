"""vLLM Responses-API compatibility patches.

OpenAI's ``ResponseOutputMessage`` schema marks ``status`` as a
**required** literal field (``in_progress`` / ``completed`` /
``incomplete``). When LangChain's
``_construct_responses_api_input`` builds an assistant ``message``
item for round-trip into the next request's ``input`` array, it
omits ``status`` entirely (see ``langchain_openai/chat_models/base.py``
around line 4458-4466). The OpenAI hosted Responses API tolerates the
omission, but **vLLM's pydantic validator is strict** and rejects the
request with a multi-page union-decomposition error before the model
ever sees it. The user-facing symptom is a 400 Bad Request with a
giant error body listing every union variant the message could match
(McpApprovalRequest, ResponseCustomToolCall, ...) and the fields each
expected.

The fix: monkey-patch ``_construct_responses_api_input`` once at
import time so every assistant message item in the outgoing ``input``
array gets ``status: "completed"`` if it's missing. Idempotent — the
patch self-marks and refuses to re-install.

This is intentionally targeted at this single LangChain↔vLLM
incompatibility. Other patches belong in their own helpers (see
``tool_call_recovery.sanitize_responses_message_for_round_trip`` for
the mcp_call strip).
"""

from __future__ import annotations

import logging
from typing import Any

LOG = logging.getLogger(__name__)

# Sentinel attribute we stamp on the patched function so calling
# ``_install_responses_api_status_patch`` repeatedly is a no-op.
_PATCH_MARKER = "_ttl_search_status_patched"


def _install_responses_api_status_patch() -> None:
    """Wrap LangChain's Responses-API input serializer so every
    assistant ``message`` item carries ``status: 'completed'``.

    Idempotent. Safe to call from multiple module imports.
    """
    import langchain_openai.chat_models.base as _lc_base

    orig = getattr(_lc_base, "_construct_responses_api_input", None)
    if orig is None:
        # LangChain version doesn't have this internal — nothing to patch.
        return
    if getattr(orig, _PATCH_MARKER, False):
        return  # already patched

    # ``*args`` / ``**kwargs`` pass through whatever extra parameters the
    # installed langchain-openai adds (1.6+ passes ``store=``).
    def patched(messages: Any, *args: Any, **kwargs: Any) -> list:
        out = orig(messages, *args, **kwargs)
        for item in out:
            if (
                isinstance(item, dict)
                and item.get("type") == "message"
                and item.get("role") == "assistant"
                and "status" not in item
            ):
                # ``completed`` is the right marker for a fully
                # materialised assistant turn (vs. ``in_progress`` for
                # streaming or ``incomplete`` for truncated output).
                # All assistant messages we round-trip are completed.
                item["status"] = "completed"
        return out

    setattr(patched, _PATCH_MARKER, True)
    _lc_base._construct_responses_api_input = patched
    LOG.debug(
        "patched langchain_openai._construct_responses_api_input to inject "
        "status='completed' on assistant message items (vLLM Responses-API "
        "compat)"
    )


# Install on first import. Modules that build a Responses-API
# ``ChatOpenAI`` should import this module so the patch is in place
# before any request is constructed.
_install_responses_api_status_patch()
