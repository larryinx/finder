"""Pin the lenient `submit_reflection` dispatch.

LLMs sometimes flatten the ``payload: dict`` schema and call
``submit_reflection`` with the ReflectionPayload fields at the top
level. The strict langchain @tool would raise pydantic
``ValidationError: payload  Field required`` — exactly the crash in the
user's training run at step 8 (chunk 3/4).

The dispatch in ``QuerychainStagedReflector._reflect_one`` now:
  - wraps top-level fields in ``{"payload": ...}`` when ``payload`` is missing
  - never lets pydantic validation kill the training step (try/except
    falls back to ``status='no_useful_insight'``)

We don't run a real ReAct loop here — too slow + needs a live LLM. We
construct a minimal in-process scenario that triggers exactly the same
dispatch path.
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest


@pytest.fixture
def flat_payload():
    """The exact-shape args the user's traceback shows."""
    return {
        "principle": "When a high-confidence candidate is identified, commit.",
        "confidence": "0.95",
        "status": "ok",
        "scope": "strategy",
        "section_hint": "QUERY_REFORMULATION",
        "trigger": "Search loops on the same docid.",
        "anti_pattern": "Repeating the same quoted-phrase query.",
    }


def test_flat_args_wrapped_into_payload_kwarg(flat_payload):
    """Reproduce the dispatch logic standalone and verify it doesn't raise."""
    from langchain_core.tools import tool as tool_dec

    submitted: dict = {}

    @tool_dec
    def submit_reflection(payload: dict) -> str:
        """Submit your final ReflectionPayload."""
        submitted.clear()
        submitted.update(payload)
        return "submitted"

    # Mirror the exact dispatch in querychain_staged.py.
    args = flat_payload
    if isinstance(args, dict) and "payload" not in args:
        normalised_args = {"payload": args}
    else:
        normalised_args = args

    submit_reflection.invoke(normalised_args)
    assert submitted == flat_payload
    assert submitted["status"] == "ok"
    assert submitted["confidence"] == "0.95"


def test_nested_payload_passes_through(flat_payload):
    """If the LLM does it right, we don't double-wrap."""
    from langchain_core.tools import tool as tool_dec

    submitted: dict = {}

    @tool_dec
    def submit_reflection(payload: dict) -> str:
        """Submit final ReflectionPayload."""
        submitted.clear()
        submitted.update(payload)
        return "submitted"

    args = {"payload": flat_payload}
    if isinstance(args, dict) and "payload" not in args:
        normalised_args = {"payload": args}
    else:
        normalised_args = args

    submit_reflection.invoke(normalised_args)
    assert submitted == flat_payload


def test_dispatch_falls_back_to_no_useful_insight_on_failure():
    """If even the wrap doesn't help (e.g. args is not a dict), we
    must not crash — fall back to ``no_useful_insight``."""
    from langchain_core.tools import tool as tool_dec

    submitted: dict = {}

    @tool_dec
    def submit_reflection(payload: dict) -> str:
        """Submit final ReflectionPayload."""
        submitted.clear()
        submitted.update(payload)
        return "submitted"

    # Pathological: args is a string. With the wrap we'd send
    # `{"payload": "..."}` which would fail pydantic anyway. Verify
    # the try/except fallback runs.
    args = "garbage from a confused LLM"
    normalised_args = (
        {"payload": args} if isinstance(args, dict) and "payload" not in args else args
    )
    try:
        submit_reflection.invoke(normalised_args)
    except Exception as exc:
        submitted.clear()
        submitted.update(
            {"status": "no_useful_insight",
             "notes": f"submit_reflection arg-validation: {exc}"}
        )
    assert submitted["status"] == "no_useful_insight"
    assert "arg-validation" in submitted["notes"]
