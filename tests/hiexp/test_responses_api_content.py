from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from hiexp.agent import _last_ai_message_text
from hiexp.graph import _split_responses_api_content


# Realistic vLLM 0.19+ /v1/responses content shape: a list of typed blocks.
_VLLM_BLOCKS = [
    {
        "type": "reasoning",
        "id": "rs_abc",
        "summary": [],
        "content": [{"text": "The user asks about France's capital."}],
    },
    {"type": "text", "text": "Paris is the capital of France."},
]

# OpenAI's documented shape: reasoning text under ``summary``, not ``content``.
_OPENAI_BLOCKS = [
    {
        "type": "reasoning",
        "summary": [{"text": "Thinking step ..."}],
    },
    {"type": "text", "text": "Final."},
]


def test_split_vllm_shape():
    reasoning, text = _split_responses_api_content(_VLLM_BLOCKS)
    assert "France's capital" in reasoning
    assert text == "Paris is the capital of France."


def test_split_openai_shape():
    reasoning, text = _split_responses_api_content(_OPENAI_BLOCKS)
    assert reasoning == "Thinking step ..."
    assert text == "Final."


def test_split_empty():
    assert _split_responses_api_content([]) == ("", "")


def test_split_skips_unknown_blocks():
    blocks = [
        {"type": "function_call", "name": "search", "arguments": "{}"},
        {"type": "text", "text": "hi"},
    ]
    reasoning, text = _split_responses_api_content(blocks)
    assert reasoning == ""
    assert text == "hi"


def test_last_ai_message_text_pulls_text_block_only():
    msgs = [
        HumanMessage(content="?"),
        AIMessage(content=_VLLM_BLOCKS),
    ]
    assert _last_ai_message_text(msgs) == "Paris is the capital of France."


def test_last_ai_message_text_string_passthrough():
    msgs = [HumanMessage(content="?"), AIMessage(content="hello")]
    assert _last_ai_message_text(msgs) == "hello"


def test_last_ai_message_text_handles_no_text_block():
    msgs = [
        HumanMessage(content="?"),
        AIMessage(content=[{"type": "reasoning", "content": [{"text": "..."}]}]),
    ]
    assert _last_ai_message_text(msgs) == ""
