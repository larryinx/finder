"""Lightweight parsing helpers internal to the agent.

Currently empty placeholder; query parsing is no longer needed because
the agent uses native tool-calling messages (``AIMessage.tool_calls``)
instead of Search-o1's regex-based ``extract_between``.

Answer extraction lives in ``hiexp.evaluation.answer_extract`` so the
agent module stays decoupled from output formatting / scoring concerns.
"""
