"""ACE-side ReAct adapter.

ACE (``third_party/ace``) wants a Generator that takes ``(question,
playbook, ...)`` and returns a final answer + reasoning trace. HiExp's
:class:`hiexp.agent.HiExpAgent` (with HEK disabled) already gives us
exactly that — a LangGraph ReAct loop over MCP search tools, BCP-shape
trajectory serialization, SGLang/vLLM compatibility — so we wrap it
rather than reimplement.

This package is **stateless framework glue**:
- :class:`ReActAgent` is constructed per query (cheap; LangGraph compile
  is cached internally) so a freshly updated playbook is picked up;
- :func:`render_playbook_preamble` is the single source of truth for how
  ACE's playbook text becomes the LLM-visible context.

No agent logic lives here. See ``src/hiexp/agent.py`` for that.
"""

from ace_react.agent import ReActAgent, render_playbook_preamble

__all__ = ["ReActAgent", "render_playbook_preamble"]
