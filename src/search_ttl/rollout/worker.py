"""Per-rollout worker hooks.

v1 keeps the rollout loop simple enough that the worker logic lives inside
``SearchAgent.arun`` — there's no separate worker process and nothing to
batch per GPU. This module is reserved as the future home of:

- a multi-process rollout pool (when MCP server can scale across cores);
- streaming-rollout hooks (LangGraph ``astream``) for real-time visualisation;
- per-worker resource limits (memory, file handles).

Adding a feature here should NOT change ``RolloutManager.generate``'s
signature — the manager is the public entry point.
"""

from __future__ import annotations
