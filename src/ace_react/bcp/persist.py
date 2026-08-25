"""Persist a HiExp ReAct trajectory as a BCP-shape ``run_*.json``.

Output schema mirrors ``third_party/BrowseComp-Plus/search_agent/qwen_client.py``
``_persist_response``: ``{metadata, query_id, tool_call_counts, status,
retrieved_docids, result[]}``. ``result[]`` items are
``{type, tool_name, arguments, output}`` with ``type`` in
``{reasoning, tool_call, output_text}``. We keep ACE-side bookkeeping
(``ace_*`` prefix) under ``metadata.ace`` so it doesn't pollute BCP's
required fields.

The BCP grader (``evaluate_with_openai.py``) reads:
* ``metadata.model`` — emitted as the LLM name in the summary;
* ``query_id`` — joins to gold;
* ``status`` — completion rate;
* ``tool_call_counts``, ``retrieved_docids`` — retrieval / tool stats;
* ``result[-1].output`` if ``type == output_text`` — the final answer.

We make sure every emitted record satisfies all of those.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from hiexp.agent import HiExpTrajectory


def trajectory_to_bcp_record(
    *,
    query_id: str | None,
    question: str,
    raw_response: str,
    trajectory: HiExpTrajectory,
    model_name: str,
    benchmark: str = "browsecomp_plus",
    ace_metadata: dict | None = None,
) -> dict[str, Any]:
    """Build a BCP-shape record dict from a HiExp trajectory.

    ``ace_metadata`` is stashed under ``metadata.ace`` for our own
    bookkeeping (training step id, playbook size, mode, etc.) without
    breaking BCP's expected shape.
    """
    bcp_items: list[dict[str, Any]] = []
    hiexp_items: list[dict[str, Any]] = []
    for it in trajectory.items:
        d = {
            "type": it.type,
            "tool_name": it.tool_name,
            "arguments": it.arguments,
            "output": it.output,
        }
        if it.type in {"e1_inject", "e2_inject"}:
            hiexp_items.append(d)
        else:
            bcp_items.append(d)

    record = {
        "metadata": {
            "model": model_name,
            "benchmark": benchmark,
            "ace": ace_metadata or {},
            "hiexp": {"experience_injections": hiexp_items},
        },
        "query_id": query_id,
        "question": question,
        "tool_call_counts": trajectory.tool_call_counts,
        "status": trajectory.status,
        "retrieved_docids": trajectory.retrieved_docids,
        "result": bcp_items,
    }
    return record


def write_bcp_run_json(
    out_dir: Path,
    record: dict[str, Any],
) -> Path:
    """Atomically write ``run_<query_id>.json`` under ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    qid = record.get("query_id") or "unknown"
    path = out_dir / f"run_{qid}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
    return path


__all__ = ["trajectory_to_bcp_record", "write_bcp_run_json"]
