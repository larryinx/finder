"""JSONL append helpers + BCP-shape conversion.

We append to a single ``rollouts.jsonl`` per step rather than dumping a
file per question. Atomic append uses a coarse lock over an asyncio loop 
because the writer is single-process.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

# Per-path lock so concurrent rollout workers can append safely.
_LOCKS: dict[Path, asyncio.Lock] = {}


def _lock_for(path: Path) -> asyncio.Lock:
    p = path.resolve()
    if p not in _LOCKS:
        _LOCKS[p] = asyncio.Lock()
    return _LOCKS[p]


def _json_default(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not json-serializable: {type(obj).__name__}")


async def append_jsonl(path: Path, record: Any) -> None:
    """Append one JSON record. Creates parent dirs and the file as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, default=_json_default, ensure_ascii=False)
    async with _lock_for(path):
        # `open` is sync but we hold the lock so writes are serialized.
        with path.open("a", encoding="utf-8") as f:
            f.write(line)
            f.write("\n")


def write_json(path: Path, obj: Any, *, indent: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, default=_json_default, ensure_ascii=False, indent=indent),
        encoding="utf-8",
    )


def write_jsonl(path: Path, records: list) -> None:
    """Overwrite ``path`` with one JSON record per line.

    Used by the resumable evaluator to trim ``rollouts.jsonl`` /
    ``rewards.jsonl`` down to the kept set before re-running the
    failed / missing rollouts.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, default=_json_default, ensure_ascii=False))
            f.write("\n")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def hash_text(text: str) -> str:
    """Stable hash for an experience snapshot — used in trajectory metadata."""
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def trajectory_to_bcp_record(
    *,
    trajectory,
    model_name: str,
    benchmark: str = "browsecomp_plus",
    extra_metadata: dict | None = None,
) -> dict:
    """BCP-shape dict, compatible with ``runs/bcp/.../run_*.json``.

    Mirrors ``ace_react.bcp.persist.trajectory_to_bcp_record`` but pulls
    fields straight off our ``Trajectory`` dataclass instead of HiExp's
    rich object — keeps the dependency one-way (search_ttl → ace_react).
    """
    return {
        "metadata": {
            "model": model_name,
            "benchmark": benchmark,
            "ttl": {**(extra_metadata or {}), **trajectory.metadata_ttl},
        },
        "query_id": trajectory.qid,
        "question": trajectory.question,
        # Prefer the explicit ``golden_answers`` list (populated by the
        # rollout layer from QuestionRecord) so multi-gold benchmarks
        # like HotpotQA round-trip every accepted alias. Fall back to
        # the single ``target`` for legacy BCP / BrowseComp rows.
        "golden_answers": (
            list(trajectory.golden_answers)
            if trajectory.golden_answers
            else ([trajectory.target] if trajectory.target else [])
        ),
        "benchmark": benchmark,
        "tool_call_counts": trajectory.tool_call_counts,
        "status": trajectory.status,
        "retrieved_docids": trajectory.retrieved_docids,
        "result": trajectory.steps,
        "prediction_raw": trajectory.raw_response,
        "prediction_exact": trajectory.extracted_answer or "",
        # Persisted so a resumed eval can recover avg_n_search /
        # avg_n_turn for already-finished rollouts without re-running
        # the agent. Records written before this field existed fall
        # back to deriving search_count from tool_call_counts.
        "metrics": dict(trajectory.metrics),
    }
