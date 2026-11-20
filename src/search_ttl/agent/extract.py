"""Answer / query extraction helpers.

We re-use HiExp's well-tested regex for ``Exact Answer:`` parsing rather
than maintaining a parallel copy — see
``src/hiexp/evaluation/answer_extract.py:21`` for the canonical pattern
(handles bold-markdown wrapping, trailing periods, etc.).
"""

from __future__ import annotations

from typing import Iterable

from search_ttl.rollout.schemas import QueryRecord, Trajectory


def extract_exact_answer(text: str) -> str | None:
    """Pull the BCP-format ``Exact Answer:`` block; None when missing."""
    from hiexp.evaluation.answer_extract import extract_exact_answer as _impl  # type: ignore[import-not-found]

    return _impl(text)


def extract_boxed_answer(text: str) -> str | None:
    """Pull the HiExp paper Table 7 ``\\boxed{...}`` block; None when missing."""
    from hiexp.evaluation.answer_extract import extract_boxed_answer as _impl  # type: ignore[import-not-found]

    boxed = _impl(text or "")
    return boxed or None


def extract_answer_for_template(text: str, template: str) -> str | None:
    """Pick the right extractor for the user-prompt template.

    ``boxed_answer`` → ``\\boxed{}``; everything else → ``Exact Answer:``
    (the BCP / BrowseComp contract).
    """
    if template == "boxed_answer":
        return extract_boxed_answer(text)
    return extract_exact_answer(text)


def extract_queries(trajectory: Trajectory, *, thought_chars: int = 200) -> list[QueryRecord]:
    """Build the masked query view that Reward / Reflector consume.

    ``preceding_thought`` is truncated; ``top_doc_ids`` keep the IDs but
    drop the snippet text, so downstream consumers can't accidentally
    leak document content into reflection prompts.
    """
    queries: list[QueryRecord] = []
    last_thought = ""
    seen_docids: set[str] = set()
    q_idx = 0
    for step in trajectory.steps:
        kind = step.get("type")
        if kind == "reasoning":
            chunks = step.get("output") or []
            if isinstance(chunks, list) and chunks:
                last_thought = " ".join(str(c) for c in chunks)[:thought_chars]
            elif isinstance(chunks, str):
                last_thought = chunks[:thought_chars]
        elif kind == "tool_call":
            args = step.get("arguments") or {}
            q = args.get("query") or args.get("q") or ""
            queries.append(
                QueryRecord(
                    idx=q_idx,
                    query=str(q),
                    preceding_thought=last_thought,
                    n_results=0,
                    top_doc_ids=[],
                )
            )
            q_idx += 1
        elif kind == "tool_result" and queries:
            out = step.get("output") or []
            ids = _doc_ids_from_tool_result(out)
            queries[-1].n_results = len(out) if isinstance(out, list) else 0
            queries[-1].top_doc_ids = ids[:5]
            seen_docids.update(ids)
    return queries


def _doc_ids_from_tool_result(out) -> list[str]:
    if not isinstance(out, list):
        return []
    ids: list[str] = []
    for entry in out:
        if isinstance(entry, dict):
            doc_id = entry.get("docid") or entry.get("doc_id") or entry.get("id")
            if doc_id is not None:
                ids.append(str(doc_id))
    return ids


def attach_query_records(trajectory: Trajectory) -> None:
    """In-place fill `trajectory.queries` from `trajectory.steps`."""
    trajectory.queries = extract_queries(trajectory)
