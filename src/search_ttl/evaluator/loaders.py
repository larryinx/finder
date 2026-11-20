"""Per-benchmark dataset loaders for Evaluator.

Each loader returns a list of plain dicts shaped like the ACE BCP
loader (``{query_id, question, target, context}``), which the
rollout layer then wraps as :class:`QuestionRecord`.

Today:

* ``load_bcp_split`` (re-exported from ``ace_react.bcp``) — needs a
  TREC-style ``queries.tsv`` + decrypted_jsonl pair, returns rows
  whose ``target`` is the first golden answer for BCP.
* ``load_browsecomp_split`` — single JSONL produced by
  ``scripts/data_prep/browsecomp/build_dataset.py``; one row =
  ``{"query_id", "question", "target"}``. No context, no qrels.

``get_loader(benchmark_name)`` resolves the right one from a small
registry so Evaluator stays benchmark-agnostic.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

# Re-export for convenience; keeps Evaluator from importing two namespaces.
from ace_react.bcp import load_bcp_split  # type: ignore[import-not-found]


def load_browsecomp_split(eval_jsonl: str | Path) -> list[dict[str, Any]]:
    """Load a single decrypted BrowseComp JSONL.

    The file is produced by ``scripts/data_prep/browsecomp/build_dataset.py``;
    each line is ``{"query_id", "question", "target"}`` (XOR-decrypted
    from the official ``browse_comp_test_set.csv`` using the per-row
    ``canary`` as the key).
    """
    path = Path(eval_jsonl)
    if not path.exists():
        raise FileNotFoundError(
            f"BrowseComp eval jsonl not found: {path}. "
            "Run `bash scripts/data_prep/prepare.sh browsecomp` to materialize it."
        )
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            rows.append(
                {
                    "query_id": str(d.get("query_id") or d.get("qid") or ""),
                    "question": str(d["question"]),
                    "target": str(d.get("target") or d.get("answer") or ""),
                    "context": "",
                }
            )
    return rows


def load_hotpotqa_split(eval_jsonl: str | Path) -> list[dict[str, Any]]:
    """Load a HotpotQA JSONL (FlashRAG-shaped).

    Accepts files produced by either:
      * ``scripts/data_prep/hotpotqa/build_dataset.py`` (the new TTL
        format, which includes a convenience ``target``), OR
      * ``scripts/data_prep/hiexp/build_eval_hotpotqa.py`` (HiExp's
        existing file at ``data/hiexp/eval/hotpotqa_dev_sample1000_seed42.jsonl``;
        only has ``golden_answers``).

    Each row contains a ``golden_answers`` list so the qa_em_f1 reward
    can take max-over-golds for F1/CEM/EM. ``target`` is synthesized
    from ``golden_answers[0]`` when absent.
    """
    path = Path(eval_jsonl)
    if not path.exists():
        raise FileNotFoundError(
            f"HotpotQA eval jsonl not found: {path}. "
            "Run `bash scripts/data_prep/prepare.sh hotpotqa` to materialize it, "
            "or point at the existing data/hiexp/eval/hotpotqa_dev_sample1000_seed42.jsonl."
        )
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            golds = list(d.get("golden_answers") or [])
            target = str(d.get("target") or "") or (golds[0] if golds else "")
            rows.append(
                {
                    "query_id": str(d.get("query_id") or d.get("qid") or ""),
                    "question": str(d["question"]),
                    "target": target,
                    "golden_answers": golds,
                    "context": "",
                }
            )
    return rows


# ── registry ─────────────────────────────────────────────────────────────


def get_loader(benchmark: str) -> Callable[..., list[dict[str, Any]]]:
    """Return the loader function for a benchmark name.

    ``browsecomp_plus`` returns ``load_bcp_split`` (TSV + decrypted
    jsonl). ``browsecomp`` returns ``load_browsecomp_split`` (single
    jsonl). Unknown names fall back to BCP for backwards compatibility.
    """
    table: dict[str, Callable[..., list[dict[str, Any]]]] = {
        "browsecomp_plus": load_bcp_split,
        "browsecomp": load_browsecomp_split,
        "hotpotqa": load_hotpotqa_split,
    }
    return table.get(benchmark, load_bcp_split)


def load_split(
    benchmark: str,
    split_path: str | Path,
    *,
    decrypted_jsonl: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Load one split of ``benchmark`` through its registered loader.

    BrowseComp-Plus needs the ``queries.tsv`` + ``decrypted_jsonl`` pair;
    every other loader takes a single file. Used by both the trainer
    (``scripts/ttl/run.py``, train split) and the Evaluator, so a new
    benchmark only has to be registered in ``get_loader``.
    """
    loader = get_loader(benchmark)
    if benchmark == "browsecomp_plus":
        return loader(split_path, decrypted_jsonl=decrypted_jsonl)
    return loader(split_path)
