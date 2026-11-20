"""Bundle a hiexp ``run_*.json`` directory into a TTL ``rollouts.jsonl``.

``scripts/hiexp_prompt/eval/run_hotpotqa.py`` writes one JSON file per
query (``run_<qid>.json``); ``scripts/ttl/rejudge_*`` scripts read a
single ``rollouts.jsonl`` (one record per line). The two record
schemas overlap in everything that matters (``query_id``, ``question``,
``golden_answers``, ``status``, ``tool_call_counts``,
``retrieved_docids``, ``result``, ``prediction_raw``) but differ in
two small ways that this script normalises:

* hiexp inlines a search's hit list inside the ``tool_call`` step's
  ``output`` field as a JSON string; TTL emits a separate
  ``tool_result`` step with the parsed list. We split each hiexp
  ``tool_call`` into a ``tool_call`` (without ``output``) followed by
  a ``tool_result`` (with the parsed list) so
  ``rejudge_rollouts._rebuild_queries`` can count results / extract
  docids → accurate ``redundant_query_count`` / ``novel_doc_ratio``.
* hiexp names the extracted answer ``prediction_boxed`` and TTL names
  it ``prediction_exact``. We carry the value under both keys so any
  reward stack reads the same value (``search_agent_lasj`` reads
  ``prediction_exact``).

Idempotent: rebuilds the bundle only when the output is missing or
``--force`` is passed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _normalise_steps(result: list) -> list:
    """Split each ``tool_call``-with-inlined-output into a call + result pair."""
    out: list = []
    for s in result:
        if not isinstance(s, dict):
            out.append(s)
            continue
        if s.get("type") != "tool_call":
            out.append(s)
            continue

        inlined = s.get("output")
        call_step = {k: v for k, v in s.items() if k != "output"}
        out.append(call_step)
        if inlined is None or inlined == "":
            continue

        parsed: list | str
        if isinstance(inlined, str):
            try:
                parsed = json.loads(inlined)
            except Exception:
                # Tool output that isn't JSON (e.g. ``open`` returning plain
                # text). Keep it as a single text chunk so the LLM still
                # sees content downstream if anything reads ``output``.
                parsed = [{"text": inlined}]
        else:
            parsed = inlined

        out.append(
            {
                "type": "tool_result",
                "tool_name": s.get("tool_name"),
                "output": parsed,
            }
        )
    return out


def _convert(record: dict) -> dict:
    steps_in = record.get("result") or []
    steps_out = _normalise_steps(list(steps_in))
    n_tool_calls = sum(
        1 for s in steps_out if isinstance(s, dict) and s.get("type") == "tool_call"
    )
    counts = dict(record.get("tool_call_counts") or {})

    boxed = record.get("prediction_boxed") or record.get("prediction_exact") or ""
    return {
        "metadata": record.get("metadata") or {},
        "query_id": str(record.get("query_id", "")),
        "question": str(record.get("question", "")),
        "golden_answers": list(record.get("golden_answers") or []),
        "benchmark": str(record.get("benchmark", "")),
        "tool_call_counts": counts,
        "status": str(record.get("status", "completed")),
        "retrieved_docids": list(record.get("retrieved_docids") or []),
        "result": steps_out,
        "prediction_raw": str(record.get("prediction_raw", "")),
        # Carry the value under BOTH keys so v3 (reads prediction_exact)
        # and v3_boxed (re-extracts from prediction_raw) both work.
        "prediction_exact": str(boxed),
        "prediction_boxed": str(boxed),
        # rejudge_rollouts.py reads `metrics` if present, falls back
        # otherwise; we provide it so search_count / turns aren't
        # silently zeroed by the fallback path.
        "metrics": {
            "search_count": sum(counts.values()),
            "turns": n_tool_calls,
        },
        "r_idx": int(record.get("r_idx", 0) or 0),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__ or "")
    ap.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing run_*.json files "
        "(e.g. runs/hotpotqa/gptoss_r8b/hiexp_e2_e1).",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output jsonl path. Default: <input_dir>/rollouts.jsonl.",
    )
    ap.add_argument(
        "--force", action="store_true", help="Rebuild even if output exists."
    )
    args = ap.parse_args()

    if not args.input_dir.is_dir():
        print(f"[fatal] not a directory: {args.input_dir}", file=sys.stderr)
        return 2

    out_path = args.output or (args.input_dir / "rollouts.jsonl")
    if out_path.exists() and not args.force:
        n = sum(1 for _ in out_path.open("r", encoding="utf-8") if _.strip())
        print(f"[skip] {out_path} exists with {n} rows (use --force to rebuild).")
        return 0

    paths = sorted(args.input_dir.glob("run_*.json"))
    if not paths:
        print(f"[fatal] no run_*.json found in {args.input_dir}", file=sys.stderr)
        return 2

    print(f"[bundle] {len(paths)} run_*.json from {args.input_dir}")
    n_written = 0
    n_failed = 0
    with out_path.open("w", encoding="utf-8") as f:
        for p in paths:
            try:
                rec = json.loads(p.read_text(encoding="utf-8"))
                f.write(json.dumps(_convert(rec), ensure_ascii=False))
                f.write("\n")
                n_written += 1
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] {p.name}: {exc}", file=sys.stderr)
                n_failed += 1

    print(f"[done] wrote {n_written} records to {out_path} ({n_failed} failed)")
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
