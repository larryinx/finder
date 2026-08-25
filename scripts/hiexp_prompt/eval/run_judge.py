"""LLM-as-judge for BCP-shaped run JSONs.

Mirrors ``third_party/BrowseComp-Plus/scripts_evaluation/evaluate_with_openai.py``:

* Reads each ``<input_dir>/*.json`` produced by BCP's ``qwen_client_async.py``
  or our ``scripts/hiexp_prompt/eval/run_eval.py``.
* Looks up the gold question + answer in the decrypted JSONL keyed by
  ``query_id``.
* Renders BCP's ``GRADER_TEMPLATE`` and asks the configured LLM whether
  ``response`` matches ``correct_answer``.
* Parses ``extracted_final_answer / reasoning / correct / confidence`` out
  of the judge text using BCP's regex contract.
* Optionally computes citation precision/recall against
  ``qrel_evidence.txt`` and retrieval recall from ``retrieved_docids``.
* Writes per-run ``<query_id>_eval.json``,
  ``evaluation_summary.json``, and ``detailed_judge_results.csv`` —
  same field names as BCP, so any downstream BCP tooling
  (``analyze_results.py``, leaderboard scripts) Just Works on our outputs.

The judge LLM defaults to ``AGENT_LLM_*`` from ``.env``, but
``--judge-model`` / ``--judge-base-url`` let you point at a stronger
external model (the BCP paper uses ``gpt-4.1`` and similar).

Usage:
    python scripts/hiexp_prompt/eval/run_judge.py \\
        --input-dir runs/browsecomp_plus/qwen3-122b/baseline \\
        --output-dir evals/browsecomp_plus/qwen3-122b/baseline
"""

from __future__ import annotations

import asyncio
import csv
import json
import os
import threading
import time
from collections import defaultdict
from datetime import date
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

from hiexp.evaluation.judge import (
    calibration_error_l2,
    compute_citation_metrics,
    extract_citations_from_response,
    parse_judge_response,
    render_grader_prompt,
)
from hiexp_prompt._bootstrap import bootstrap
from hiexp_prompt._llm_utils import achat_once, make_chat_model

_PROJECT_ROOT = Path(__file__).resolve().parents[3]

bootstrap()

app = typer.Typer(add_completion=False)
console = Console()


# ── BCP defaults ─────────────────────────────────────────────────────────


_DEFAULT_GROUND_TRUTH = (
    _PROJECT_ROOT / "data/BCP/data/browsecomp_plus_decrypted.jsonl"
)
_DEFAULT_QREL_EVIDENCE = (
    _PROJECT_ROOT / "data/BCP/topics-qrels-200/qrel_evidence.txt"
)


# Grader template + parsing helpers come from ``hiexp.evaluation.judge``
# (single source of truth). We keep this script as the *post-hoc* batch
# grader: read run_*.json, call the judge, write *_eval.json + summary.
#
# For *inline* judging during evaluation, see
# ``scripts/hiexp_prompt/eval/run_eval.py`` (browsecomp_plus benchmark)
# and during HEK rollouts see ``sample_rollouts.py``.


# ── I/O ──────────────────────────────────────────────────────────────────


def load_ground_truth(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            qid = str(obj.get("query_id"))
            out[qid] = {
                "question": obj.get("query") or obj.get("question") or "",
                "answer": obj.get("answer") or "",
            }
    return out


def load_qrel(path: Path) -> dict[str, list[str]]:
    qrel: dict[str, list[str]] = defaultdict(list)
    if not path.exists():
        return dict(qrel)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 4:
                continue
            qid, _, docid, _ = parts
            qrel[qid].append(docid)
    return dict(qrel)


def extract_response_text(run: dict) -> str:
    """BCP convention: the agent's final answer is the *last* output_text item."""
    items = run.get("result") or []
    if items and items[-1].get("type") == "output_text":
        return items[-1].get("output") or ""
    return ""


# ── Driver ───────────────────────────────────────────────────────────────


@app.command()
def main(
    input_dir: Path = typer.Option(..., help="Directory of run_*.json files."),
    output_dir: Path = typer.Option(..., help="Where to write *_eval.json + summaries."),
    ground_truth: Path = typer.Option(
        _DEFAULT_GROUND_TRUTH,
        help="Decrypted JSONL with query_id/query/answer (default: data/BCP/...).",
    ),
    qrel_evidence: Path = typer.Option(
        _DEFAULT_QREL_EVIDENCE,
        help="TREC qrel positives file used for citation/retrieval recall.",
    ),
    judge_model: str | None = typer.Option(
        None, help="Override EVAL_JUDGE_MODEL / AGENT_LLM_MODEL."
    ),
    judge_base_url: str | None = typer.Option(
        None, help="Override EVAL_JUDGE_BASE_URL / AGENT_LLM_BASE_URL."
    ),
    judge_api_key: str | None = typer.Option(
        None, help="Override EVAL_JUDGE_API_KEY / AGENT_LLM_API_KEY."
    ),
    max_output_tokens: int = typer.Option(1024),
    temperature: float = typer.Option(0.0, help="Judge sampling temperature."),
    concurrency: int = typer.Option(
        8,
        envvar=["EVAL_JUDGE_CONCURRENCY", "AGENT_LLM_CONCURRENCY"],
        help="Concurrent judge calls in flight.",
    ),
    force: bool = typer.Option(False, help="Re-judge runs even if *_eval.json exists."),
    pattern: str = typer.Option(
        "run_*.json", help="Glob inside input_dir to pick up run files."
    ),
) -> None:
    if not input_dir.is_dir():
        raise typer.BadParameter(f"input_dir {input_dir} is not a directory")
    if not ground_truth.is_file():
        raise typer.BadParameter(f"ground truth file not found: {ground_truth}")
    output_dir.mkdir(parents=True, exist_ok=True)

    judge_model = (
        judge_model or os.getenv("EVAL_JUDGE_MODEL") or os.getenv("AGENT_LLM_MODEL")
    )
    judge_base_url = (
        judge_base_url
        or os.getenv("EVAL_JUDGE_BASE_URL")
        or os.getenv("AGENT_LLM_BASE_URL")
    )
    judge_api_key = (
        judge_api_key
        or os.getenv("EVAL_JUDGE_API_KEY")
        or os.getenv("AGENT_LLM_API_KEY")
        or "EMPTY"
    )
    if not judge_model or not judge_base_url:
        raise typer.BadParameter(
            "judge LLM is not configured. Set AGENT_LLM_* (or EVAL_JUDGE_*) "
            "in .env, or pass --judge-model / --judge-base-url."
        )

    run_files = sorted(input_dir.glob(pattern))
    if not run_files:
        console.print(f"[yellow]no run files matching {pattern!r} in {input_dir}[/yellow]")
        return

    console.print(
        f"[bold]Judge[/bold] model={judge_model} base={judge_base_url} "
        f"runs={len(run_files)} concurrency={concurrency}"
    )

    gt = load_ground_truth(ground_truth)
    qrel = load_qrel(qrel_evidence)
    console.print(
        f"[bold]Sources[/bold] gt={len(gt)} qids  qrel={len(qrel)} qids ({qrel_evidence.name})"
    )

    llm = make_chat_model(
        model=judge_model,
        base_url=judge_base_url,
        api_key=judge_api_key,
        temperature=temperature,
        top_p=1.0,
        max_tokens=max_output_tokens,
    )

    write_lock = threading.Lock()
    all_results: list[dict] = []
    all_results_lock = threading.Lock()

    def _run_to_eval_path(run_path: Path) -> Path:
        return output_dir / f"{run_path.stem}_eval.json"

    def _build_skip_record(run_path: Path, run: dict, retrieval_recall: float) -> dict:
        return {
            "json_path": str(run_path),
            "query_id": run.get("query_id"),
            "response": extract_response_text(run),
            "correct_answer": gt.get(str(run.get("query_id")), {}).get("answer", ""),
            "is_completed": run.get("status") == "completed",
            "judge_prompt": None,
            "judge_response": None,
            "judge_result": {
                "parse_error": True,
                "error": "Response incomplete or cannot be parsed",
            },
            "tool_call_counts": run.get("tool_call_counts", {}),
            "citations": None,
            "retrieval": {
                "recall": retrieval_recall,
                "retrieved_docids": sorted(set(run.get("retrieved_docids") or [])),
            },
            "model_info": {
                "judge_model": judge_model,
                "max_output_tokens": max_output_tokens,
            },
        }

    def _persist(eval_path: Path, record: dict) -> None:
        with write_lock:
            eval_path.write_text(
                json.dumps(record, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    async def _judge_one(run_path: Path) -> tuple[Path, dict | None, Exception | None]:
        eval_path = _run_to_eval_path(run_path)
        if eval_path.exists() and not force:
            try:
                existing = json.loads(eval_path.read_text(encoding="utf-8"))
                return run_path, existing, None
            except Exception:
                pass

        try:
            run = json.loads(run_path.read_text(encoding="utf-8"))
        except Exception as exc:
            return run_path, None, exc

        qid = str(run.get("query_id") or "")
        gt_row = gt.get(qid)
        if gt_row is None:
            return run_path, None, RuntimeError(f"no ground truth for query_id={qid!r}")

        question = gt_row["question"]
        correct_answer = gt_row["answer"]
        is_completed = run.get("status") == "completed"
        response = extract_response_text(run)

        retrieved = set(run.get("retrieved_docids") or [])
        positives = qrel.get(qid, [])
        if positives:
            retrieval_recall = len(retrieved & set(positives)) / float(len(positives))
        else:
            retrieval_recall = 0.0

        if not response or not is_completed:
            record = _build_skip_record(run_path, run, retrieval_recall)
            _persist(eval_path, record)
            return run_path, record, None

        judge_prompt = render_grader_prompt(
            question=question, response=response, correct_answer=correct_answer
        )
        try:
            judge_text = await achat_once(llm, judge_prompt)
        except Exception as exc:
            return run_path, None, exc

        judge_result = parse_judge_response(judge_text)
        cited = extract_citations_from_response(response)
        citation_metrics = compute_citation_metrics(cited, positives)

        record = {
            "json_path": str(run_path),
            "query_id": run.get("query_id"),
            "question": question,
            "response": response,
            "correct_answer": correct_answer,
            "is_completed": is_completed,
            "judge_prompt": judge_prompt,
            "judge_response": judge_text,
            "judge_result": judge_result,
            "tool_call_counts": run.get("tool_call_counts", {}),
            "citations": {
                "cited_docids": cited,
                "metrics": citation_metrics,
            },
            "retrieval": {
                "retrieved_docids": sorted(retrieved),
                "recall": retrieval_recall,
            },
            "model_info": {
                "judge_model": judge_model,
                "max_output_tokens": max_output_tokens,
            },
        }
        _persist(eval_path, record)
        return run_path, record, None

    async def _drive() -> None:
        sem = asyncio.Semaphore(max(1, concurrency))

        async def _bounded(p: Path):
            async with sem:
                return await _judge_one(p)

        tasks = [asyncio.create_task(_bounded(p)) for p in run_files]
        with Progress(
            SpinnerColumn(),
            *Progress.get_default_columns(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task_id = progress.add_task("judge", total=len(tasks))
            for fut in asyncio.as_completed(tasks):
                run_path, record, err = await fut
                progress.update(task_id, advance=1)
                if err is not None:
                    console.print(f"[red]error[/red] {run_path.name}: {err!r}")
                    continue
                if record is not None:
                    with all_results_lock:
                        all_results.append(record)

    start = time.time()
    asyncio.run(_drive())

    if not all_results:
        console.print("[yellow]no judge results produced; nothing to summarize[/yellow]")
        return

    # ── Summary (BCP-shape) ─────────────────────────────────────────────
    detected_model: str | None = None
    try:
        first = json.loads(run_files[0].read_text(encoding="utf-8"))
        meta = first.get("metadata") or {}
        if isinstance(meta, dict):
            detected_model = str(meta.get("model") or "") or None
    except Exception:
        pass

    tool_counts: dict[str, float] = defaultdict(float)
    for r in all_results:
        for k, v in (r.get("tool_call_counts") or {}).items():
            tool_counts[k] += float(v)
    n = len(all_results)
    avg_tool_stats = {k: v / n for k, v in tool_counts.items()}

    confidences: list[float] = []
    correctness: list[bool] = []
    missing_conf = 0
    for r in all_results:
        jr = r.get("judge_result") or {}
        if jr.get("parse_error") or jr.get("correct") is None:
            continue
        if jr.get("confidence") is None:
            missing_conf += 1
            continue
        confidences.append(float(jr["confidence"]))
        correctness.append(bool(jr["correct"]))
    if missing_conf:
        console.print(
            f"[yellow]{missing_conf}/{n} results missing judge confidence; "
            "calibration excludes them.[/yellow]"
        )
    cal_err_pct = (
        round(calibration_error_l2(confidences, correctness), 2)
        if len(confidences) >= 100
        else 0.0
    )

    correct_n = sum(1 for r in all_results if (r.get("judge_result") or {}).get("correct"))
    accuracy_pct = round(100.0 * correct_n / n, 2)

    retrieval_recalls = [
        r.get("retrieval", {}).get("recall", 0.0)
        for r in all_results
        if qrel.get(str(r.get("query_id")))
    ]
    retrieval_recall_pct = (
        round(100.0 * sum(retrieval_recalls) / len(retrieval_recalls), 2)
        if retrieval_recalls
        else None
    )

    per_query = []
    for r in all_results:
        qid = r.get("query_id")
        rec = r.get("retrieval", {}).get("recall")
        per_query.append(
            {
                "query_id": qid,
                "correct": bool((r.get("judge_result") or {}).get("correct", False)),
                "recall": (round(rec * 100.0, 2) if isinstance(rec, (int, float)) else None),
            }
        )

    cited_results = [
        r for r in all_results
        if isinstance(r.get("citations"), dict)
        and r["citations"].get("cited_docids")
    ]
    coverage_pct = round(100.0 * len(cited_results) / n, 2)
    avg_citations = (
        sum(len(r["citations"]["cited_docids"]) for r in cited_results) / len(cited_results)
        if cited_results
        else 0.0
    )
    cit_precision = (
        sum(r["citations"]["metrics"].get("precision", 0.0) for r in cited_results)
        / len(cited_results)
        if cited_results
        else 0.0
    )
    cit_recall = (
        sum(r["citations"]["metrics"].get("recall", 0.0) for r in cited_results)
        / len(cited_results)
        if cited_results
        else 0.0
    )

    summary = {
        "LLM": detected_model or "change me when submitting",
        "Accuracy (%)": accuracy_pct,
        "Recall (%)": retrieval_recall_pct,
        "avg_tool_stats": avg_tool_stats,
        "Calibration Error (%)": cal_err_pct,
        "Retriever": "change me when submitting",
        "Link": "change me when submitting",
        "Evaluation Date": date.today().isoformat(),
        "per_query_metrics": per_query,
        "judge_model": judge_model,
        "n_runs": n,
        "n_correct": correct_n,
        "citations": {
            "coverage_pct": coverage_pct,
            "avg_citations_per_response": round(avg_citations, 2),
            "precision_pct": round(cit_precision * 100.0, 2),
            "recall_pct": round(cit_recall * 100.0, 2),
        },
        "wall_time_seconds": round(time.time() - start, 1),
    }
    summary_path = output_dir / "evaluation_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    csv_path = output_dir / "detailed_judge_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "query_id",
                "predicted_answer",
                "correct_answer",
                "judge_correct",
                "confidence",
                "is_completed",
                "parse_error",
                "json_path",
                "num_citations",
                "precision_positives",
                "recall_positives",
            ],
        )
        writer.writeheader()
        for r in all_results:
            jr = r.get("judge_result") or {}
            cit = r.get("citations") if isinstance(r.get("citations"), dict) else {}
            metrics = (cit.get("metrics") if cit else None) or {}
            predicted = jr.get("extracted_final_answer") or ""
            if not predicted:
                resp = r.get("response") or ""
                predicted = resp[:200] + "..." if len(resp) > 200 else resp
            writer.writerow(
                {
                    "query_id": r.get("query_id", ""),
                    "predicted_answer": predicted,
                    "correct_answer": r.get("correct_answer", ""),
                    "judge_correct": jr.get("correct", ""),
                    "confidence": jr.get("confidence", ""),
                    "is_completed": r.get("is_completed", ""),
                    "parse_error": jr.get("parse_error", False),
                    "json_path": r.get("json_path", ""),
                    "num_citations": len(cit.get("cited_docids", []) if cit else []),
                    "precision_positives": metrics.get("precision", 0),
                    "recall_positives": metrics.get("recall", 0),
                }
            )

    console.print()
    console.print(
        f"[bold green]✓[/bold green] N={n}  "
        f"Accuracy={accuracy_pct:.2f}%  "
        + (f"RetrievalRecall={retrieval_recall_pct:.2f}%  " if retrieval_recall_pct is not None else "")
        + f"CalErr={cal_err_pct:.2f}%"
    )
    console.print(f"  summary → {summary_path}")
    console.print(f"  csv     → {csv_path}")


if __name__ == "__main__":
    app()
