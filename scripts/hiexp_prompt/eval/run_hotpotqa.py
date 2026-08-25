"""Evaluate HiExpAgent on a sampled HotpotQA dev split.

Reproduces the protocol from HiExp paper §A:
    "we randomly sample 1,000 examples from their respective dev sets as
    our final test set, with a fixed random seed 42"

For each example, runs ``HiExpAgent.run(question, return_trajectory=True)``,
extracts the boxed answer with ``hiexp.evaluation.extract_boxed_answer``,
scores it with the Search-o1-compatible F1/CEM/EM metrics
(``hiexp.evaluation.metrics``), and writes:

    <output_dir>/run_<query_id>.json    # per-example trajectory + metrics
    <output_dir>/metrics.json           # aggregate F1 / CEM / EM / N

Toggle ``--enable-hek/--no-enable-hek`` to compare Search-o1* baseline
vs Search-o1* + HiExp.

Prerequisites:
  * A vLLM instance serving the LLM (Qwen2.5-7B-Instruct by default)
    with tool-calling enabled (``--enable-auto-tool-choice
    --tool-call-parser hermes``).
  * An MCP search server reachable at ``AGENT_MCP_URL``.
  * (When ``--enable-hek``) a built HEK directory at ``AGENT_HEK_PATH``.

Usage:
    python scripts/hiexp_prompt/eval/run_hotpotqa.py \\
        --num-samples 1000 \\
        --output-dir runs/hotpotqa/qwen2.5-7b/hiexp \\
        --enable-hek
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

from hiexp import HiExpAgent, HiExpConfig, compose
from hiexp.evaluation import (
    aggregate,
    extract_boxed_answer,
    score_one,
)
from hiexp_prompt._shared import (
    JsonEncoder,
    load_hotpotqa_subset,
    trajectory_to_dict,
)

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(
    num_samples: int = typer.Option(1000, help="Sample N examples (paper §A: 1000)."),
    seed: int = typer.Option(42, help="Sampling seed (paper §A: 42)."),
    split: str = typer.Option("dev", help="HotpotQA split (paper uses dev)."),
    output_dir: Path = typer.Option(
        ...,
        help="Where to write run_*.json and metrics.json.",
    ),
    enable_hek: bool = typer.Option(
        True,
        help="Set --no-enable-hek to evaluate the Search-o1* baseline.",
    ),
    hek_path: Path | None = typer.Option(
        None,
        help="HEK directory. Required when --enable-hek; defaults to AGENT_HEK_PATH.",
    ),
    question_template: str = typer.Option(
        "hiexp_boxed",
        help="One of: hiexp_boxed, browsecomp_plus, raw.",
    ),
    llm_model: str | None = typer.Option(None, help="Override AGENT_LLM_MODEL."),
    llm_base_url: str | None = typer.Option(None, help="Override AGENT_LLM_BASE_URL."),
    mcp_url: str | None = typer.Option(None, help="Override AGENT_MCP_URL."),
    max_search_limit: int = typer.Option(10, help="Search-o1 multi-hop default."),
    max_turn: int = typer.Option(15, help="Search-o1 multi-hop default."),
    resume: bool = typer.Option(
        True,
        help="Skip query_ids already present as run_*.json in output_dir.",
    ),
    dry_run: bool = typer.Option(
        False,
        help="Load dataset and exit without invoking the agent.",
    ),
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    yaml_cfg = compose(family="hiexp_prompt", benchmark="hotpotqa")
    overrides: dict = {
        "question_template": question_template,
        "max_search_limit": max_search_limit,
        "max_turn": max_turn,
        "enable_hek": enable_hek,
    }
    if llm_model:
        overrides["llm_model"] = llm_model
    if llm_base_url:
        overrides["llm_base_url"] = llm_base_url
    if mcp_url:
        overrides["mcp_url"] = mcp_url
    if hek_path is not None:
        overrides["hek_path"] = str(hek_path)

    cfg = HiExpConfig.from_omegaconf(yaml_cfg, **overrides)
    cfg.validate()
    console.print(
        f"[bold]Config[/bold] llm={cfg.llm_model} base={cfg.llm_base_url} "
        f"mcp={cfg.mcp_url} hek={'on@'+str(cfg.hek_path) if cfg.enable_hek else 'off'}"
    )

    examples = load_hotpotqa_subset(
        split=split, num_samples=num_samples, seed=seed
    )
    console.print(f"[bold]Loaded[/bold] {len(examples)} examples ({split} split)")

    if dry_run:
        console.print("[yellow]Dry run[/yellow] — exiting before agent init.")
        for ex in examples[:3]:
            console.print(f"  {ex.query_id}: {ex.question[:80]} → {ex.golden_answers}")
        return

    done_ids: set[str] = set()
    if resume:
        for p in output_dir.glob("run_*.json"):
            try:
                done_ids.add(json.loads(p.read_text(encoding="utf-8"))["query_id"])
            except Exception:
                continue
    pending = [ex for ex in examples if ex.query_id not in done_ids]
    console.print(f"[bold]Pending[/bold] {len(pending)} (skipping {len(done_ids)} already done)")

    agent = HiExpAgent(cfg)

    metrics_path = output_dir / "metrics.json"
    start = time.time()

    with Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("HotpotQA eval", total=len(pending))
        for ex in pending:
            try:
                raw, traj = agent.run(ex.question, return_trajectory=True)
            except Exception as exc:  # noqa: BLE001
                console.print(f"[red]error[/red] qid={ex.query_id}: {exc}")
                progress.update(task, advance=1)
                continue

            pred = extract_boxed_answer(raw)
            metric = score_one(pred, ex.golden_answers)

            record = trajectory_to_dict(traj)
            record["query_id"] = ex.query_id
            record["question"] = ex.question
            record["golden_answers"] = ex.golden_answers
            record["prediction_raw"] = raw
            record["prediction_boxed"] = pred
            record["metrics"] = {"em": metric.em, "cem": metric.cem, "f1": metric.f1}

            run_file = output_dir / f"run_{ex.query_id}.json"
            run_file.write_text(
                json.dumps(record, ensure_ascii=False, indent=2, cls=JsonEncoder),
                encoding="utf-8",
            )
            progress.update(task, advance=1)

    # Aggregate over everything in output_dir (including resumed runs).
    all_metrics = []
    for p in output_dir.glob("run_*.json"):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            m = d.get("metrics", {})
            from hiexp.evaluation.metrics import QaMetric

            all_metrics.append(QaMetric(em=m.get("em", 0), cem=m.get("cem", 0), f1=m.get("f1", 0.0)))
        except Exception:
            continue
    summary = aggregate(all_metrics)
    summary["enable_hek"] = cfg.enable_hek
    summary["hek_path"] = str(cfg.hek_path) if cfg.enable_hek else None
    summary["llm_model"] = cfg.llm_model
    summary["question_template"] = cfg.question_template
    summary["seed"] = cfg.seed
    summary["wall_time_seconds"] = round(time.time() - start, 1)
    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    console.print()
    console.print("[bold green]✓ done[/bold green]")
    console.print(
        f"  N={summary['n']}  F1={summary['f1']:.4f}  CEM={summary['cem']:.4f}  EM={summary['em']:.4f}"
    )
    console.print(f"  metrics → {metrics_path}")


if __name__ == "__main__":
    app()
