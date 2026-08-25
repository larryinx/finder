"""Evaluate ``HiExpAgent`` on a benchmark — single-rollout per example.

Generic over benchmarks via :mod:`hiexp.evaluation.benchmarks`. Adding a
new benchmark = registering a ``BenchmarkSpec`` (the spec carries
question template, answer extractor, and scorer kind), plus a dataset
loader in ``src/hiexp_prompt/_shared.py`` and a YAML under
``config/hiexp_prompt/benchmark/``. No code in this file changes for a
new benchmark.

Behavior per scorer kind:

* ``word_level`` (HotpotQA et al.) — computes EM / CEM / F1 inline using
  ``score_one``. Aggregate metric.
* ``llm_judge`` (BrowseComp-Plus) — calls BCP's ``GRADER_TEMPLATE``
  through the judge LLM (``EVAL_JUDGE_*`` env vars, falling back to
  ``AGENT_LLM_*``). Aggregate = accuracy (correct / total).

Per-example output: ``<output_dir>/run_<query_id>.json`` containing
trajectory, raw response, extracted answer, and the scorer's full
return dict under ``scoring``. Aggregate summary under
``<output_dir>/metrics.json``.

Usage:
    # HotpotQA
    python scripts/hiexp_prompt/eval/run_eval.py \\
        --benchmark hotpotqa --retriever qwen3-8b \\
        --output-dir runs/hotpotqa/qwen2.5-7b/hiexp \\
        --enable-hek --hek-path data/hek/paper_mix/qwen9b

    # BrowseComp-Plus (judge runs inline)
    python scripts/hiexp_prompt/eval/run_eval.py \\
        --benchmark bcp --retriever qwen3-8b \\
        --output-dir runs/bcp/qwen2.5-7b/baseline
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

from hiexp import HiExpAgent, HiExpConfig, compose, init_weave, list_options
from hiexp.tracing import finish as _finish_tracing
from hiexp.evaluation import (
    BenchmarkSpec,
    QaMetric,
    aggregate,
    get_spec,
    make_scorer,
)
from hiexp_prompt._bootstrap import bootstrap
from hiexp_prompt._llm_utils import make_chat_model
from hiexp_prompt._shared import (
    JsonEncoder,
    QaExample,
    load_browsecomp_plus_split,
    load_hotpotqa_subset,
    trajectory_to_dict,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]

bootstrap()

app = typer.Typer(add_completion=False)
console = Console()


def _load_examples(
    *,
    cfg,
    num_samples: int,
    seed: int,
    split: str,
    input_jsonl: Path | None,
    bcp_queries_tsv: Path | None,
    bcp_decrypted_jsonl: Path | None,
) -> list[QaExample]:
    """Dispatch to the right loader. The benchmark's *spec* doesn't bundle
    loaders (FS layout is a script concern); this is the only place that
    knows where each benchmark's files live.

    File-system paths come from the composed YAML (``cfg.data.*``); CLI
    flags can still override them when iterating against custom data.
    """
    name = cfg.benchmark.name
    yaml_test = cfg.data.splits.get("test") if "splits" in cfg.data else None

    if name == "hotpotqa":
        # YAML default (data.splits.test) takes effect when no CLI
        # override is given. Empty/null means fall back to HF loader.
        if input_jsonl is None and yaml_test:
            candidate = _PROJECT_ROOT / yaml_test
            if candidate.exists():
                input_jsonl = candidate
        if input_jsonl is not None and Path(input_jsonl).exists():
            examples: list[QaExample] = []
            with Path(input_jsonl).open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    examples.append(
                        QaExample(
                            query_id=str(obj["query_id"]),
                            question=str(obj["question"]),
                            golden_answers=list(obj.get("golden_answers") or []),
                            metadata=dict(obj.get("metadata") or {}),
                        )
                    )
            if num_samples and num_samples < len(examples):
                examples = examples[:num_samples]
            return examples
        n = num_samples or int(cfg.data.get("default_num_samples", 1000))
        s = split or str(cfg.data.get("default_split", "dev"))
        return load_hotpotqa_subset(split=s, num_samples=n, seed=seed)

    if name == "browsecomp_plus":
        if not yaml_test and bcp_queries_tsv is None:
            raise typer.BadParameter(
                "BCP requires data.splits.test in the benchmark YAML or "
                "--bcp-queries-tsv on the CLI."
            )
        qtsv = Path(bcp_queries_tsv) if bcp_queries_tsv else _PROJECT_ROOT / yaml_test
        djsonl = (
            Path(bcp_decrypted_jsonl)
            if bcp_decrypted_jsonl
            else _PROJECT_ROOT / cfg.data.decrypted_jsonl
        )
        for required in (qtsv, djsonl):
            if not Path(required).exists():
                raise FileNotFoundError(
                    f"BCP eval source missing: {required} — run "
                    "`bash scripts/data_prep/prepare.sh hiexp` first, or pass "
                    "--bcp-queries-tsv / --bcp-decrypted-jsonl."
                )
        examples = load_browsecomp_plus_split(
            queries_tsv=Path(qtsv), decrypted_jsonl=Path(djsonl)
        )
        if num_samples and num_samples < len(examples):
            examples = examples[:num_samples]
        return examples

    raise typer.BadParameter(
        f"benchmark {name!r} has a registered spec but no loader; "
        "wire one in scripts/hiexp_prompt/eval/run_eval.py:_load_examples"
    )


def _build_judge_llm(*, cfg):
    """Materialize a ChatOpenAI for the judge.

    Endpoint resolution: ``EVAL_JUDGE_*`` env > ``AGENT_LLM_*`` env.
    Sampling knobs (max_tokens / temperature) come from ``cfg.judge``.
    Returns ``None`` if neither model nor base URL is set, in which case
    ``llm_judge`` benchmarks will fail at scorer-build time with a clear
    error.
    """
    model = os.getenv("EVAL_JUDGE_MODEL") or os.getenv("AGENT_LLM_MODEL")
    base_url = os.getenv("EVAL_JUDGE_BASE_URL") or os.getenv("AGENT_LLM_BASE_URL")
    api_key = (
        os.getenv("EVAL_JUDGE_API_KEY")
        or os.getenv("AGENT_LLM_API_KEY")
        or "EMPTY"
    )
    if not model or not base_url:
        return None
    return make_chat_model(
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=float(cfg.judge.temperature),
        top_p=1.0,
        max_tokens=int(cfg.judge.max_tokens),
    )


@app.command()
def main(
    benchmark: str = typer.Option(
        ...,
        help=f"Benchmark YAML stem: {' | '.join(list_options('hiexp_prompt', 'benchmark')) or '(none)'}.",
    ),
    retriever: str | None = typer.Option(
        None,
        help=f"Retriever YAML stem: {' | '.join(list_options('hiexp_prompt', 'retriever')) or '(none)'}.",
    ),
    output_dir: Path = typer.Option(
        ..., help="Where to write run_*.json and metrics.json."
    ),
    input_jsonl: Path | None = typer.Option(
        None,
        help="HotpotQA only: override the eval JSONL path (default from YAML).",
    ),
    bcp_queries_tsv: Path | None = typer.Option(
        None,
        help="BCP only: override the BCP topics TSV (default from YAML).",
    ),
    bcp_decrypted_jsonl: Path | None = typer.Option(
        None,
        help="BCP only: override the BCP decrypted gold-answer JSONL (default from YAML).",
    ),
    num_samples: int = typer.Option(
        0,
        help="Cap the number of examples (0 = use the dataset default).",
    ),
    seed: int = typer.Option(42, help="HotpotQA fallback sampling seed."),
    split: str = typer.Option("", help="HotpotQA fallback split (default from YAML)."),
    enable_hek: bool = typer.Option(
        False, help="Set --enable-hek to evaluate the +HiExp variant."
    ),
    hek_path: Path | None = typer.Option(None, help="HEK directory (when --enable-hek)."),
    resume: bool = typer.Option(
        True, help="Skip query_ids already present as run_*.json in output_dir."
    ),
    dry_run: bool = typer.Option(
        False, help="Load dataset + show first few rows; do not invoke the agent."
    ),
    overrides: list[str] = typer.Option(
        [], "--override", "-o",
        help="OmegaConf dotlist overrides (e.g. agent.max_search_limit=20).",
    ),
) -> None:
    cfg = compose(
        family="hiexp_prompt",
        benchmark=benchmark,
        retriever=retriever,
        overrides=overrides,
    )

    try:
        spec: BenchmarkSpec = get_spec(cfg.benchmark.name)
    except KeyError as exc:
        raise typer.BadParameter(str(exc))

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Tracing init (Weave + W&B) ──────────────────────────────────────
    from omegaconf import OmegaConf
    init_weave(
        run_name=f"hiexp-eval-{cfg.benchmark.name}-{retriever or 'none'}-{'hek' if enable_hek else 'baseline'}",
        config=OmegaConf.to_container(cfg, resolve=True),
        tags=[f"benchmark:{cfg.benchmark.name}",
              f"retriever:{retriever}" if retriever else "retriever:none",
              "hiexp_prompt",
              "hek" if enable_hek else "baseline"],
        job_type="hiexp_eval",
        phase="hiexp_eval",  # gate via WEAVE_DISABLED_HIEXP_EVAL
    )

    base_cfg = HiExpConfig.from_omegaconf(
        cfg,
        enable_hek=enable_hek,
        hek_path=str(hek_path) if hek_path else None,
    )
    base_cfg.validate()

    # Build the scorer. ``llm_judge`` benchmarks need a judge LLM; all
    # others get ``None`` and ``make_scorer`` ignores it.
    judge_llm = (
        _build_judge_llm(cfg=cfg) if spec.scorer_kind == "llm_judge" else None
    )
    scorer = make_scorer(spec, judge_llm=judge_llm)

    concurrency = int(cfg.concurrency.eval or cfg.concurrency.llm)

    console.print(
        f"[bold]Config[/bold] benchmark={cfg.benchmark.name}  "
        f"retriever={cfg.get('retriever', {}).get('name', 'n/a')}  "
        f"scorer={spec.scorer_kind}  "
        f"llm={base_cfg.llm_model}  base={base_cfg.llm_base_url}  mcp={base_cfg.mcp_url}  "
        f"hek={'on@'+str(base_cfg.hek_path) if base_cfg.enable_hek else 'off'}  "
        f"qt={base_cfg.question_template}  concurrency={concurrency}"
    )

    examples = _load_examples(
        cfg=cfg,
        num_samples=num_samples,
        seed=seed,
        split=split,
        input_jsonl=input_jsonl,
        bcp_queries_tsv=bcp_queries_tsv,
        bcp_decrypted_jsonl=bcp_decrypted_jsonl,
    )
    console.print(f"[bold]Loaded[/bold] {len(examples)} examples ({cfg.benchmark.name})")

    if dry_run:
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
    console.print(
        f"[bold]Pending[/bold] {len(pending)} (skipping {len(done_ids)} already done)"
    )

    agent = HiExpAgent(base_cfg)
    metrics_path = output_dir / "metrics.json"
    write_lock = threading.Lock()

    async def _persist(ex: QaExample, raw: str, traj) -> dict:
        pred = spec.extractor(raw)
        record = trajectory_to_dict(traj)
        record["query_id"] = ex.query_id
        record["question"] = ex.question
        record["golden_answers"] = ex.golden_answers
        record["benchmark"] = cfg.benchmark.name
        record["prediction_raw"] = raw
        record[spec.extractor_field] = pred

        scoring = await scorer.ascore(
            question=ex.question,
            golden_answers=ex.golden_answers,
            prediction_raw=raw,
            prediction_extracted=pred,
        )
        record["scoring"] = scoring

        out_path = output_dir / f"run_{ex.query_id}.json"
        payload = json.dumps(record, ensure_ascii=False, indent=2, cls=JsonEncoder)
        with write_lock:
            out_path.write_text(payload, encoding="utf-8")
        return scoring

    async def _drive() -> None:
        sem = asyncio.Semaphore(max(1, concurrency))

        async def _one(ex: QaExample):
            async with sem:
                try:
                    raw, traj = await agent.arun(ex.question, return_trajectory=True)
                except Exception as exc:  # noqa: BLE001
                    return ex, exc
                try:
                    await _persist(ex, raw, traj)
                except Exception as exc:  # noqa: BLE001
                    return ex, exc
                return ex, None

        tasks = [asyncio.create_task(_one(ex)) for ex in pending]
        with Progress(
            SpinnerColumn(),
            *Progress.get_default_columns(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task_id = progress.add_task(
                f"{cfg.benchmark.name} eval ({concurrency} in-flight)", total=len(pending)
            )
            for fut in asyncio.as_completed(tasks):
                ex, err = await fut
                progress.update(task_id, advance=1)
                if err is not None:
                    console.print(f"[red]error[/red] qid={ex.query_id}: {err!r}")

    start = time.time()
    asyncio.run(_drive())

    # Aggregate. Read every run_*.json under output_dir (covers both this
    # run and any pre-existing records that were skipped by --resume).
    records: list[dict] = []
    for p in output_dir.glob("run_*.json"):
        try:
            records.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            continue

    summary: dict = {
        "benchmark": cfg.benchmark.name,
        "retriever": cfg.get("retriever", {}).get("name"),
        "scorer_kind": spec.scorer_kind,
        "enable_hek": base_cfg.enable_hek,
        "hek_path": str(base_cfg.hek_path) if base_cfg.enable_hek else None,
        "llm_model": base_cfg.llm_model,
        "question_template": base_cfg.question_template,
        "seed": base_cfg.seed,
        "wall_time_seconds": round(time.time() - start, 1),
        "n": len(records),
    }

    if spec.scorer_kind == "word_level":
        all_metrics: list[QaMetric] = []
        for d in records:
            s = d.get("scoring") or {}
            all_metrics.append(
                QaMetric(
                    em=int(s.get("em", 0)),
                    cem=int(s.get("cem", 0)),
                    f1=float(s.get("f1", 0.0)),
                )
            )
        summary.update(aggregate(all_metrics))
        console.print()
        console.print(
            f"[bold green]✓[/bold green] N={summary['n']}  "
            f"F1={summary['f1']:.4f}  CEM={summary['cem']:.4f}  EM={summary['em']:.4f}"
        )
    else:
        n_correct = sum(1 for d in records if (d.get("scoring") or {}).get("correct"))
        n = len(records) or 1
        summary.update(
            n_correct=n_correct,
            accuracy=n_correct / n,
        )
        console.print()
        console.print(
            f"[bold green]✓[/bold green] N={summary['n']}  "
            f"Accuracy={summary['accuracy']:.4f} ({n_correct}/{summary['n']})"
        )

    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    console.print(f"  metrics → {metrics_path}")
    _finish_tracing()


if __name__ == "__main__":
    app()
