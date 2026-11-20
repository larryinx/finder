"""Phase 1 of HEK construction: sample K rollouts per training example.

Implements paper §3.1.1:
    "For each sample x_i in the training set, we execute K independent
    rollouts to obtain a trajectory set Y_i. ... Guided by the outcome
    reward r_orm, we partition these trajectories into successful paths
    Y_i^+ and failed paths Y_i^-."

Generic over benchmarks via :mod:`hiexp.evaluation.benchmarks` — the
spec carries the question template, answer extractor, and scorer kind
(``word_level`` for HotpotQA et al.; ``llm_judge`` for BrowseComp-Plus).
The same :class:`HiExpAgent` that's used at eval time is used here for
rollouts, so distilled E1/E2 patterns transfer cleanly.

Output is a JSONL (or sharded JSONLs when ``--shard-size`` > 0):

    {"query_id", "question", "golden_answers",
     "k", "scorer_kind", "decisive_metric",
     "n_pos", "n_neg",
     "pos": [traj_dict, ...], "neg": [traj_dict, ...]}

Each ``traj_dict`` carries ``prediction_raw``, the spec's
``extractor_field`` (e.g. ``prediction_boxed``), ``rollout_index``, and
the scorer's full return dict under ``scoring`` (``correct``, plus
metric-specific fields like ``cem`` or ``judge_confidence``).

Examples already present in the output are skipped on resume.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from collections import defaultdict
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

from hiexp import HiExpAgent, HiExpConfig, compose, init_weave, list_options
from hiexp.tracing import finish as _finish_tracing
from hiexp.evaluation import (
    BenchmarkSpec,
    Scorer,
    get_spec,
    make_scorer,
)
from hiexp_prompt._bootstrap import bootstrap
from hiexp_prompt._llm_utils import make_chat_model
from hiexp_prompt._shared import (
    JsonEncoder,
    QaExample,
    load_browsecomp_plus_jsonl,
    load_browsecomp_plus_split,
    load_flashrag_split,
    load_hotpotqa_subset,
    load_paper_hek_training_mix,
    trajectory_to_dict,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]

bootstrap()

app = typer.Typer(add_completion=False)
console = Console()


# ── Default training data sources ────────────────────────────────────────


def _load_training_examples(
    *,
    cfg,
    cap: int | None,
    seed: int,
    bcp_jsonl: Path | None,
) -> list[QaExample]:
    """Resolve the HEK training source from ``cfg`` + a CLI override.

    Dispatch is benchmark-driven (no separate "training_data" tag):

    * ``benchmark.name == 'browsecomp_plus'``:
      - if ``--bcp-jsonl`` is set, load that custom JSONL;
      - else load ``data.splits.train`` (a queries.tsv) joined against
        ``data.decrypted_jsonl``.
    * ``benchmark.name == 'hotpotqa'``: load via HuggingFace, keyed by
      ``hek.hf_source`` (``hotpotqa`` / ``2wikimultihopqa`` / ``musique``
      / ``paper_mix`` / ``paper_mix_approx``).
    """
    name = cfg.benchmark.name

    if name == "browsecomp_plus":
        if bcp_jsonl is not None:
            if not bcp_jsonl.exists():
                raise typer.BadParameter(f"--bcp-jsonl path missing: {bcp_jsonl}")
            full = load_browsecomp_plus_jsonl(bcp_jsonl)
            return full if cap is None else full[:cap]
        train_path = cfg.data.splits.get("train")
        if not train_path:
            raise typer.BadParameter(
                "BCP HEK build needs either data.splits.train (queries.tsv) "
                "or --bcp-jsonl <path>."
            )
        qtsv = _PROJECT_ROOT / train_path
        djsonl = _PROJECT_ROOT / cfg.data.decrypted_jsonl
        if not qtsv.exists() or not djsonl.exists():
            raise typer.BadParameter(
                f"BCP train sources missing: {qtsv} or {djsonl}. "
                "Run `bash scripts/data_prep/prepare.sh hiexp` first."
            )
        full = load_browsecomp_plus_split(queries_tsv=qtsv, decrypted_jsonl=djsonl)
        return full if cap is None else full[:cap]

    if name == "hotpotqa":
        hf_source = cfg.hek.get("hf_source")
        if not hf_source:
            raise typer.BadParameter(
                "HotpotQA HEK build needs hek.hf_source set in the YAML "
                "(e.g. paper_mix / paper_mix_approx / hotpotqa / "
                "2wikimultihopqa / musique)."
            )
        if hf_source == "hotpotqa":
            return load_hotpotqa_subset(split="train", num_samples=cap, seed=seed)
        if hf_source in {"2wikimultihopqa", "musique"}:
            return load_flashrag_split(hf_source, split="train", num_samples=cap, seed=seed)
        if hf_source == "paper_mix":
            full = load_paper_hek_training_mix(seed=seed, use_r1_subset=True)
            return full if cap is None else full[:cap]
        if hf_source == "paper_mix_approx":
            full = load_paper_hek_training_mix(seed=seed, use_r1_subset=False)
            return full if cap is None else full[:cap]
        raise typer.BadParameter(f"unknown hek.hf_source: {hf_source!r}")

    raise typer.BadParameter(f"no training-data dispatch for benchmark={name!r}")


# ── Sharded JSONL writer ─────────────────────────────────────────────────


class ShardedJsonlWriter:
    """Append-mode JSONL writer that rolls over after ``shard_size`` records.

    ``shard_size <= 0`` means "single file" (writes to ``out_dir/rollouts.jsonl``).
    Otherwise files are ``rollouts-00000.jsonl``, ``rollouts-00001.jsonl``, …
    Resume-safe: on init we count records in existing shards so the next write
    continues into the current (possibly partial) shard rather than reopening
    a fresh one.
    """

    _SINGLE_NAME = "rollouts.jsonl"

    def __init__(self, out_dir: Path, shard_size: int) -> None:
        self.out_dir = out_dir
        self.shard_size = shard_size
        self._lock = threading.Lock()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._records_in_current_shard = 0
        self._shard_index = 0

        if shard_size <= 0:
            self._path = out_dir / self._SINGLE_NAME
            self._handle = self._path.open("a", encoding="utf-8")
            return

        existing = sorted(out_dir.glob("rollouts-*.jsonl"))
        if existing:
            last = existing[-1]
            self._shard_index = int(last.stem.split("-")[-1])
            self._records_in_current_shard = sum(
                1 for ln in last.read_text(encoding="utf-8").splitlines() if ln.strip()
            )
            if self._records_in_current_shard >= shard_size:
                self._shard_index += 1
                self._records_in_current_shard = 0
        self._path = self._shard_path()
        self._handle = self._path.open("a", encoding="utf-8")

    def _shard_path(self) -> Path:
        if self.shard_size <= 0:
            return self.out_dir / self._SINGLE_NAME
        return self.out_dir / f"rollouts-{self._shard_index:05d}.jsonl"

    def write(self, payload: dict) -> None:
        line = json.dumps(payload, ensure_ascii=False, cls=JsonEncoder) + "\n"
        with self._lock:
            self._handle.write(line)
            self._handle.flush()
            self._records_in_current_shard += 1
            if self.shard_size > 0 and self._records_in_current_shard >= self.shard_size:
                self._handle.close()
                self._shard_index += 1
                self._records_in_current_shard = 0
                self._path = self._shard_path()
                self._handle = self._path.open("a", encoding="utf-8")

    def close(self) -> None:
        with self._lock:
            if not self._handle.closed:
                self._handle.close()


def _read_done_query_ids(out_dir: Path) -> set[str]:
    """Union of query_ids across single + sharded rollout files."""
    done: set[str] = set()
    for p in [out_dir / "rollouts.jsonl", *out_dir.glob("rollouts-*.jsonl")]:
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(line)["query_id"])
            except Exception:
                continue
    return done


# ── Judge LLM bootstrap (same convention as run_eval.py) ─────────────────


def _build_judge_llm(*, cfg):
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


# ── Driver ───────────────────────────────────────────────────────────────


async def _drive_rollouts(
    *,
    agent: HiExpAgent,
    spec: BenchmarkSpec,
    scorer: Scorer,
    pending: list[QaExample],
    rollouts_per_example: int,
    concurrency: int,
    writer: ShardedJsonlWriter,
) -> None:
    """Fire ``len(pending) × K`` rollouts with at most ``concurrency`` in flight,
    score each one via ``scorer.ascore``, and flush per-example records once
    all K rollouts of an example have completed.

    Trajectories are accumulated in ``buffers[qid]``; ``remaining[qid]``
    tracks how many rollouts owe results so we know when a record is
    flushable.
    """
    K = rollouts_per_example
    total = len(pending) * K
    if total == 0:
        return

    sem = asyncio.Semaphore(max(1, concurrency))
    buffers: dict[str, list[tuple[int, dict | None]]] = defaultdict(list)
    remaining: dict[str, int] = {ex.query_id: K for ex in pending}
    by_id: dict[str, QaExample] = {ex.query_id: ex for ex in pending}

    async def _one_rollout(ex: QaExample, k: int):
        async with sem:
            try:
                raw, traj = await agent.arun(ex.question, return_trajectory=True)
            except Exception as exc:  # noqa: BLE001
                return ex, k, None, exc
            try:
                rec = trajectory_to_dict(traj)
                pred = spec.extractor(raw)
                rec["prediction_raw"] = raw
                rec[spec.extractor_field] = pred
                rec["rollout_index"] = k
                scoring = await scorer.ascore(
                    question=ex.question,
                    golden_answers=ex.golden_answers,
                    prediction_raw=raw,
                    prediction_extracted=pred,
                )
                rec["scoring"] = scoring
                return ex, k, rec, None
            except Exception as exc:  # noqa: BLE001
                return ex, k, None, exc

    tasks = [
        asyncio.create_task(_one_rollout(ex, k))
        for ex in pending
        for k in range(K)
    ]

    with Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task(
            f"rollouts (×{K} per ex, {concurrency} in-flight, {spec.scorer_kind})",
            total=total,
        )
        for fut in asyncio.as_completed(tasks):
            ex, k, rec, err = await fut
            progress.update(task_id, advance=1)
            qid = ex.query_id

            if err is not None:
                console.print(f"[red]rollout err[/red] qid={qid} k={k}: {err!r}")
                buffers[qid].append((k, None))
            else:
                buffers[qid].append((k, rec))

            remaining[qid] -= 1
            if remaining[qid] > 0:
                continue

            pos: list[dict] = []
            neg: list[dict] = []
            for _, rec_or_none in sorted(buffers[qid], key=lambda x: x[0]):
                if rec_or_none is None:
                    continue
                bucket = pos if (rec_or_none.get("scoring") or {}).get("correct") else neg
                bucket.append(rec_or_none)
            payload = {
                "query_id": qid,
                "question": ex.question,
                "golden_answers": ex.golden_answers,
                "benchmark": spec.name,
                "scorer_kind": spec.scorer_kind,
                "decisive_metric": (
                    spec.decisive_metric if spec.scorer_kind == "word_level" else None
                ),
                "k": K,
                "n_pos": len(pos),
                "n_neg": len(neg),
                "pos": pos,
                "neg": neg,
            }
            writer.write(payload)
            buffers.pop(qid, None)
            by_id.pop(qid, None)


@app.command()
def main(
    out_dir: Path = typer.Option(..., help="Where to write rollouts.jsonl (or shards)."),
    benchmark: str = typer.Option(
        ...,
        help=f"Benchmark YAML stem: {' | '.join(list_options('hiexp_prompt', 'benchmark')) or '(none)'}.",
    ),
    retriever: str | None = typer.Option(
        None,
        help=f"Retriever YAML stem: {' | '.join(list_options('hiexp_prompt', 'retriever')) or '(none)'}.",
    ),
    bcp_jsonl: Path | None = typer.Option(
        None,
        help="When --benchmark=bcp, an optional custom JSONL training source "
        "(otherwise data.splits.train is used).",
    ),
    overrides: list[str] = typer.Option(
        [], "--override", "-o",
        help="OmegaConf dotlist overrides (e.g. hek.num_train=200, "
        "hek.rollouts_per_example=4, concurrency.rollout=32).",
    ),
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

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

    # ── Tracing init ────────────────────────────────────────────────────
    from omegaconf import OmegaConf
    init_weave(
        run_name=f"hek-rollouts-{cfg.benchmark.name}-K{cfg.hek.rollouts_per_example}",
        config=OmegaConf.to_container(cfg, resolve=True),
        tags=[f"benchmark:{cfg.benchmark.name}",
              f"retriever:{retriever}" if retriever else "retriever:none",
              "hiexp_prompt", "phase:rollouts"],
        job_type="hek_build",
        phase="hiexp_rollout",  # gate via WEAVE_DISABLED_HIEXP_ROLLOUT
    )

    # Force-disable HEK for baseline rollouts — we're generating the data
    # the HEK is built from, so we can't have HEK injection bleed in.
    base_cfg = HiExpConfig.from_omegaconf(
        cfg,
        enable_hek=False,
        question_template=spec.question_template,
    )
    base_cfg.validate()

    judge_llm = (
        _build_judge_llm(cfg=cfg) if spec.scorer_kind == "llm_judge" else None
    )
    scorer = make_scorer(spec, judge_llm=judge_llm)

    # HEK-build params from cfg.
    num_train = int(cfg.hek.num_train)
    rollouts_per_example = int(cfg.hek.rollouts_per_example)
    shard_size = int(cfg.hek.trajectory_shard_size)
    seed = int(cfg.misc.seed)
    concurrency = int(cfg.concurrency.rollout or cfg.concurrency.llm)

    source_label = (
        "bcp:--bcp-jsonl" if (cfg.benchmark.name == "browsecomp_plus" and bcp_jsonl)
        else f"bcp:{cfg.data.splits.get('train', '?')}" if cfg.benchmark.name == "browsecomp_plus"
        else f"hf:{cfg.hek.get('hf_source', '?')}"
    )
    console.print(
        f"[bold]Rollouts[/bold] benchmark={cfg.benchmark.name} scorer={spec.scorer_kind} "
        f"K={rollouts_per_example} N_train={num_train} llm={base_cfg.llm_model} "
        f"concurrency={concurrency}"
    )

    cap = None if num_train < 0 else num_train
    examples = _load_training_examples(
        cfg=cfg,
        cap=cap,
        seed=seed,
        bcp_jsonl=bcp_jsonl,
    )
    console.print(f"[bold]Source[/bold] {source_label} → {len(examples)} examples")

    done = _read_done_query_ids(out_dir)
    pending = [ex for ex in examples if ex.query_id not in done]
    console.print(f"[bold]Pending[/bold] {len(pending)} (skipping {len(done)})")

    agent = HiExpAgent(base_cfg)
    writer = ShardedJsonlWriter(out_dir, shard_size=shard_size)
    start = time.time()
    try:
        asyncio.run(
            _drive_rollouts(
                agent=agent,
                spec=spec,
                scorer=scorer,
                pending=pending,
                rollouts_per_example=rollouts_per_example,
                concurrency=concurrency,
                writer=writer,
            )
        )
    finally:
        writer.close()

    written = sorted(out_dir.glob("rollouts*.jsonl"))
    console.print(
        f"[bold green]✓[/bold green] {len(written)} rollout file(s) in {out_dir} "
        f"({round(time.time()-start, 1)}s)"
    )
    _finish_tracing()


if __name__ == "__main__":
    app()
