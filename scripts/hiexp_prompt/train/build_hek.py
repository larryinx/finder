"""End-to-end orchestrator: rollouts → distillation → clustering → HEK.

Wraps the three phase scripts so a single CLI builds a usable HEK
under ``<hek_path>/``. Each phase script can also be run individually
when iterating.

The orchestrator is a thin shell. All algorithmic / experiment-defining
knobs (training source, K, num_train, trajectory_shard_size, cluster
threshold) live in ``config/hiexp_prompt/{base,benchmark/X}.yaml``
under the ``hek`` block. Override on the CLI via
``-o hek.rollouts_per_example=4`` etc.; the orchestrator forwards every
``-o`` to each phase script. CLI here is just for run-specific paths
(``--hek-path``, ``--rollouts-path``, ``--bcp-jsonl``) and debug
toggles (``--skip-phase{1,2,3}``).

Training source is benchmark-driven:

* ``benchmark.name == 'browsecomp_plus'``: read from ``data.splits.train``
  (or ``--bcp-jsonl`` for a custom JSONL).
* ``benchmark.name == 'hotpotqa'``: read via the HuggingFace loader
  keyed by ``hek.hf_source``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import typer
from rich.console import Console

from hiexp import compose, init_weave, list_options, log_hek_artifacts
from hiexp.tracing import finish as _finish_tracing
from hiexp_prompt._bootstrap import bootstrap

bootstrap()

app = typer.Typer(add_completion=False)
console = Console()

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parents[2]


def _run(cmd: list[str]) -> None:
    console.rule(f"[bold]exec[/bold] {' '.join(cmd[1:])}")
    proc = subprocess.run(cmd, cwd=_PROJECT_ROOT, check=False)
    if proc.returncode != 0:
        console.print(f"[red]phase failed[/red] (returncode={proc.returncode})")
        raise typer.Exit(code=proc.returncode)


@app.command()
def main(
    hek_path: Path = typer.Option(..., help="Output HEK directory."),
    rollouts_path: Path | None = typer.Option(
        None,
        help="Where to keep rollouts (file or directory). Defaults to <hek_path>/rollouts/.",
    ),
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
    do_e3: bool = typer.Option(False, help="Also build E3 by clustering E2."),
    skip_phase1: bool = typer.Option(False),
    skip_phase2: bool = typer.Option(False),
    skip_phase3: bool = typer.Option(False),
    overrides: list[str] = typer.Option(
        [], "--override", "-o",
        help="OmegaConf dotlist overrides forwarded to each phase script "
        "(e.g. hek.num_train=200, hek.rollouts_per_example=4).",
    ),
) -> None:
    hek_path.mkdir(parents=True, exist_ok=True)
    if rollouts_path is None:
        rollouts_path = hek_path / "rollouts"
    rollouts_path.mkdir(parents=True, exist_ok=True)

    cfg = compose(
        family="hiexp_prompt",
        benchmark=benchmark,
        retriever=retriever,
        overrides=overrides,
    )

    # ── Tracing init for the orchestrator. Phase scripts (rollouts /
    # distill / cluster) are launched as separate Python processes and
    # each does its own ``init_weave`` — they appear as sibling runs
    # under the same W&B project so the trace history reads naturally.
    from omegaconf import OmegaConf
    init_weave(
        run_name=f"hek-build-{cfg.benchmark.name}-{retriever or 'none'}-{hek_path.name}",
        config=OmegaConf.to_container(cfg, resolve=True),
        tags=[f"benchmark:{cfg.benchmark.name}",
              f"retriever:{retriever}" if retriever else "retriever:none",
              "hiexp_prompt", "phase:orchestrator"],
        job_type="hek_build",
        # phase="hek_build" gates the *orchestrator* (which uploads the
        # final HEK artifact). To skip rollout traces but keep the
        # artifact upload, set WEAVE_DISABLED_HIEXP_ROLLOUT instead —
        # that gates only sample_rollouts.py's process.
        phase="hek_build",
    )

    cluster_threshold = float(cfg.hek.cluster_threshold)
    source = (
        f"local:{cfg.data.splits.get('train', '?')}" if cfg.benchmark.name == "browsecomp_plus"
        else f"hf:{cfg.hek.get('hf_source', '?')}"
    )
    console.print(
        f"[bold]HEK build[/bold] benchmark={cfg.benchmark.name} "
        f"retriever={cfg.get('retriever', {}).get('name', 'n/a')} "
        f"source={source} "
        f"K={cfg.hek.rollouts_per_example} "
        f"num_train={cfg.hek.num_train} "
        f"trajectory_shard_size={cfg.hek.trajectory_shard_size}"
    )

    py = sys.executable
    train_dir = _THIS_DIR
    forwarded_overrides: list[str] = []
    for o in overrides:
        forwarded_overrides.extend(["-o", o])

    if not skip_phase1:
        cmd = [
            py,
            str(train_dir / "sample_rollouts.py"),
            "--out-dir", str(rollouts_path),
            "--benchmark", benchmark,
        ]
        if retriever is not None:
            cmd.extend(["--retriever", retriever])
        if bcp_jsonl is not None:
            cmd.extend(["--bcp-jsonl", str(bcp_jsonl)])
        cmd.extend(forwarded_overrides)
        _run(cmd)
    else:
        console.print("[yellow]skip[/yellow] phase 1 (rollouts)")

    if not skip_phase2:
        _run(
            [
                py,
                str(train_dir / "distill_e1.py"),
                "--rollouts-path", str(rollouts_path),
                "--hek-path", str(hek_path),
            ]
        )
    else:
        console.print("[yellow]skip[/yellow] phase 2 (distill E1)")

    if not skip_phase3:
        _run(
            [
                py,
                str(train_dir / "cluster_e2.py"),
                "--hek-path", str(hek_path),
                "--distance-threshold", str(cluster_threshold),
                "--target-level", "2",
            ]
        )
        if do_e3:
            _run(
                [
                    py,
                    str(train_dir / "cluster_e2.py"),
                    "--hek-path", str(hek_path),
                    "--distance-threshold", str(cluster_threshold),
                    "--target-level", "3",
                ]
            )
    else:
        console.print("[yellow]skip[/yellow] phase 3 (cluster)")

    console.rule("[bold green]HEK build complete[/bold green]")
    for level in (1, 2, 3):
        p = hek_path / f"e{level}.jsonl"
        if p.exists():
            n = sum(1 for _ in p.open("r", encoding="utf-8"))
            console.print(f"  e{level}.jsonl  →  {n} entries")

    # Upload the finished e1/e2/(e3) JSONLs as a versioned W&B Artifact.
    # No-op when WEAVE_DISABLED.
    art = log_hek_artifacts(
        hek_path,
        name_prefix="hek",
        metadata={
            "benchmark": cfg.benchmark.name,
            "retriever": cfg.get("retriever", {}).get("name"),
            "hf_source": cfg.hek.get("hf_source"),
            "data_splits_train": cfg.data.splits.get("train") if "splits" in cfg.data else None,
            "rollouts_per_example": cfg.hek.rollouts_per_example,
            "num_train": cfg.hek.num_train,
            "cluster_threshold": cluster_threshold,
        },
    )
    if art is not None:
        console.print(f"[green]✓[/green] uploaded HEK artifact: {art.name}")
    _finish_tracing()


if __name__ == "__main__":
    app()
