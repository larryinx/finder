"""Materialize the HotpotQA evaluation split into ``data/hiexp/eval/``.

Source: ``RUC-NLPIR/FlashRAG_datasets`` (subset ``hotpotqa``) on
HuggingFace.

Reproduces the protocol from HiExp paper §A:

    "we randomly sample 1,000 examples from their respective dev sets
    as our final test set, with a fixed random seed 42"

Output schema (one JSON object per line) matches the HEK training
output so the eval runner can use the same loader:

    {
      "query_id":     "<flashrag id>",
      "question":     "...",
      "golden_answers": ["..."],
      "metadata":     {"source_dataset": "hotpotqa", "source_split": "dev",
                       "sampled_from_full_dev": <N_full_dev>,
                       "sampling_seed": 42}
    }

Idempotent: if the output JSONL already has the expected row count and
the same sampling parameters in its first record's ``metadata``, the
script returns immediately unless ``--force`` is passed.

Usage:
    python scripts/data_prep/hiexp/build_eval_hotpotqa.py
    python scripts/data_prep/hiexp/build_eval_hotpotqa.py --num-samples 200 --seed 7
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer(add_completion=False)
console = Console()

DEFAULT_HF_DATASET = "RUC-NLPIR/FlashRAG_datasets"
DEFAULT_CONFIG = "hotpotqa"


def _load_full(dataset_id: str, config: str, split: str) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset(dataset_id, config, split=split)
    rows: list[dict] = []
    for row in ds:
        rows.append(
            {
                "query_id": str(row.get("id", row.get("_id", ""))),
                "question": str(row["question"]),
                "golden_answers": list(row.get("golden_answers") or []),
                "metadata": dict(row.get("metadata") or {}),
            }
        )
    return rows


def _existing_signature(path: Path) -> tuple[int, int, int] | None:
    """Return ``(line_count, sampling_seed, sampled_from_full_dev)`` if known."""
    if not path.exists():
        return None
    n = 0
    seed = -1
    full = -1
    first = True
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            n += 1
            if first:
                try:
                    rec = json.loads(line)
                    meta = rec.get("metadata") or {}
                    seed = int(meta.get("sampling_seed", -1))
                    full = int(meta.get("sampled_from_full_dev", -1))
                except Exception:
                    return None
                first = False
    return n, seed, full


@app.command()
def main(
    out_dir: Path = typer.Option(
        Path("data/hiexp/eval"),
        help="Where to write the sampled JSONL.",
    ),
    out_name: str | None = typer.Option(
        None,
        help="Output filename (defaults to hotpotqa_dev_sample<N>_seed<S>.jsonl).",
    ),
    num_samples: int = typer.Option(1000, help="Paper §A protocol: 1000."),
    seed: int = typer.Option(42, help="Paper §A protocol: seed 42."),
    split: str = typer.Option("dev"),
    dataset_id: str = typer.Option(DEFAULT_HF_DATASET),
    config: str = typer.Option(DEFAULT_CONFIG),
    force: bool = typer.Option(False),
) -> None:
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (out_name or f"{config}_{split}_sample{num_samples}_seed{seed}.jsonl")

    sig = _existing_signature(out_path)
    if not force and sig is not None and sig[0] == num_samples and sig[1] == seed:
        console.print(
            f"[green]✓[/green] {out_path} already has {num_samples} rows "
            f"(seed={seed}); skipping. Use --force to rebuild."
        )
        return

    console.print(
        f"[bold]Loading[/bold] {dataset_id}::{config} split={split} from HuggingFace ..."
    )
    rows = _load_full(dataset_id, config, split)
    n_full = len(rows)
    console.print(f"  full split has {n_full} rows")

    if num_samples >= n_full:
        sampled = rows
    else:
        rng = random.Random(seed)
        sampled = rng.sample(rows, num_samples)
    # Deterministic ordering inside the JSONL: sort by query_id so two
    # builds with the same (seed, num_samples) produce byte-identical files.
    sampled.sort(key=lambda r: r["query_id"])

    with out_path.open("w", encoding="utf-8") as f:
        for rec in sampled:
            meta = dict(rec["metadata"])
            meta.update(
                source_dataset=config,
                source_split=split,
                sampled_from_full_dev=n_full,
                sampling_seed=seed,
            )
            f.write(
                json.dumps(
                    {
                        "query_id": rec["query_id"],
                        "question": rec["question"],
                        "golden_answers": rec["golden_answers"],
                        "metadata": meta,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    console.print(f"[green]✓[/green] wrote {len(sampled)} rows → {out_path}")
    console.rule("[bold green]HotpotQA eval split ready[/bold green]")
    console.print(
        f"Next: python scripts/hiexp_prompt/eval/run_hotpotqa.py --num-samples {num_samples} --seed {seed} ..."
    )


if __name__ == "__main__":
    app()
