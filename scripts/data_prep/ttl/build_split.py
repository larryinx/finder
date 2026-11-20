"""Partition all 830 BCP decrypted examples into TTL train/val/test (100/100/630).

Output (BCP-native: queries.tsv + TREC qrels), under ``data/BCP/``:

    topics-qrels-ttl-train100/
    topics-qrels-ttl-val100/
    topics-qrels-ttl-test630/

Splits are disjoint and exhaustive (100+100+630 = 830). Naming differs from
the existing topics-qrels-* dirs so this lives alongside the hiexp split.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import typer
from rich.console import Console

_PROJECT_ROOT = Path(__file__).resolve().parents[3]

app = typer.Typer(add_completion=False)
console = Console()


def _read_decrypted_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_queries_tsv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            qid = str(row["query_id"])
            qtext = str(row["query"]).replace("\t", " ").replace("\n", " ")
            f.write(f"{qid}\t{qtext}\n")


def _write_trec_qrel(path: Path, rows: list[dict], doc_field: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            qid = str(row["query_id"])
            for docid in row.get(doc_field) or []:
                f.write(f"{qid} Q0 {docid} 1\n")


def _write_split(out_dir: Path, rows: list[dict], label: str) -> None:
    rows = sorted(rows, key=lambda r: str(r["query_id"]))
    _write_queries_tsv(out_dir / "queries.tsv", rows)
    _write_trec_qrel(out_dir / "qrel_golds.txt", rows, doc_field="gold_docs")
    _write_trec_qrel(out_dir / "qrel_evidence.txt", rows, doc_field="evidence_docs")
    n_gold = sum(len(r.get("gold_docs") or []) for r in rows)
    n_evid = sum(len(r.get("evidence_docs") or []) for r in rows)
    console.print(
        f"[green]✓[/green] {label:25s} → {out_dir} "
        f"({len(rows)} queries, {n_gold} gold rels, {n_evid} evidence rels)"
    )


@app.command()
def main(
    out_root: Path = typer.Option(
        Path("data/BCP"),
        help="Where to materialize the new ttl-* splits.",
    ),
    decrypted_jsonl: Path = typer.Option(
        Path("data/BCP/data/browsecomp_plus_decrypted.jsonl"),
        help="Source decrypted jsonl (830 rows).",
    ),
    train_size: int = typer.Option(100),
    val_size: int = typer.Option(100),
    test_size: int = typer.Option(630),
    seed: int = typer.Option(42),
    train_dir: str = typer.Option("topics-qrels-ttl-train100"),
    val_dir: str = typer.Option("topics-qrels-ttl-val100"),
    test_dir: str = typer.Option("topics-qrels-ttl-test630"),
    force: bool = typer.Option(False, help="Overwrite existing files."),
) -> None:
    out_root = (out_root if out_root.is_absolute() else _PROJECT_ROOT / out_root).resolve()
    decrypted_jsonl = (
        decrypted_jsonl if decrypted_jsonl.is_absolute()
        else _PROJECT_ROOT / decrypted_jsonl
    ).resolve()

    if not decrypted_jsonl.exists():
        raise typer.BadParameter(f"missing source: {decrypted_jsonl}")

    rows = _read_decrypted_jsonl(decrypted_jsonl)
    n_total = len(rows)
    n_want = train_size + val_size + test_size
    if n_want != n_total:
        raise typer.BadParameter(
            f"split sizes sum to {n_want} but decrypted jsonl has {n_total} rows"
        )

    rows.sort(key=lambda r: str(r["query_id"]))
    random.Random(seed).shuffle(rows)

    train = rows[:train_size]
    val = rows[train_size : train_size + val_size]
    test = rows[train_size + val_size :]

    splits = [
        (out_root / train_dir, train, train_dir),
        (out_root / val_dir, val, val_dir),
        (out_root / test_dir, test, test_dir),
    ]
    if not force and all((d / "queries.tsv").exists() for d, _, _ in splits):
        for d, _, label in splits:
            n = sum(1 for _ in (d / "queries.tsv").open() if _.strip())
            console.print(
                f"[green]✓[/green] {label:25s} → {d} (already has {n} queries; --force to rebuild)"
            )
        return

    for d, rows_, label in splits:
        _write_split(d, rows_, label)

    console.rule("[bold green]ttl splits ready[/bold green]")


if __name__ == "__main__":
    app()
