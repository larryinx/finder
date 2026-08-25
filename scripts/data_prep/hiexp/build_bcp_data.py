"""Materialize ``data/BCP/`` mirroring BrowseComp-Plus's own layout.

Rationale: the user wants to drive baseline rollouts through BCP's own
``search_agent/qwen_client_async.py`` (which expects ``queries.tsv``)
and grade with BCP's ``GRADER_TEMPLATE``. So we lay everything out in
BCP-native form under one local directory:

    data/BCP/
      data/
        browsecomp_plus_decrypted.jsonl       # copied from third_party
      topics-qrels-200/                       # copied from third_party
        queries.tsv
        qrel_evidence.txt
        qrel_golds.txt
      topics-qrels-train500/                  # built here (seed=42, 500 from non-eval)
        queries.tsv
        qrel_evidence.txt
        qrel_golds.txt
      topics-qrels-train100/                  # first 100 ids of train500 (subset, no new sampling)
        queries.tsv                           # for fast iteration / smoke runs
        qrel_evidence.txt
        qrel_golds.txt
      topics-qrels-val100/                    # built here (seed=43, 100 from non-eval ∧ non-train)
        queries.tsv                           # used by ACE offline-mode validation
        qrel_evidence.txt
        qrel_golds.txt
      topics-qrels-val20/                     # first 20 ids of val100 (subset)
        queries.tsv                           # for cheap val checkpoints during fast runs
        qrel_evidence.txt
        qrel_golds.txt

The eval split (200) is the existing topics-qrels-200 from the BCP
repo, just copied for self-containment. The train split (500) is
sampled from the non-eval pool of the 830-example decrypted JSONL.

Output is BCP-native (TSV + TREC-style qrels), so any downstream BCP
script — ``qwen_client_async.py``, ``evaluate_with_openai.py`` —
points at ``data/BCP/...`` paths and Just Works.
"""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

import typer
from rich.console import Console

_PROJECT_ROOT = Path(__file__).resolve().parents[3]

app = typer.Typer(add_completion=False)
console = Console()


# ── BCP-native I/O ─────────────────────────────────────────────────────────


def _read_decrypted_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _read_query_ids(tsv_path: Path) -> set[str]:
    ids: set[str] = set()
    with tsv_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            ids.add(line.split("\t", 1)[0].strip())
    return ids


def _write_queries_tsv(path: Path, rows: list[dict]) -> None:
    """``<query_id>\\t<query>`` per line, matching BCP's own format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            qid = str(row["query_id"])
            qtext = str(row["query"]).replace("\t", " ").replace("\n", " ")
            f.write(f"{qid}\t{qtext}\n")


def _write_trec_qrel(path: Path, rows: list[dict], doc_field: str) -> None:
    """``<qid> Q0 <docid> 1`` per (query, gold doc) — BCP's TREC qrel format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            qid = str(row["query_id"])
            for docid in row.get(doc_field) or []:
                f.write(f"{qid} Q0 {docid} 1\n")


# ── Driver ─────────────────────────────────────────────────────────────────


def _copy_or_link(src: Path, dst: Path, *, force: bool) -> None:
    if dst.exists():
        if not force:
            return
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)


@app.command()
def main(
    out_root: Path = typer.Option(
        Path("data/BCP"),
        help="Where to materialize the BCP-native layout.",
    ),
    bcp_root: Path = typer.Option(
        _PROJECT_ROOT / "third_party/BrowseComp-Plus",
        help="Source BrowseComp-Plus checkout.",
    ),
    train_split_name: str = typer.Option(
        "topics-qrels-train500",
        help="Folder name for the new train split (mirrors topics-qrels-200 style).",
    ),
    train_size: int = typer.Option(500, help="Number of train queries to sample."),
    val_split_name: str = typer.Option(
        "topics-qrels-val100",
        help="Folder name for the validation split (used by ACE offline mode).",
    ),
    val_size: int = typer.Option(
        100,
        help=(
            "Number of val queries. Sampled from the non-eval, non-train pool "
            "with a different seed than train so the two splits are disjoint."
        ),
    ),
    eval_split_dir_name: str = typer.Option(
        "topics-qrels-200",
        help="The existing BCP eval split to mirror into data/BCP/ (excluded from train + val pool).",
    ),
    seed: int = typer.Option(42),
    val_seed: int = typer.Option(
        43, help="Distinct seed for val sampling so val ∩ train = ∅ by design."
    ),
    small_train_split_name: str = typer.Option(
        "topics-qrels-train100",
        help=(
            "Folder name for a smaller train *subset* derived from the parent "
            "train split. Useful for fast offline-ACE iteration."
        ),
    ),
    small_train_size: int = typer.Option(
        100,
        help="Number of queries in the small train subset (prefix of train sorted by qid).",
    ),
    small_val_split_name: str = typer.Option(
        "topics-qrels-val20",
        help="Folder name for a smaller val *subset* derived from the parent val split.",
    ),
    small_val_size: int = typer.Option(
        20,
        help="Number of queries in the small val subset (prefix of val sorted by qid).",
    ),
    force: bool = typer.Option(False, help="Overwrite existing files."),
) -> None:
    out_root = out_root.resolve()
    bcp_root = bcp_root.resolve()

    src_decrypted = bcp_root / "data" / "browsecomp_plus_decrypted.jsonl"
    src_eval_dir = bcp_root / eval_split_dir_name

    for required in (src_decrypted, src_eval_dir / "queries.tsv"):
        if not required.exists():
            raise typer.BadParameter(f"missing source file: {required}")

    # 1) Mirror the decrypted JSONL (gold answers + gold/evidence docids)
    dst_decrypted = out_root / "data" / src_decrypted.name
    if force or not dst_decrypted.exists():
        _copy_or_link(src_decrypted, dst_decrypted, force=force)
    console.print(f"[green]✓[/green] decrypted JSONL → {dst_decrypted}")

    # 2) Mirror the eval split as-is.
    dst_eval = out_root / eval_split_dir_name
    if force or not dst_eval.exists():
        _copy_or_link(src_eval_dir, dst_eval, force=force)
    console.print(f"[green]✓[/green] eval split    → {dst_eval}")

    # 3) Read the source pools (decrypted minus eval ids).
    eval_ids = _read_query_ids(src_eval_dir / "queries.tsv")
    rows = _read_decrypted_jsonl(src_decrypted)
    non_eval_pool = [r for r in rows if str(r.get("query_id")) not in eval_ids]

    # 3a) Train split: sample train_size from non-eval pool.
    dst_train = out_root / train_split_name
    train_queries_tsv = dst_train / "queries.tsv"
    train_sampled: list[dict]
    if not force and train_queries_tsv.exists():
        # Already-built; load its ids so we can exclude them from val.
        train_ids = _read_query_ids(train_queries_tsv)
        train_sampled = [r for r in rows if str(r["query_id"]) in train_ids]
        n = sum(1 for _ in train_queries_tsv.open("r", encoding="utf-8") if _.strip())
        console.print(
            f"[green]✓[/green] train split   → {dst_train} (already has {n} queries; --force to rebuild)"
        )
    else:
        if train_size > len(non_eval_pool):
            raise typer.BadParameter(
                f"asked for {train_size} train queries but only {len(non_eval_pool)} non-eval available"
            )
        rng = random.Random(seed)
        train_sampled = rng.sample(non_eval_pool, train_size)
        train_sampled.sort(key=lambda r: str(r["query_id"]))
        _write_queries_tsv(dst_train / "queries.tsv", train_sampled)
        _write_trec_qrel(dst_train / "qrel_golds.txt", train_sampled, doc_field="gold_docs")
        _write_trec_qrel(dst_train / "qrel_evidence.txt", train_sampled, doc_field="evidence_docs")
        console.print(
            f"[green]✓[/green] train split   → {dst_train} "
            f"(seed={seed}, {len(train_sampled)} queries, "
            f"{sum(len(r.get('gold_docs') or []) for r in train_sampled)} gold rels, "
            f"{sum(len(r.get('evidence_docs') or []) for r in train_sampled)} evidence rels)"
        )

    # 3b) Val split: sample val_size from non-eval ∧ non-train pool. Used by
    #     ACE offline mode to pick the best playbook across epochs.
    dst_val = out_root / val_split_name
    val_queries_tsv = dst_val / "queries.tsv"
    if not force and val_queries_tsv.exists():
        n = sum(1 for _ in val_queries_tsv.open("r", encoding="utf-8") if _.strip())
        console.print(
            f"[green]✓[/green] val split     → {dst_val} (already has {n} queries; --force to rebuild)"
        )
    else:
        train_ids = {str(r["query_id"]) for r in train_sampled}
        val_pool = [r for r in non_eval_pool if str(r["query_id"]) not in train_ids]
        if val_size > len(val_pool):
            raise typer.BadParameter(
                f"asked for {val_size} val queries but only {len(val_pool)} pool available "
                f"(non-eval={len(non_eval_pool)}, train_used={len(train_sampled)})"
            )
        rng_val = random.Random(val_seed)
        val_sampled = rng_val.sample(val_pool, val_size)
        val_sampled.sort(key=lambda r: str(r["query_id"]))
        _write_queries_tsv(dst_val / "queries.tsv", val_sampled)
        _write_trec_qrel(dst_val / "qrel_golds.txt", val_sampled, doc_field="gold_docs")
        _write_trec_qrel(dst_val / "qrel_evidence.txt", val_sampled, doc_field="evidence_docs")
        console.print(
            f"[green]✓[/green] val split     → {dst_val} "
            f"(seed={val_seed}, {len(val_sampled)} queries, "
            f"{sum(len(r.get('gold_docs') or []) for r in val_sampled)} gold rels, "
            f"{sum(len(r.get('evidence_docs') or []) for r in val_sampled)} evidence rels)"
        )

    # 4) Smaller subsets (deterministic prefixes of the parent splits).
    #    These are *strict subsets* — train100 ⊂ train500, val20 ⊂ val100 —
    #    so any insight learned on the small split also lives in the big one.
    #    No new randomness involved; we slice the already-sorted-by-qid lists.
    def _write_subset(
        *,
        parent_rows: list[dict],
        parent_dir: Path,
        out_dir: Path,
        size: int,
        label: str,
    ) -> None:
        if size > len(parent_rows):
            raise typer.BadParameter(
                f"asked for {size} {label} queries but parent only has {len(parent_rows)}"
            )
        subset_tsv = out_dir / "queries.tsv"
        if not force and subset_tsv.exists():
            n = sum(1 for _ in subset_tsv.open("r", encoding="utf-8") if _.strip())
            console.print(
                f"[green]✓[/green] {label:13s} → {out_dir} (already has {n} queries; --force to rebuild)"
            )
            return
        subset = parent_rows[:size]
        _write_queries_tsv(subset_tsv, subset)
        _write_trec_qrel(out_dir / "qrel_golds.txt", subset, doc_field="gold_docs")
        _write_trec_qrel(out_dir / "qrel_evidence.txt", subset, doc_field="evidence_docs")
        console.print(
            f"[green]✓[/green] {label:13s} → {out_dir} "
            f"(prefix of {parent_dir.name}, {len(subset)} queries)"
        )

    # train100 = first N of (already-sorted, seed=42 sampled) train_sampled
    # We need ``train_sampled`` populated in both branches (force / not force).
    # The "not force" branch loads it from disk; ensure it's sorted by qid for
    # subset stability.
    train_sampled.sort(key=lambda r: str(r["query_id"]))
    _write_subset(
        parent_rows=train_sampled,
        parent_dir=dst_train,
        out_dir=out_root / small_train_split_name,
        size=small_train_size,
        label=small_train_split_name,
    )

    # val20 = first M of val_sampled. Loading val from disk if it already
    # existed is a bit awkward — re-read it from the just-written tsv to
    # keep the code simple and the subset stable.
    val_ids = _read_query_ids(dst_val / "queries.tsv")
    val_rows = [r for r in rows if str(r["query_id"]) in val_ids]
    val_rows.sort(key=lambda r: str(r["query_id"]))
    _write_subset(
        parent_rows=val_rows,
        parent_dir=dst_val,
        out_dir=out_root / small_val_split_name,
        size=small_val_size,
        label=small_val_split_name,
    )

    console.rule("[bold green]data/BCP ready[/bold green]")
    console.print(
        f"Next: cd {bcp_root} && conda activate bcp &&\n"
        f"      python search_agent/qwen_client_async.py \\\n"
        f"          --query {out_root}/{train_split_name}/queries.tsv \\\n"
        f"          --mcp-url http://127.0.0.1:8080/mcp \\\n"
        f"          --output-dir {out_root}/runs/baseline_train500/<exp>"
    )


if __name__ == "__main__":
    app()
