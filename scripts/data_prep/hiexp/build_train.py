"""Materialize the HEK training set into ``data/hiexp/train/``.

Source: ``XXsongLALA/RAG-RL-Hotpotqa-with-2wiki`` on HuggingFace.

The HiExp paper §A says the search-agent training data is the
"stage-2 data from Song et al. (2025)" — which is the 8,148-row split
in this dataset (HotpotQA + 2WikiMultiHopQA examples that R1-Searcher
filtered to "solvable through search"). Their stage_1 is the smaller
350-row SFT precursor; HiExp does not use it, but we keep it as an
optional target in case downstream work wants the SFT seed pool.

Output schema (one JSON object per line) matches the rest of the HiExp
training pipeline (``scripts/hiexp_prompt/_shared.py:QaExample``):

    {
      "query_id":     "r1s2:<idx>",
      "question":     "...",
      "golden_answers": ["..."],
      "metadata":     {"source_dataset": "r1_searcher_stage2", "stage_idx": <int>}
    }

Idempotent: if the output JSONL already exists with the expected row
count, the script returns immediately unless ``--force`` is passed.

Usage:
    python scripts/data_prep/hiexp/build_train.py
    python scripts/data_prep/hiexp/build_train.py --force
    python scripts/data_prep/hiexp/build_train.py --include-stage1
    python scripts/data_prep/hiexp/build_train.py \\
        --out-dir /alt/path/data/hiexp/train
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer(add_completion=False)
console = Console()

DEFAULT_HF_DATASET = "XXsongLALA/RAG-RL-Hotpotqa-with-2wiki"
EXPECTED_STAGE2_ROWS = 8148  # paper §A
EXPECTED_STAGE1_ROWS = 350


@dataclass
class _Rec:
    query_id: str
    question: str
    golden_answers: list[str]
    metadata: dict


def _row_to_rec(row: dict, stage: int) -> _Rec | None:
    ans = row.get("answer")
    if ans is None or ans == "":
        return None
    idx = row.get("idx")
    return _Rec(
        query_id=f"r1s{stage}:{idx}",
        question=str(row["question"]),
        golden_answers=[str(ans)],
        metadata={
            "source_dataset": f"r1_searcher_stage{stage}",
            "stage_idx": int(idx) if idx is not None else None,
        },
    )


def _download_jsonl(dataset_id: str, filename: str) -> Path:
    """Fetch a raw JSONL from the HF dataset repo.

    The HF parquet auto-conversion can't reconcile the differing
    ``pred_anses`` column types across stage_1 and stage_2, so the
    standard ``datasets.load_dataset`` path fails. The repo only
    contains two raw JSONL files anyway, so we pull them directly.
    """
    from huggingface_hub import hf_hub_download

    local = hf_hub_download(
        repo_id=dataset_id,
        filename=filename,
        repo_type="dataset",
    )
    return Path(local)


def _read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _load_stage(dataset_id: str, stage: int) -> list[_Rec]:
    filename = f"stage_{stage}.jsonl"
    local_path = _download_jsonl(dataset_id, filename)
    out: list[_Rec] = []
    for row in _read_jsonl(local_path):
        rec = _row_to_rec(row, stage=stage)
        if rec is not None:
            out.append(rec)
    return out


def _write_jsonl(path: Path, records: list[_Rec]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(
                json.dumps(
                    {
                        "query_id": r.query_id,
                        "question": r.question,
                        "golden_answers": r.golden_answers,
                        "metadata": r.metadata,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def _line_count(path: Path) -> int:
    if not path.exists():
        return 0
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for _ in f:
            n += 1
    return n


@app.command()
def main(
    out_dir: Path = typer.Option(
        Path("data/hiexp/train"),
        help="Where to write the processed JSONL files.",
    ),
    dataset_id: str = typer.Option(
        DEFAULT_HF_DATASET,
        help="HuggingFace dataset id for R1-Searcher stage data.",
    ),
    include_stage1: bool = typer.Option(
        False,
        help="Also materialize the 350-row stage_1 SFT split (HiExp does not use it).",
    ),
    force: bool = typer.Option(
        False, help="Re-materialize even if the output already has the expected row count."
    ),
) -> None:
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    stage2_path = out_dir / "r1_searcher_stage2.jsonl"
    stage1_path = out_dir / "r1_searcher_stage1.jsonl"

    need_stage2 = force or _line_count(stage2_path) != EXPECTED_STAGE2_ROWS
    need_stage1 = include_stage1 and (force or _line_count(stage1_path) != EXPECTED_STAGE1_ROWS)

    if not need_stage2 and not need_stage1:
        console.print(
            "[green]✓[/green] all targets already up to date; use --force to rebuild."
        )
    else:
        if need_stage2:
            console.print(
                f"[bold]Fetching[/bold] {dataset_id} :: stage_2.jsonl from HuggingFace ..."
            )
            stage2_recs = _load_stage(dataset_id, stage=2)
            if len(stage2_recs) != EXPECTED_STAGE2_ROWS:
                console.print(
                    f"[yellow]warn[/yellow] expected {EXPECTED_STAGE2_ROWS} stage_2 rows "
                    f"(paper §A); got {len(stage2_recs)}. Continuing."
                )
            _write_jsonl(stage2_path, stage2_recs)
            console.print(
                f"[green]✓[/green] wrote {len(stage2_recs)} rows → {stage2_path}"
            )
        elif _line_count(stage2_path) == EXPECTED_STAGE2_ROWS:
            console.print(
                f"[green]✓[/green] {stage2_path} already has {EXPECTED_STAGE2_ROWS} rows; skipped."
            )

        if need_stage1:
            console.print(
                f"[bold]Fetching[/bold] {dataset_id} :: stage_1.jsonl from HuggingFace ..."
            )
            stage1_recs = _load_stage(dataset_id, stage=1)
            if len(stage1_recs) != EXPECTED_STAGE1_ROWS:
                console.print(
                    f"[yellow]warn[/yellow] expected {EXPECTED_STAGE1_ROWS} stage_1 rows; "
                    f"got {len(stage1_recs)}. Continuing."
                )
            _write_jsonl(stage1_path, stage1_recs)
            console.print(
                f"[green]✓[/green] wrote {len(stage1_recs)} rows → {stage1_path}"
            )

    console.rule("[bold green]HEK training data ready[/bold green]")
    console.print(
        "Next: scripts/hiexp_prompt/train/build_hek.py --benchmark hotpotqa "
        "(reads the local JSONL automatically; benchmark/hotpotqa.yaml has "
        "hek.hf_source=paper_mix)."
    )


if __name__ == "__main__":
    app()
