#!/usr/bin/env python3
"""
Sample 300 queries from topics-qrels/ and split into a 100-query train set
and a 200-query eval set, written to topics-qrels-100/ and topics-qrels-200/.

Same seed (42) reproduces the same split. Re-running is a no-op unless
--force is passed (avoids silently overwriting an existing split).

Each output dir mirrors topics-qrels/:
    queries.tsv            (qid <TAB> query_text)
    qrel_golds.txt         (TREC: qid Q0 docid rel)
    qrel_evidence.txt      (TREC: qid Q0 docid rel)
"""

import argparse
import csv
import random
import sys
from pathlib import Path

SEED = 42
SAMPLE_SIZE = 300
TRAIN_SIZE = 100
EVAL_SIZE = 200


def read_queries(path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            assert len(row) == 2, f"Malformed row in {path}: {row}"
            rows.append((row[0].strip(), row[1].strip()))
    return rows


def write_queries(path: Path, rows: list[tuple[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        for qid, qtext in rows:
            writer.writerow([qid, qtext])


def filter_qrel(src: Path, dst: Path, qids: set[str]) -> int:
    n = 0
    with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            qid = line.split(None, 1)[0]
            if qid in qids:
                fout.write(line)
                n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default="topics-qrels", help="source dir (default: %(default)s)")
    ap.add_argument("--train-dir", default="topics-qrels-100", help="train output dir")
    ap.add_argument("--eval-dir", default="topics-qrels-200", help="eval output dir")
    ap.add_argument("--force", action="store_true", help="overwrite existing output dirs")
    args = ap.parse_args()

    src = Path(args.src)
    train_dir = Path(args.train_dir)
    eval_dir = Path(args.eval_dir)

    queries_path = src / "queries.tsv"
    golds_path = src / "qrel_golds.txt"
    evidence_path = src / "qrel_evidence.txt"
    for p in (queries_path, golds_path, evidence_path):
        if not p.is_file():
            print(f"[ERROR] missing source file: {p}", file=sys.stderr)
            return 1

    if not args.force and (train_dir.exists() or eval_dir.exists()):
        print(f"[skip] {train_dir} or {eval_dir} already exists; pass --force to overwrite.")
        return 0

    queries = read_queries(queries_path)
    if len(queries) < SAMPLE_SIZE:
        print(
            f"[ERROR] only {len(queries)} queries available, need {SAMPLE_SIZE}",
            file=sys.stderr,
        )
        return 1

    rng = random.Random(SEED)
    sampled = rng.sample(queries, SAMPLE_SIZE)
    train = sampled[:TRAIN_SIZE]
    eval_ = sampled[TRAIN_SIZE : TRAIN_SIZE + EVAL_SIZE]

    train_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_dir, split_rows in (
        ("train", train_dir, train),
        ("eval", eval_dir, eval_),
    ):
        qids = {qid for qid, _ in split_rows}
        write_queries(split_dir / "queries.tsv", split_rows)
        n_golds = filter_qrel(golds_path, split_dir / "qrel_golds.txt", qids)
        n_ev = filter_qrel(evidence_path, split_dir / "qrel_evidence.txt", qids)
        print(
            f"[{split_name}] {len(split_rows)} queries -> {split_dir} "
            f"(golds: {n_golds} lines, evidence: {n_ev} lines)"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())