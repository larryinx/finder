"""Shared helpers for the HotpotQA prompt-based reproduction scripts.

Used by both ``train/`` and ``eval/`` so dataset loading, JSONL I/O, and
trajectory rendering stay consistent.
"""

from __future__ import annotations

import json
import random
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

# Allow running ``python scripts/hiexp_prompt/eval/run_hotpotqa.py`` from
# anywhere by inserting the project ``src/`` on the path.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from hiexp.agent import HiExpTrajectory, TrajectoryItem  # noqa: E402

__all__ = [
    "load_hotpotqa_subset",
    "load_flashrag_split",
    "load_paper_hek_training_mix",
    "load_r1_searcher_stage2",
    "load_browsecomp_plus_jsonl",
    "load_browsecomp_plus_split",
    "QaExample",
    "read_jsonl",
    "write_jsonl",
    "append_jsonl",
    "trajectory_to_dict",
    "render_trajectory_for_prompt",
    "JsonEncoder",
]


# ── Dataset ────────────────────────────────────────────────────────────────


class QaExample:
    """Tiny QA record with the fields we use everywhere in this folder."""

    __slots__ = ("query_id", "question", "golden_answers", "metadata")

    def __init__(
        self,
        query_id: str,
        question: str,
        golden_answers: list[str],
        metadata: dict | None = None,
    ) -> None:
        self.query_id = query_id
        self.question = question
        self.golden_answers = golden_answers
        self.metadata = metadata or {}

    def to_dict(self) -> dict:
        return {
            "query_id": self.query_id,
            "question": self.question,
            "golden_answers": self.golden_answers,
            "metadata": self.metadata,
        }


def load_flashrag_split(
    config_name: str,
    split: str,
    num_samples: int | None = None,
    seed: int = 42,
    dataset_id: str = "RUC-NLPIR/FlashRAG_datasets",
    tag_source: bool = True,
) -> list[QaExample]:
    """Generic FlashRAG_datasets loader.

    Args:
        config_name: ``hotpotqa`` / ``2wikimultihopqa`` / ``musique`` / etc.
        split: ``train`` / ``dev`` / ``test``.
        num_samples: cap with random sampling.
        seed: sampling seed.
        tag_source: when True, prefix ``query_id`` with ``"<config>:"`` so
            entries from different sources don't collide on shared ids.
    """
    from datasets import load_dataset

    ds = load_dataset(dataset_id, config_name, split=split)
    examples: list[QaExample] = []
    for row in ds:
        raw_id = str(row.get("id", row.get("_id", "")))
        qid = f"{config_name}:{raw_id}" if tag_source else raw_id
        meta = dict(row.get("metadata") or {})
        meta["source_dataset"] = config_name
        meta["source_split"] = split
        examples.append(
            QaExample(
                query_id=qid,
                question=row["question"],
                golden_answers=list(row.get("golden_answers") or []),
                metadata=meta,
            )
        )
    if num_samples is not None and num_samples < len(examples):
        rng = random.Random(seed)
        examples = rng.sample(examples, num_samples)
    return examples


def load_hotpotqa_subset(
    split: str = "dev",
    num_samples: int | None = 1000,
    seed: int = 42,
    dataset_id: str = "RUC-NLPIR/FlashRAG_datasets",
    config_name: str = "hotpotqa",
) -> list[QaExample]:
    """Load and randomly sample HotpotQA from FlashRAG_datasets.

    Paper §A: "we randomly sample 1,000 examples from their respective
    dev sets as our final test set, with a fixed random seed 42".
    """
    return load_flashrag_split(
        config_name=config_name,
        split=split,
        num_samples=num_samples,
        seed=seed,
        dataset_id=dataset_id,
        tag_source=False,  # backward-compat: keep raw HotpotQA ids for eval runs
    )


def load_r1_searcher_stage2(
    seed: int = 42,
    num_samples: int | None = None,
    dataset_id: str | None = None,
    local_jsonl: str | Path | None = None,
) -> list[QaExample]:
    """Load R1-Searcher stage-2 training data (paper §A).

    Resolution order:
      1. Explicit ``local_jsonl`` argument (if provided).
      2. ``data/hiexp/train/r1_searcher_stage2.jsonl`` under the project
         root, if it exists (the default output of
         ``scripts/data_prep/hiexp/build_train.py``).
      3. Fall through to HuggingFace via the ``dataset_id`` argument
         (defaults to ``XXsongLALA/RAG-RL-Hotpotqa-with-2wiki``, the id
         the HiExp paper §A points to). The data-prep script always
         uses this id, so option 3 only fires when the user skipped
         data-prep entirely.
    """
    # 1: explicit local JSONL.
    candidate_local: Path | None = Path(local_jsonl) if local_jsonl else None

    # 2: project default location.
    if candidate_local is None:
        project_default = _PROJECT_ROOT / "data" / "hiexp" / "train" / "r1_searcher_stage2.jsonl"
        if project_default.exists():
            candidate_local = project_default

    if candidate_local is not None and candidate_local.exists():
        return _load_from_local_jsonl(
            candidate_local,
            tag="r1s2",
            source_dataset="r1_searcher_stage2",
            seed=seed,
            num_samples=num_samples,
        )

    # 3: fall through to HuggingFace.
    dataset_id = dataset_id or "XXsongLALA/RAG-RL-Hotpotqa-with-2wiki"

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "R1-Searcher data not available locally and `datasets` is not "
            "installed. Run `uv add datasets` or "
            "`bash scripts/data_prep/prepare.sh hiexp`."
        ) from exc

    ds = load_dataset(dataset_id, split="stage_2")
    examples: list[QaExample] = []
    for row in ds:
        question = row.get("question") or row.get("query") or row.get("input")
        if not question:
            continue
        ans = row.get("answer")
        golden = (
            row.get("golden_answers")
            or row.get("answers")
            or ([ans] if ans else [])
        )
        raw_id = str(
            row.get("idx")
            if row.get("idx") is not None
            else (row.get("id") or row.get("_id") or row.get("query_id") or "")
        )
        examples.append(
            QaExample(
                query_id=f"r1s2:{raw_id}",
                question=str(question),
                golden_answers=[str(g) for g in golden],
                metadata={"source_dataset": "r1_searcher_stage2"},
            )
        )
    if num_samples is not None and num_samples < len(examples):
        rng = random.Random(seed)
        examples = rng.sample(examples, num_samples)
    return examples


# ── BrowseComp-Plus ────────────────────────────────────────────────────────


def load_browsecomp_plus_jsonl(
    jsonl_path: str | Path,
    only_query_ids: set[str] | None = None,
) -> list[QaExample]:
    """Load BrowseComp-Plus examples from JSONL.

    Accepts two equivalent shapes:

    * Raw BCP ``data/browsecomp_plus_decrypted.jsonl``:
      ``{query_id, query, answer, gold_docs, evidence_docs, negative_docs}``
    * Our processed JSONL produced by ``QaExample.to_dict()``:
      ``{query_id, question, golden_answers, metadata}``

    ``only_query_ids`` (when given) filters to a specific subset, e.g. the
    200 query ids listed in ``topics-qrels-200/queries.tsv``.
    """
    path = Path(jsonl_path)
    out: list[QaExample] = []
    for obj in read_jsonl(path):
        qid = str(obj.get("query_id", ""))
        if only_query_ids is not None and qid not in only_query_ids:
            continue

        question = str(obj.get("query") or obj.get("question") or "")

        # Gold answers: prefer pre-listified ``golden_answers`` (our
        # processed shape) over the raw scalar ``answer`` (BCP's shape).
        if obj.get("golden_answers"):
            golden = [str(g) for g in obj["golden_answers"] if g]
        elif obj.get("answer"):
            golden = [str(obj["answer"])]
        else:
            golden = []

        # Metadata: prefer the nested ``metadata`` block when present
        # (round-trip), otherwise build it from the raw BCP fields.
        meta = obj.get("metadata")
        if isinstance(meta, dict) and meta:
            metadata = dict(meta)
            metadata.setdefault("source_dataset", "browsecomp_plus")
        else:
            metadata = {
                "source_dataset": "browsecomp_plus",
                "gold_docs": list(obj.get("gold_docs") or []),
                "evidence_docs": list(obj.get("evidence_docs") or []),
                "negative_docs": list(obj.get("negative_docs") or []),
            }

        out.append(
            QaExample(
                query_id=qid,
                question=question,
                golden_answers=golden,
                metadata=metadata,
            )
        )
    return out


def _read_query_ids_from_tsv(path: Path) -> list[str]:
    """Read the first column of a ``queries.tsv`` file as a list of ids.

    BCP's ``topics-qrels*/queries.tsv`` is ``<id>\\t<question>`` per line.
    """
    ids: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            ids.append(line.split("\t", 1)[0].strip())
    return ids


def load_browsecomp_plus_split(
    *,
    queries_tsv: str | Path,
    decrypted_jsonl: str | Path,
    num_samples: int | None = None,
    seed: int = 42,
) -> list[QaExample]:
    """Load the BCP examples whose ids appear in ``queries_tsv``.

    Joins ``queries_tsv`` (id list) against
    ``browsecomp_plus_decrypted.jsonl`` (full records with gold answers).
    Optionally caps to ``num_samples`` via deterministic sampling.
    """
    queries_tsv = Path(queries_tsv)
    keep_ids = set(_read_query_ids_from_tsv(queries_tsv))
    examples = load_browsecomp_plus_jsonl(decrypted_jsonl, only_query_ids=keep_ids)
    missing = keep_ids - {e.query_id for e in examples}
    if missing:
        # Hard fail: the gold-answer JSONL is supposed to be a superset.
        raise RuntimeError(
            f"{len(missing)} query ids in {queries_tsv} not found in "
            f"{decrypted_jsonl}; sample missing: {sorted(missing)[:5]}"
        )
    if num_samples is not None and num_samples < len(examples):
        rng = random.Random(seed)
        examples = rng.sample(examples, num_samples)
    return examples


# ── Local JSONL helpers ────────────────────────────────────────────────────


def _load_from_local_jsonl(
    path: Path,
    *,
    tag: str,
    source_dataset: str,
    seed: int,
    num_samples: int | None,
) -> list[QaExample]:
    """Load ``QaExample`` records from a local JSONL produced by
    ``scripts/data_prep/hiexp/build_train.py``.
    """
    examples: list[QaExample] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            examples.append(
                QaExample(
                    query_id=str(obj.get("query_id") or f"{tag}:{len(examples)}"),
                    question=str(obj["question"]),
                    golden_answers=list(obj.get("golden_answers") or []),
                    metadata=dict(obj.get("metadata") or {"source_dataset": source_dataset}),
                )
            )
    if num_samples is not None and num_samples < len(examples):
        rng = random.Random(seed)
        examples = rng.sample(examples, num_samples)
    return examples


def load_paper_hek_training_mix(
    *,
    n_hotpot_2wiki: int = 8148,
    n_musique: int = 8000,
    seed: int = 42,
    use_r1_subset: bool = True,
) -> list[QaExample]:
    """Build the HEK training mix described in paper §A.

    Paper §A: the search-agent training data consists of
    "8,148 examples from HotpotQA and 2WikiMultiHopQA, which are
    selected through data selection in R1-Searcher" plus
    "8,000 examples from the training set of MuSiQue".

    Two paths:
      * ``use_r1_subset=True`` (paper-faithful) — calls
        :func:`load_r1_searcher_stage2` to obtain the 8,148 selected
        examples. Reads from a local JSONL when one is on disk
        (default location ``data/hiexp/train/r1_searcher_stage2.jsonl``,
        materialized by ``scripts/data_prep/hiexp/build_train.py``);
        otherwise falls through to HuggingFace.
      * ``use_r1_subset=False`` (approximation) — samples
        ``n_hotpot_2wiki`` total from FlashRAG HotpotQA+2Wiki train
        splits in proportion to their sizes. Use this when reproducing
        without R1-Searcher's exact subset.

    The MuSiQue side is always 8,000 random from FlashRAG MuSiQue train.

    Returns a single concatenated list with provenance recorded in each
    example's ``metadata.source_dataset``.
    """
    rng = random.Random(seed)

    if use_r1_subset:
        hotpot_2wiki = load_r1_searcher_stage2(seed=seed, num_samples=n_hotpot_2wiki)
    else:
        # Approximation: pull from FlashRAG HotpotQA + 2Wiki train,
        # split half-half (or proportional once we know the sizes).
        half = n_hotpot_2wiki // 2
        rest = n_hotpot_2wiki - half
        hotpot = load_flashrag_split(
            "hotpotqa", "train", num_samples=half, seed=seed
        )
        twiki = load_flashrag_split(
            "2wikimultihopqa", "train", num_samples=rest, seed=seed + 1
        )
        hotpot_2wiki = hotpot + twiki
        rng.shuffle(hotpot_2wiki)

    musique = load_flashrag_split(
        "musique", "train", num_samples=n_musique, seed=seed + 2
    )

    mix = list(hotpot_2wiki) + list(musique)
    rng.shuffle(mix)
    return mix


# ── JSONL ──────────────────────────────────────────────────────────────────


class JsonEncoder(json.JSONEncoder):
    """JSON encoder that handles dataclasses and pydantic models."""

    def default(self, o: Any) -> Any:
        if hasattr(o, "model_dump"):
            return o.model_dump()
        if is_dataclass(o):
            return asdict(o)
        if isinstance(o, set):
            return sorted(o)
        return super().default(o)


def read_jsonl(path: Path) -> Iterator[dict]:
    if not path.exists():
        return iter(())
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, records: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False, cls=JsonEncoder) + "\n")
            n += 1
    return n


def append_jsonl(path: Path, records: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False, cls=JsonEncoder) + "\n")
            n += 1
    return n


# ── Trajectory rendering ───────────────────────────────────────────────────


def trajectory_to_dict(traj: HiExpTrajectory) -> dict:
    """Serialize ``HiExpTrajectory`` to JSON-compatible dict.

    Layout matches BrowseComp-Plus's ``run_*.json`` for the items list
    (``qwen_client.py:_persist_response``); HiExp-only ``e1_inject`` /
    ``e2_inject`` entries land in ``metadata.hiexp`` so existing
    BrowseComp-Plus eval can ignore them safely.
    """
    bcp_items = []
    hiexp_items = []
    for it in traj.items:
        d = {
            "type": it.type,
            "tool_name": it.tool_name,
            "arguments": it.arguments,
            "output": it.output,
        }
        if it.type in {"e1_inject", "e2_inject"}:
            hiexp_items.append(d)
        else:
            bcp_items.append(d)

    return {
        "metadata": {
            "hiexp": {"experience_injections": hiexp_items},
        },
        "query_id": None,  # filled by callers
        "tool_call_counts": traj.tool_call_counts,
        "status": traj.status,
        "retrieved_docids": traj.retrieved_docids,
        "result": bcp_items,
    }


def _shorten(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def render_trajectory_for_prompt(
    traj_dict: dict,
    max_chars_per_item: int = 800,
) -> str:
    """Compact textual rendering of a trajectory for use inside an LLM prompt.

    Used by ``train/distill_e1.py`` to feed successful and failed
    trajectories into the Table 11 contrastive distillation prompt
    without blowing past the model's context window.
    """
    lines: list[str] = []
    items = traj_dict.get("result", [])
    for i, it in enumerate(items):
        kind = it.get("type")
        if kind == "reasoning":
            output = it.get("output") or []
            joined = "\n".join(output) if isinstance(output, list) else str(output)
            lines.append(f"[think #{i}] {_shorten(joined, max_chars_per_item)}")
        elif kind == "tool_call":
            tn = it.get("tool_name")
            args = it.get("arguments")
            out = it.get("output")
            lines.append(
                f"[tool_call #{i}] {tn}({json.dumps(args, ensure_ascii=False)}) "
                f"-> {_shorten(str(out), max_chars_per_item)}"
            )
        elif kind == "output_text":
            lines.append(f"[final #{i}] {_shorten(str(it.get('output')), max_chars_per_item)}")
    return "\n".join(lines)
