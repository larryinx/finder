from __future__ import annotations

import json

from hiexp_prompt._shared import load_browsecomp_plus_jsonl


def test_loads_raw_bcp_decrypted_shape(tmp_path):
    path = tmp_path / "raw.jsonl"
    path.write_text(
        json.dumps(
            {
                "query_id": "769",
                "query": "Q?",
                "answer": "Queen Arwa University",
                "gold_docs": ["86190"],
                "evidence_docs": ["1", "2"],
                "negative_docs": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    examples = load_browsecomp_plus_jsonl(path)
    assert len(examples) == 1
    ex = examples[0]
    assert ex.query_id == "769"
    assert ex.question == "Q?"
    assert ex.golden_answers == ["Queen Arwa University"]
    assert ex.metadata["gold_docs"] == ["86190"]


def test_loads_processed_qa_example_shape(tmp_path):
    path = tmp_path / "processed.jsonl"
    path.write_text(
        json.dumps(
            {
                "query_id": "771",
                "question": "Q2?",
                "golden_answers": ["Vakkorama"],
                "metadata": {
                    "source_dataset": "browsecomp_plus",
                    "gold_docs": ["30692"],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    examples = load_browsecomp_plus_jsonl(path)
    assert len(examples) == 1
    ex = examples[0]
    assert ex.query_id == "771"
    assert ex.question == "Q2?"
    assert ex.golden_answers == ["Vakkorama"]
    assert ex.metadata["gold_docs"] == ["30692"]
    # Original ``source_dataset`` survives the round-trip.
    assert ex.metadata["source_dataset"] == "browsecomp_plus"


def test_only_query_ids_filter(tmp_path):
    path = tmp_path / "mix.jsonl"
    path.write_text(
        "\n".join(
            json.dumps({"query_id": qid, "query": "q", "answer": "a"})
            for qid in ["1", "2", "3"]
        )
        + "\n",
        encoding="utf-8",
    )
    examples = load_browsecomp_plus_jsonl(path, only_query_ids={"2"})
    assert [e.query_id for e in examples] == ["2"]
