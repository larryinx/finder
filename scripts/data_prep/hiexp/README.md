# `scripts/data_prep/hiexp/` — HiExp data preparation

One-time downloaders + processors. The output drops under
``data/hiexp/{train,eval}/`` and is consumed by:

- ``scripts/hiexp_prompt/train/`` (HEK construction pipeline)
- ``scripts/hiexp_prompt/eval/run_hotpotqa.py`` (HotpotQA evaluation)

## Quick start

```bash
# Through the project's dispatcher (recommended)
bash scripts/data_prep/prepare.sh hiexp

# Or invoke each step directly
python scripts/data_prep/hiexp/build_train.py
python scripts/data_prep/hiexp/build_eval_hotpotqa.py
```

## What gets produced

```
data/hiexp/
  train/
    r1_searcher_stage2.jsonl              # 8148 rows; HiExp paper §A
  eval/
    hotpotqa_dev_sample1000_seed42.jsonl  # 1000 rows; paper §A protocol
```

Each line is a JSON object with:

```json
{
  "query_id": "r1s2:37653",
  "question": "...",
  "golden_answers": ["..."],
  "metadata": {"source_dataset": "r1_searcher_stage2", "stage_idx": 37653}
}
```

This shape matches `scripts/hiexp_prompt/_shared.py:QaExample` so the
agent code can consume it with no further conversion.

## Data sources

| Output | HuggingFace source | Paper anchor |
|---|---|---|
| `train/r1_searcher_stage2.jsonl` | `XXsongLALA/RAG-RL-Hotpotqa-with-2wiki` (split `stage_2`) | §A: "8,148 examples from HotpotQA and 2WikiMultiHopQA, which are selected through data selection in R1-Searcher" |
| `eval/hotpotqa_dev_sample1000_seed42.jsonl` | `RUC-NLPIR/FlashRAG_datasets` (config `hotpotqa`, split `dev`) | §A: "we randomly sample 1,000 examples from their respective dev sets as our final test set, with a fixed random seed 42" |

The R1-Searcher dataset also has a 350-row `stage_1` split (SFT
precursor); HiExp does not use it. Pass `--include-stage1` to
`build_train.py` if you want it materialized anyway.

## Idempotency

Both scripts no-op when the output already has the expected row count
(and matching seed for the eval sampler). Pass `--force` to rebuild.

## Wiring to the rest of the pipeline

After running data prep, `scripts/hiexp_prompt/_shared.py:load_r1_searcher_stage2()`
automatically picks up the local JSONL — no env var needed. So:

```bash
# Once the local JSONL is in place, the loader uses it directly (no
# HuggingFace fetch needed):
python scripts/hiexp_prompt/train/build_hek.py \
    --hek-path data/hek/paper_mix \
    --benchmark hotpotqa \
    -o hek.hf_source=paper_mix
```

If your stage-2 JSONL lives outside the default location, point the
loader at it via the function arg:

```python
from hiexp_prompt._shared import load_r1_searcher_stage2
examples = load_r1_searcher_stage2(local_jsonl="/some/other/path/stage2.jsonl")
```

(or symlink ``data/hiexp/train/r1_searcher_stage2.jsonl`` to wherever
your file lives — that's what the data-prep script is for.)
