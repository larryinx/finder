# `scripts/hiexp_prompt/` — HiExp End-to-End Reproduction Scripts

Two independently runnable tracks share the same `HiExpAgent`; switch between them using environment variables.

| Goal | Dataset | Retrieval backend | Evaluation | Objective |
|---|---|---|---|---|
| **A** | HotpotQA dev (1000, seed=42) | Our multilingual-e5-base + 2018 Wikipedia index | F1 / CEM / EM | Reproduce the "Prompt Based / Qwen2.5-7B / Search-o1\* + HiExp" row in Table 1 (paper §A protocol) |
| **B** | BrowseComp-Plus 200 (`topics-qrels-200/queries.tsv`) | BCP's built-in BM25 MCP | LLM-as-judge (BCP `GRADER_TEMPLATE`) | Transfer HiExp's prompt-based gains to BCP with a fair reproduction |

Milestones: `docs/stage0/hiexp/06_hotpotqa_reproduction_plan.md` and `07_m7_smoke_test.md`.

---

## Directory Structure

```
src/hiexp_prompt/            # Importable library (registered in pyproject; editable install)
  _shared.py                 # Dataset loading (HotpotQA/2Wiki/Musique/R1-Searcher/BCP) + JSONL/trajectories
  _prompts.py                # Paper Tables 11 and 12, verbatim
  _llm_utils.py              # ChatOpenAI wrapper + fault-tolerant JSON parsing
  _bootstrap.py              # .env loading and cursor restore handler
  _term.py                   # Rich terminal utilities
scripts/hiexp_prompt/        # CLI entry points (not a package; no __init__.py)
  README.md                  # This document
  eval/
    run_eval.py              # Main entry point (recommended): --benchmark hotpotqa | browsecomp_plus
    run_hotpotqa.py          # Legacy entry point (HotpotQA only), still supported
    run_judge.py             # Offline LLM-as-judge rescoring
  train/
    sample_rollouts.py       # Stage 1: K rollouts/example, bucketed by reward
    distill_e1.py            # Stage 2: success × failure → E1 (Table 11)
    cluster_e2.py            # Stage 3: cluster E1 → E2 (Ward + Table 12)
    build_hek.py             # Orchestrator: runs stages 1/2/3 in sequence
```

---

## 0. One-Time Setup (Shared by Both Tracks)

### 0.1 Prepare the Data

```bash
source .venv/bin/activate
bash scripts/data_prep/prepare.sh hiexp
```

Outputs (HotpotQA and BCP each have their own subtree):

```
data/hiexp/
  train/
    r1_searcher_stage2.jsonl              # 8,148 rows; paper §A training set (HotpotQA+2Wiki samples selected by R1-Searcher)
  eval/
    hotpotqa_dev_sample1000_seed42.jsonl  # 1,000 rows; paper §A protocol

data/BCP/                                 # Native BCP format (TSV + TREC qrel), read directly by the BCP runner and our judge
  data/
    browsecomp_plus_decrypted.jsonl       # All 830 examples (gold answers + gold/evidence docids)
  topics-qrels-200/                       # Direct mirror of third_party/.../topics-qrels-200/ (paper's fixed eval split)
    queries.tsv  qrel_evidence.txt  qrel_golds.txt
  topics-qrels-train500/                  # 500 randomly sampled by build_bcp_data.py from the non-eval pool (seed=42)
    queries.tsv  qrel_evidence.txt  qrel_golds.txt
```

> `topics-qrels-train500` and `topics-qrels-200` are strictly disjoint: before sampling, the non-eval pool is filtered by the `query_id`s in `eval200` (`build_bcp_data.py:_read_query_ids` → `pool = [r for r in rows if r.query_id not in eval_ids]`). To verify that the intersection is empty yourself, see the short script in §B.0.

### 0.2 Index (Goal A Only)

Encode `corpus.jsonl` (1.6M passages) into e5-base shards once. See `scripts/build_index/README.md` for the complete procedure. This machine already has `data/wiki2018-corpus/{corpus.jsonl, corpus/, shards-intfloat-multilingual-e5-base/}`.

Goal B uses BCP's built-in BM25; the BCP repository manages its own index (see `third_party/BrowseComp-Plus/docs/qwen.md`).

### 0.3 `.env` (Project Root)

Use the following as the shared baseline for both tracks, then override track-specific values with `KEY=val` at the start of each command.

```dotenv
# ── LLM (OpenAI-compatible, e.g. local vLLM) ──
HIEXP_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
HIEXP_LLM_BASE_URL=http://127.0.0.1:8000/v1
HIEXP_LLM_API_KEY=EMPTY

# ── Sampling settings from paper §4.1 ──
HIEXP_TEMPERATURE=1.0
HIEXP_TOP_P=0.95
HIEXP_MAX_RESPONSE_TOKENS=8192

# The agent always uses the OpenAI Responses API (/v1/responses).
# For reasoning models, the chain of thought is stored in the trajectory as a ``reasoning`` block;
# this is transparent for non-reasoning models. Requires vLLM 0.19+.

# ── Search-o1 multi-hop QA defaults (values not specified in the paper) ──
HIEXP_MAX_SEARCH_LIMIT=10
HIEXP_MAX_TURN=15

# ── Retrieval MCP (run only one at a time; switch according to the Goal) ──
HIEXP_MCP_URL=http://127.0.0.1:8080/mcp
HIEXP_MCP_TRANSPORT=sse
HIEXP_MCP_SEARCH_TOOL=search

# ── HEK (HiExp private experience base; switch according to the Goal) ──
HIEXP_HEK_EMBEDDING_MODEL=intfloat/multilingual-e5-base
HIEXP_E1_THRESHOLD=0.8
HIEXP_E1_TOP_K=3
HIEXP_E2_TOP_K=5
HIEXP_USE_E3=false
HIEXP_ENABLE_HEK=true
HIEXP_HEK_PATH=          # Leave empty; pass --hek-path explicitly in each evaluation command

HIEXP_SEED=42
HIEXP_CACHE_DIR=./.hiexp_cache
```

> **Temperature** `T=1.0`, **top_p** `0.95`, and **max_response** `8192` come from paper §4.1. The `E1` threshold of `0.8` and `E2` top-5 come from §A. `max_search_limit=10` and `max_turn=15` are the official Search-o1 multi-hop QA defaults (not specified in the paper).

---

## Goal A — HotpotQA (Paper §A Protocol)

### A.1 Start the Services (Brief Version)

- **LLM**: Run `Qwen2.5-7B-Instruct` on `:8000` with vLLM. We recommend `--enable-auto-tool-choice --tool-call-parser hermes`; without these options, the agent falls back to parsing text-form `<tool_call>` blocks.
- **MCP**: Serve the e5-base + 2018 Wiki index on `:8080` with `bash scripts/build_index/serve_mcp.sh`.

### A.2 Baseline Smoke Test (50 Questions, No HEK)

```bash
source .venv/bin/activate
python scripts/hiexp_prompt/eval/run_eval.py \
    --benchmark hotpotqa \
    --num-samples 50 \
    --output-dir runs/hotpotqa/qwen2.5-7b/baseline_smoke \
    --no-enable-hek \
    --max-search-limit 10 --max-turn 15
```

Expected: the `f1 / cem / em` values in `runs/hotpotqa/qwen2.5-7b/baseline_smoke/metrics.json` should be in the same range as the paper's Search-o1\* row (44.4 / 41.2 / 34.2). If they differ substantially, first inspect retrieval recall (the length of `retrieved_docids` and whether the first hit is relevant to the gold answer) and verify that tool calling was actually triggered.

### A.3 Build the HEK (Paper §A: R1-Searcher stage_2 + MuSiQue)

```bash
python scripts/hiexp_prompt/train/build_hek.py \
    --hek-path data/hek/paper_mix \
    --benchmark hotpotqa \
    --training-data paper_mix \
    --num-train -1 \
    --rollouts-per-example 8 \
    --shard-size 1000 \
    --cluster-threshold 1.5
```

Parameters:
- `--benchmark hotpotqa` → Use the HotpotQA spec (`hiexp_boxed` template + `WordLevelScorer` CEM bucketing). The question template and scorer are both loaded from the registry, so they do not need to be passed separately.
- `--training-data paper_mix` → Use the paper §A data: `R1-Searcher stage_2` (8,148) + MuSiQue (8,000 random samples, seed=42). The former is preloaded at `data/hiexp/train/r1_searcher_stage2.jsonl`; FlashRAG MuSiQue is loaded from Hugging Face. To skip R1-Searcher entirely and use a FlashRAG-only approximation, replace `paper_mix` with `paper_mix_approx`.
- `--num-train -1` → Use all ~16K samples. To validate the pipeline first, try a small-scale run with `--num-train 200` (see `06_hotpotqa_reproduction_plan.md` §10).
- `--rollouts-per-example 8` → Sample eight trajectories per question. Paper §C reports ~7,000 trajectories on MuSiQue, roughly corresponding to the success/failure pairing rate at K≈1; K=8 is a reasonable engineering default.
- `--shard-size 1000` → Start a new `rollouts-NNNNN.jsonl` file every 1,000 records to avoid an oversized single file at the scale of `paper_mix`; `distill_e1.py` transparently merges all shards in the directory.
- `--cluster-threshold 1.5` → Ward linkage distance threshold; the paper does not specify a value.

> The former `--reward em` option has been replaced by `BenchmarkSpec.decisive_metric`. The HotpotQA spec defaults to `cem` (matching the primary metric in paper §4.1). To use EM instead, set `decisive_metric="em"` in the spec.

Output:

```
data/hek/paper_mix/
  rollouts/rollouts.jsonl     # One {pos: [...], neg: [...]} tuple per line
  e1.jsonl                    # Case-level experiences (Table 11 schema)
  e2.jsonl                    # Strategy-level experiences (Table 12 schema)
  .embeddings/                # e5-base passage embedding cache
```

Estimated budget: on one 80 GB A100, K=8 × ~16K questions × an average of 5–10 searches requires tens of GPU-hours. Validate the pipeline at small scale before scaling up.

### A.4 +HiExp Evaluation (1,000 Questions, HEK Enabled)

```bash
python scripts/hiexp_prompt/eval/run_eval.py \
    --benchmark hotpotqa \
    --num-samples 1000 \
    --hek-path data/hek/paper_mix \
    --enable-hek \
    --output-dir runs/hotpotqa/qwen2.5-7b/hiexp_e2_e1 \
    --max-search-limit 10 --max-turn 15
```

Expected: `f1 / cem / em` should approach the paper's +HiExp row (48.7 / 47.7 / 37.1), for relative gains of approximately +4.3 / +6.5 / +2.9.

### A.5 Compare

```bash
diff <(jq '. | {n, f1, cem, em}' runs/hotpotqa/qwen2.5-7b/baseline_smoke/metrics.json) \
     <(jq '. | {n, f1, cem, em}' runs/hotpotqa/qwen2.5-7b/hiexp_e2_e1/metrics.json)
```

Alternatively, run both groups (baseline + HiExp) for a small-scale ablation: expand the baseline to 1,000 questions as well and keep the output directories at the same level.

---

## Goal B — BrowseComp-Plus (Unified Agent, Pluggable Benchmark)

> "Fair" means that all HiExp-specific hyperparameters (sampling, E1/E2 thresholds/top-k, trajectory length, and clustering) match paper §4.1/§A. Only the benchmark's question template, answer extraction, and evaluation method change. The agent framework is identical to Goal A; the only difference is `--benchmark browsecomp_plus`.
>
> **End-to-end design document**: `docs/stage0/hiexp/08_bcp_integration_pipeline.md` (M8). This section is a command-focused quick reference; if anything is ambiguous, follow the M8 document.

There are only three differences from Goal A, all injected by `BenchmarkSpec` (`src/hiexp/evaluation/benchmarks.py`):

| Dimension | Goal A (hotpotqa) | Goal B (browsecomp_plus) |
|---|---|---|
| Question template | `hiexp_boxed` (`\boxed{}`) | `browsecomp_plus` (`Exact Answer:`) |
| Answer extractor | `extract_boxed_answer` | `extract_exact_answer` |
| Scorer | `WordLevelScorer` (EM/CEM/F1) | `JudgeScorer` (BCP `GRADER_TEMPLATE`) |

Pipeline (same structure as Goal A; only `--benchmark` changes):

```
data/BCP/topics-qrels-200       (eval, 200)
data/BCP/topics-qrels-train500  (train, 500)
                            │
                            ▼
sample_rollouts.py --benchmark browsecomp_plus    # K rollouts via HiExpAgent
                            │                       Each rollout is scored online by JudgeScorer
                            ▼
data/hek/bcp/rollouts/rollouts-NNNNN.jsonl        # Sharded JSONL
                            │
                            ▼
distill_e1.py + cluster_e2.py  →  data/hek/bcp/{e1.jsonl, e2.jsonl, *.npy}
                            │
                            ▼
run_eval.py --benchmark browsecomp_plus --enable-hek    # Same agent, HEK enabled
                            │
                            ▼
runs/bcp/.../hiexp/{run_*.json, metrics.json}
                            │
                  diff metrics.json (baseline vs. +HiExp)
```

### B.0 Data Sanity Check (Optional)

Confirm that train-500 and eval-200 do not overlap:

```bash
python - <<'EOF'
from pathlib import Path
def qids(p): return {ln.split("\t",1)[0] for ln in Path(p).read_text().splitlines() if ln.strip()}
e = qids("data/BCP/topics-qrels-200/queries.tsv")
t = qids("data/BCP/topics-qrels-train500/queries.tsv")
print(f"|eval200|={len(e)}  |train500|={len(t)}  overlap={len(e & t)}")
EOF
# Expected: |eval200|=200  |train500|=500  overlap=0
```

### B.1 Start the Services

- **LLM (agent)**: Use the same port as Goal A (`:8000`, vLLM).
- **MCP**: Run BCP's built-in BM25 (or Qwen3-Embedding) on `:8080` as described in `third_party/BrowseComp-Plus/docs/qwen.md`. **It cannot share port 8080 with Goal A's e5-base + Wiki MCP**—stop the old service before switching Goals.
- **Judge LLM**: Uses `HIEXP_JUDGE_*` by default and falls back to `HIEXP_LLM_*` when unset. Add the following to `.env`:

  ```dotenv
  # Empty = reuse the agent's vLLM for self-evaluation; to align with the paper,
  # point these variables to a stronger judge (e.g. GPT-4.1)
  HIEXP_JUDGE_MODEL=
  HIEXP_JUDGE_BASE_URL=
  HIEXP_JUDGE_API_KEY=
  HIEXP_JUDGE_CONCURRENCY=16
  ```

### B.2 Baseline Smoke Test (30 Questions, No HEK)

```bash
source .venv/bin/activate
python scripts/hiexp_prompt/eval/run_eval.py \
    --benchmark browsecomp_plus \
    --num-samples 30 \
    --output-dir runs/bcp/qwen2.5-7b/baseline_smoke \
    --no-enable-hek
```

Each question produces one `run_<query_id>.json` containing `prediction_raw / prediction_exact / scoring.{correct,judge_*}`. `metrics.json` reports `accuracy = n_correct / n`. **The judge is called online**, so there is no need to run `run_judge.py` afterward.

### B.3 Build the HEK (500 BCP Training Examples, Sharded Output)

```bash
python scripts/hiexp_prompt/train/build_hek.py \
    --hek-path data/hek/bcp \
    --benchmark browsecomp_plus \
    --training-data browsecomp_plus_train500 \
    --num-train -1 \
    --rollouts-per-example 4 \
    --shard-size 1000 \
    --cluster-threshold 1.5
```

Key points:

* `--benchmark browsecomp_plus` → Use the BCP spec: BCP template + `JudgeScorer`, fully automated.
* `--training-data browsecomp_plus_train500` → By default, join `data/BCP/topics-qrels-train500/queries.tsv` with `decrypted.jsonl`; use `--bcp-queries-tsv` to override the queries file explicitly.
* `--rollouts-per-example 4` → Start with K=4. At K=1, almost no examples satisfy `n_pos>0 and n_neg>0`; increase K if needed.
* `--shard-size 1000` → Write 1,000 records per `data/hek/bcp/rollouts/rollouts-NNNNN.jsonl` shard; `distill_e1.py` merges them transparently.
* There is no need to pass judge configuration separately—`build_hek.py` forwards `HIEXP_JUDGE_*` to `sample_rollouts.py`.

Output:

```
data/hek/bcp/
  rollouts/rollouts-00000.jsonl ...    # K=4, with n_pos+n_neg entries per query
  e1.jsonl                             # Case-level
  e2.jsonl                             # Strategy-level
  *.npy                                # Retrieval embeddings
```

### B.4 +HiExp Evaluation (All 200 Questions, Online Judge)

```bash
python scripts/hiexp_prompt/eval/run_eval.py \
    --benchmark browsecomp_plus \
    --hek-path data/hek/bcp \
    --enable-hek \
    --output-dir runs/bcp/qwen2.5-7b/hiexp_e2_e1
```

Omit `--num-samples` to run all 200 questions. By default, `data/BCP/topics-qrels-200/queries.tsv` is automatically joined with `decrypted.jsonl`; use `--bcp-queries-tsv` / `--bcp-decrypted-jsonl` to select a different split.

### B.5 Compare

```bash
diff <(jq '. | {n, accuracy}' runs/bcp/qwen2.5-7b/baseline_smoke/metrics.json) \
     <(jq '. | {n, accuracy}' runs/bcp/qwen2.5-7b/hiexp_e2_e1/metrics.json)
```

Expected: an increase in `accuracy` indicates that HEK is effective in the BCP domain. Expand the baseline to 200 questions as well for a small-scale ablation.

### B.6 Optional: Offline Rescoring / Cross-Validation

`run_eval.py` already scores online. Use `run_judge.py` only in the following cases:

* Rescore with a different judge model (for example, replace vLLM self-evaluation with GPT-4.1).
* Cross-validate against BCP's upstream `evaluate_run.py`.
* Evaluate run files produced separately by the BCP runner (`qwen_client_async.py`).

```bash
python scripts/hiexp_prompt/eval/run_judge.py \
    --input-dir  runs/bcp/qwen2.5-7b/hiexp_e2_e1 \
    --output-dir evals/bcp/qwen2.5-7b/hiexp_e2_e1 \
    --judge-model gpt-4.1 --judge-base-url https://api.openai.com/v1
```

Outputs are `*_eval.json`, `evaluation_summary.json`, and `detailed_judge_results.csv`, with fields matching BCP's own `evaluate_with_openai.py`.

### B.7 Add a New Benchmark

1. Add a `BenchmarkSpec(...)` entry to `_REGISTRY`.
2. Add a dataset loader to `_shared.py` (if the data source is also new).
3. Add a branch to both `run_eval.py:_load_examples` and `sample_rollouts.py:_load_training_examples`.

That's it. **No changes to the agent, trajectories, scoring, or HEK are required.**


## Debugging Individual Scripts

Each stage can run independently (without using `build_hek.py`):

```bash
# Stage 1: rollouts. Start with 5 questions × 2 rollouts to test connectivity
python scripts/hiexp_prompt/train/sample_rollouts.py \
    --out-dir data/hek/dryrun/rollouts \
    --benchmark hotpotqa --training-data hotpotqa \
    --num-train 5 --rollouts-per-example 2

# Stage 2: distill E1 (accepts a single file or a directory containing rollouts*.jsonl shards)
python scripts/hiexp_prompt/train/distill_e1.py \
    --rollouts-path data/hek/dryrun/rollouts \
    --hek-path data/hek/dryrun

# Stage 3: cluster E2
python scripts/hiexp_prompt/train/cluster_e2.py \
    --hek-path data/hek/dryrun --distance-threshold 1.5
```

```bash
# Evaluation: dry run (load the dataset without invoking the agent)
python scripts/hiexp_prompt/eval/run_eval.py \
    --benchmark hotpotqa --output-dir /tmp/eval_dryrun --dry-run

# Evaluation: resumable by default; reruns only failed or missing query_ids
python scripts/hiexp_prompt/eval/run_eval.py --benchmark hotpotqa ...   # --resume is the default
```

---

## Reproduction Limitations (See `06_hotpotqa_reproduction_plan.md` §2.2 and `07_m7_smoke_test.md` §6)

- The paper does not specify the rollout count K, clustering threshold τ_l, E1 top-k, or maximum turn count. The script defaults are K=8 / τ=1.5 / E1 top-k=3 / max_turn=15; all are configurable through the CLI.
- LLM-as-judge: HotpotQA does not need it (F1/CEM/EM are computed directly). BCP uses `scripts/hiexp_prompt/eval/run_judge.py` (implemented in M8, mirroring BCP's `evaluate_with_openai.py`). `run_eval.py` itself does not score, avoiding coupling between trajectory generation and the judge.
- The temporary model (`Qwen3.5-122B-A10B`) emits text-form `<tool_call>` blocks and mixed JSON+XML tool calls. The agent has a fallback for these (`graph._parse_text_tool_calls`); after switching back to `Qwen2.5-7B-Instruct`, this path naturally stops being triggered.

---

## Debugging Tips

- **Trajectories are too short / tools are never called**: if `prediction_raw` contains `<tool_call>` but `tool_call_counts={}`, the text fallback did not recognize it. Share the full block or add a test case to `tests/hiexp/test_text_tool_calls.py`.
- **Incorrect MCP response structure**: if `retrieved_docids=[]` and the `<tool_response>` content in `prediction_raw` is `[{"type":"text",...}]`, unpacking `TextContent` failed. Check the `graph._unwrap_mcp_text` unit tests.
- **Trajectories are too long (prompt overflow during distillation)**: reduce `distill_e1.py --max-chars-per-traj`.
- **Inspect the HEK**:
  ```bash
  wc -l data/hek/paper_mix/{e1.jsonl,e2.jsonl}
  jq -c '. | {level, title, tags}' data/hek/paper_mix/e2.jsonl | head
  ```
- **Resume from checkpoints**: `run_eval.py` defaults to `--resume` and does not rerun existing `run_<id>.json` files. Each of the three `build_hek.py` stages is also idempotent.
