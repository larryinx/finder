# FINDER

**FINDER: Studying and Repairing What Test-Time Learning Changes in Deep-Research Agents**

This repository contains:

| Path | What it is |
|---|---|
| `src/search_ttl/` | **TTL-Search** — a training-free test-time learning loop for a search agent. Each step rolls out the agent, scores the rollouts, reflects on them, and lets a *curator* LLM edit a markdown **experience playbook** that is injected into the agent's system prompt. |
| `src/hiexp/`, `src/hiexp_prompt/`, `scripts/hiexp_prompt/` | **HiExp** baseline (hierarchical experience knowledge base: build a HEK, then evaluate with it). |
| `src/ace_react/`, `third_party/ace/`, `scripts/ace/` | **ACE** baseline (agentic context engineering) on the same agent / benchmark. |
| `third_party/BrowseComp-Plus/` | The **BrowseComp-Plus** benchmark: decrypted questions, corpus, prebuilt retrieval indexes, and the MCP search server the agents talk to. |
| `config/{ttl,ace,hiexp_prompt}/` | YAML config groups for each method. |
| `scripts/experiments/` | End-to-end experiment scripts (train / eval / re-judge). |
| `doc/` | Developer wiki — architecture, configuration, and **how to add a new reward / reflector / curator / experience / benchmark / tool provider**. |

## 1. Environment setup

Prerequisites: [uv](https://docs.astral.sh/uv/), [conda](https://docs.conda.io/) (only for the BrowseComp-Plus tooling), a CUDA GPU for the dense retriever, and an LLM endpoint — either an OpenAI-compatible server (e.g. [vLLM](https://docs.vllm.ai/)) or [OpenRouter](https://openrouter.ai/).

```bash
# 1. Python environment for this repo (Python ≥ 3.12; installs search_ttl,
#    hiexp, hiexp_prompt and ace_react as editable packages + dev tools)
uv sync
source .venv/bin/activate

# 2. Conda env `bcp` for BrowseComp-Plus (Python 3.11 + openjdk 21 + faiss-gpu
#    + uv, then the pinned requirements in scripts/venv_prep/bcp/requirements.txt)
bash scripts/venv_prep/prepare.sh

# 3. Environment variables
cp .env.example .env
```

Edit `.env`:

| Variable | Purpose |
|---|---|
| `AGENT_LLM_MODEL`, `AGENT_LLM_BASE_URL`, `AGENT_LLM_API_KEY` | The agent's LLM (OpenAI-compatible endpoint). The reflector, curator and judge fall back to these unless overridden with `REFLECTOR_LLM_*`, `CURATOR_LLM_*`, `EVAL_JUDGE_*`. |
| `AGENT_LLM_PROVIDER` | `openai_compat` (default) or `openrouter`. |
| `AGENT_MCP_URL` | The BrowseComp-Plus search server (see §3). Default `http://127.0.0.1:8080/mcp`; transport via `AGENT_MCP_TRANSPORT` (default `sse`). |
| `EVAL_JUDGE_MODEL`, `EVAL_JUDGE_BASE_URL`, `EVAL_JUDGE_API_KEY`, `EVAL_JUDGE_PROVIDER` | The LLM-as-judge used for scoring. The experiment scripts point it at a stronger model than the agent. |
| `SERPER_API_KEY` | Only for open-web runs with `--retriever serper` (not needed for BrowseComp-Plus). |
| `LANGSMITH_*`, `WANDB_*`, `WEAVE_*` | Optional tracing. |

> **Note on the conda step.** `scripts/venv_prep/bcp/browsecomp-plus.sh` sets `CONDA_OVERRIDE_CUDA=12.6` so `faiss-gpu` resolves on machines without a visible GPU driver (e.g. a cluster login node). Override it if your driver reports a different CUDA major/minor.

## 2. Data preparation

```bash
bash scripts/data_prep/prepare.sh          # all three targets, in order
# or one at a time:
bash scripts/data_prep/prepare.sh browsecomp-plus
bash scripts/data_prep/prepare.sh hiexp
bash scripts/data_prep/prepare.sh ttl
```

| Target | Runs in | What it does |
|---|---|---|
| `browsecomp-plus` | conda env `bcp`, inside `third_party/BrowseComp-Plus/` | Decrypts the dataset (`data/browsecomp_plus_decrypted.jsonl`), caches the corpus from Hugging Face, downloads the prebuilt indexes into `third_party/BrowseComp-Plus/indexes/` (BM25 + Qwen3-Embedding 0.6B/4B/8B — several GB), and splits topics/qrels. |
| `hiexp` | `.venv` | Builds the HiExp training / eval sets under `data/hiexp/` and the BCP-native layout under `data/BCP/` (decrypted jsonl + `topics-qrels-*` splits). |
| `ttl` | `.venv` | Partitions the 830 BCP questions into the disjoint TTL splits `data/BCP/topics-qrels-ttl-{train100,val100,test630}/`. |

Resulting layout (the parts the experiment scripts read):

```
data/BCP/
  data/browsecomp_plus_decrypted.jsonl
  topics-qrels-ttl-train100/queries.tsv
  topics-qrels-ttl-val100/queries.tsv
  topics-qrels-ttl-test630/queries.tsv
data/hiexp/{train,eval}/
third_party/BrowseComp-Plus/indexes/{bm25,qwen3-embedding-8b,...}/
```

## 3. Start the search (MCP) server

The agents retrieve from the BrowseComp-Plus corpus through an MCP server. Run it from the benchmark directory in the `bcp` env (needs a GPU for the dense index + embedding model):

```bash
cd third_party/BrowseComp-Plus
conda activate bcp
python searcher/mcp_server.py --searcher-type faiss \
    --index-path "indexes/qwen3-embedding-8b/corpus.shard*.pkl" \
    --model-name "Qwen/Qwen3-Embedding-8B" --normalize --port 8080
```

The server speaks the `sse` transport at `/mcp` by default, which is what `AGENT_MCP_URL=http://127.0.0.1:8080/mcp` / `AGENT_MCP_TRANSPORT=sse` in `.env` expect. Useful flags: `--k` (hits per query, default 5), `--snippet-max-tokens`, `--transport streamable-http`. Every `scripts/ttl/run.py` invocation probes the server before doing anything else.

You also need the agent's LLM reachable at `AGENT_LLM_BASE_URL` — e.g. `vllm serve <model> --port 8000` — or set `AGENT_LLM_PROVIDER=openrouter` and point `AGENT_LLM_BASE_URL` at `https://openrouter.ai/api/v1`.

## 4. Running the experiments

Each script under `scripts/experiments/` is self-contained; open it first — the `# TODO` lines are where you set the agent model / base URL / API key, the judge, and `SAVE_PATH`.

| Script | What it does |
|---|---|
| `scripts/experiments/ttl/eval_bcp_baseline.sh` | Baseline: evaluates the agent with an **empty** playbook on `test630`. |
| `scripts/experiments/ttl/run_bcp.sh` | **TTL-Search**: trains on `train100` (periodic eval on `val100`), then evaluates the final playbook on `test630`. |
| `scripts/experiments/ace/run_bcp_offline_gt.sh` | ACE baseline (offline mode, ground truth shown to its reflector). |
| `scripts/experiments/hiexp/run_bcp_hek_r8b.sh` | HiExp baseline: builds a HEK on BCP, then evaluates with it (Qwen3-Embedding-8B retriever). |
| `scripts/experiments/hiexp/run_bcp_hek_bm25.sh` | Same with the BM25 retriever. |
| `scripts/experiments/rejudge/rejudge_ace_hiexp.sh` | Re-scores finished ACE / HiExp run directories with the same LLM judge TTL-Search uses (`search_agent_lasj`), so every method is graded identically. |

### Driving TTL-Search directly

```bash
python scripts/ttl/run.py \
    --benchmark bcp --retriever qwen3-8b \
    --reward search_agent_lasj --reflector querychain_staged \
    --curator react_staged --experience staged_playbook \
    --save-path runs/ttl/my_run \
    -o data.train_batch_size=4 rollout.n=4 actor.mini_batch_size=16 actor.micro_batch_size=4

# 5-minute end-to-end smoke test
python scripts/ttl/run.py --benchmark bcp --retriever qwen3-8b --mode smoke --save-path runs/ttl/smoke

# Evaluate a frozen playbook on any split
python scripts/ttl/eval_experience.py --experience-path runs/ttl/my_run/experience/final.md \
    --split test --tag final_test --save-path runs/ttl/my_run
```

Every `--<group>` flag names a YAML under `config/ttl/<group>/`; `-o key=value ...` applies OmegaConf overrides last. A run directory looks like:

```
runs/ttl/my_run/
  experience/   step_0000_initial.md, step_NNNN_after_curator.md, best.md (+ best.meta.json), final.md
  steps/step_NNNN/   rollouts.jsonl, rewards.jsonl, reflections.jsonl, curator/summary_pass_*.json
  checkpoints/global_step_N/   experience.md, sampler_state.json, trainer_state.json
  eval/<tag>/   rollouts.jsonl, rewards.jsonl, metrics.json
```

## 5. Extending the code

The pipeline is made of pluggable parts — **reward**, **reflector**, **curator**, **experience** backend, plus benchmark loaders and search-tool providers — each registered under a string `kind` and selected from YAML. If you want to add or change any of them, start at **[`doc/README.md`](doc/README.md)**: it explains the architecture and configuration, and has a step-by-step page for each part.

## 6. Running the tests

The tests are offline (no LLM or MCP server needed) and run from the repo root:

```bash
uv run pytest                        # full suite; coverage report written to cov.xml
uv run pytest tests/search_ttl -q --no-cov   # just the TTL-Search tests, fast
uv run pytest tests/search_ttl/test_registry.py -q --no-cov
```

`tests/conftest.py` puts `third_party/ace` on `sys.path` for the ACE tests. Lint / format / type-check with `uv run ruff check .`, `uv run ruff format .`, `uv run ty check`.

## License

Released under the [Creative Commons Attribution 4.0](LICENSE) license.

## Citation

<!-- TODO: add the BibTeX entry for the paper. -->
