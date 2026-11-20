# FINDER: Studying and Repairing What Test-Time Learning Changes in Deep-Research Agents

![arXiv paper coming soon](https://img.shields.io/badge/arXiv-coming_soon-lightgrey?logo=arxiv)
[![Hugging Face artifacts](https://img.shields.io/badge/Hugging_Face-Artifacts-FFD21E?logo=huggingface)](https://huggingface.co/datasets/Yinxxx/finder-bcp-artifacts)

FINDER studies how search agents learn from past attempts on BrowseComp-Plus. After each batch of questions, TTL-Search uses the agent's search traces and answer scores to revise a playbook in its prompt.

## 📌 Set up the environment and data

You'll need Python 3.12+, uv, conda (for preparing the retriever MCP developed by BrowseComp-Plus), a CUDA GPU for the dense retriever, and access to an OpenAI-compatible LLM endpoint or OpenRouter. Run these commands from the repository root.

```bash
uv sync
bash scripts/venv_prep/prepare.sh
cp -n .env.example .env
bash scripts/data_prep/prepare.sh
```

The data script downloads the BrowseComp-Plus corpus and retrieval indexes, then builds the experiment splits. In `.env`, set `AGENT_LLM_MODEL`, `AGENT_LLM_BASE_URL`, and `AGENT_LLM_API_KEY` for your endpoint. For OpenRouter, also set `AGENT_LLM_PROVIDER=openrouter`.

## 🧭 Start the search server

Open a second terminal at the repository root and leave the server running there. The agent connects to it at `http://127.0.0.1:8080/mcp` by default.

```bash
cd third_party/BrowseComp-Plus
conda activate bcp
python searcher/mcp_server.py --searcher-type faiss \
    --index-path "indexes/qwen3-embedding-8b/corpus.shard*.pkl" \
    --model-name "Qwen/Qwen3-Embedding-8B" --normalize --port 8080
```

## 🔎 Run a smoke test

Back in the first terminal, run a two-step smoke test to check that the agent can search and update its playbook.

```bash
uv run python scripts/ttl/run.py --benchmark bcp --retriever qwen3-8b \
    --mode smoke --save-path runs/ttl/smoke
```

## 🧪 Run the experiments

Choose a BrowseComp-Plus experiment below. Open its script to check the model settings and replace any `model_name` placeholders in the paths.

| Experiment script | What it does |
| --- | --- |
| [ttl/run_bcp.sh](scripts/experiments/ttl/run_bcp.sh) | Runs TTL-Search on 100 training questions, checks the playbook periodically on 100 validation questions, and evaluates the final playbook on 630 test questions. |
| [ttl/eval_bcp_baseline.sh](scripts/experiments/ttl/eval_bcp_baseline.sh) | Evaluates the same agent on the 630 test questions with an empty playbook. |
| [ace/run_bcp_offline_gt.sh](scripts/experiments/ace/run_bcp_offline_gt.sh) | Runs the offline ACE baseline. Its reflector sees the reference answers during training. |
| [hiexp/run_bcp_hek_r8b.sh](scripts/experiments/hiexp/run_bcp_hek_r8b.sh) | Builds a hierarchical experience knowledge base (HEK) and evaluates HiExp with it, using the Qwen3-Embedding-8B search server. |
| [hiexp/run_bcp_hek_bm25.sh](scripts/experiments/hiexp/run_bcp_hek_bm25.sh) | Builds and evaluates a HiExp knowledge base using BM25. Start the search server with the BM25 index for this run. |
| [rejudge/rejudge_ace_hiexp.sh](scripts/experiments/rejudge/rejudge_ace_hiexp.sh) | Re-scores completed ACE or HiExp runs with the same LLM judge as TTL-Search. Pass one or more run directories as arguments. |

The TTL-Search and rejudging scripts use an OpenRouter judge for test scoring. Set `OPENROUTER_API_KEY` in your shell and replace the chosen script's empty key assignment with `export EVAL_JUDGE_API_KEY="$OPENROUTER_API_KEY"`.

To run the full TTL-Search experiment, set `SAVE_PATH` in `run_bcp.sh` and launch it from the repository root.

```bash
bash scripts/experiments/ttl/run_bcp.sh
```

### TTL-Search outputs

The final playbook and test scores are under `SAVE_PATH`. To see how a particular step changed the playbook, inspect its rollouts, rewards, and reflections in `steps/`.

```text
experience/final.md             Final playbook
steps/step_NNNN/                Rollouts, rewards, reflections, and curator edits
checkpoints/global_step_N/      Saved state for resuming training
eval/final_test/metrics.json    Scores on the held-out test set
```

## 🧩 5. Extend the code

TTL-Search uses YAML files in `config/ttl/` to select the reward, reflector, curator, and playbook implementation. Pass `-o key=value` to override a setting for one run. The [developer guide](doc/README.md) explains how these components fit together and how to add a component, benchmark, or search tool.

## 🔧 6. Run the tests

Run the offline tests from the repository root.

```bash
uv run pytest                               # Full suite with coverage
uv run pytest tests/search_ttl -q --no-cov  # TTL-Search tests only
```

Use `uv run ruff check .` for linting, `uv run ruff format .` for formatting, and `uv run ty check` for type checking.

## 🍺 Acknowledgments

This work builds on and is inspired by the following open-source projects:
- [BrowseComp-Plus](https://github.com/texttron/BrowseComp-Plus)
- [LangChain](https://github.com/langchain-ai)
- [ACE](https://github.com/ace-agent/ace)

## 📝 Citation

A BibTeX entry will be available when the paper is public on arXiv.


## 📄 License

FINDER is released under the [Creative Commons Attribution 4.0 International license](LICENSE).