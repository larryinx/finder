#!/bin/bash
# HiExp end-to-end on BCP, BM25 retriever.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "$REPO_DIR"
source .venv/bin/activate

HEK_PATH=data/hek/bcp/model_name_bm25

# TODO: add your own modelm (default is Qwen/Qwen3.6-35B-A3B)
# export AGENT_LLM_MODEL=""
# TODO: add your own base URL (default is http://127.0.0.1:8000/v1)
# export AGENT_LLM_BASE_URL="https://openrouter.ai/api/v1
# TODO: add your own API key
# export AGENT_LLM_API_KEY=""

python scripts/hiexp_prompt/train/build_hek.py \
    --hek-path "$HEK_PATH" \
    --benchmark bcp \
    --retriever bm25 \
    # TODO: Please uncomment this if you are using a model that supports responses API
    # -o llm.api_flavor=responses

python scripts/hiexp_prompt/eval/run_eval.py \
    --benchmark bcp \
    --retriever bm25 \
    --enable-hek \
    --hek-path "$HEK_PATH" \
    --output-dir runs/bcp/model_name_bm25/hiexp_e2_e1 \
    -o agent.max_search_limit=50 \
    -o agent.max_turn=50
    # TODO: Please uncomment this if you are using a model that supports responses API
    # -o llm.api_flavor=responses
