#!/bin/bash
# HiExp end-to-end on BCP, Qwen3-Embedding-8B retriever.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "$REPO_DIR"
source .venv/bin/activate

HEK_PATH=data/hek/bcp/model_name

# TODO: add your own modelm (default is Qwen/Qwen3.6-35B-A3B)
# export AGENT_LLM_MODEL=""
# TODO: add your own base URL (default is http://127.0.0.1:8000/v1)
# export AGENT_LLM_BASE_URL="https://openrouter.ai/api/v1
# TODO: add your own API key
# export AGENT_LLM_API_KEY=""

export HIXEP_MCP_URL=http://127.0.0.1:8080/mcp

python scripts/hiexp_prompt/train/build_hek.py \
    --hek-path "$HEK_PATH" \
    --benchmark bcp \
    --retriever qwen3-8b \
    -o agent.max_search_limit=50 \
    -o agent.max_turn=50
    # TODO: Please uncomment this if you are using a model that supports responses API
    # -o llm.api_flavor=responses

python scripts/hiexp_prompt/eval/run_eval.py \
    --benchmark bcp \
    --retriever qwen3-8b \
    --enable-hek \
    --hek-path "$HEK_PATH" \
    --output-dir runs/bcp/model_name/hiexp_e2_e1 \
    -o agent.max_search_limit=50 \
    -o agent.max_turn=50
    # TODO: Please uncomment this if you are using a model that supports responses API
    # -o llm.api_flavor=responses

echo "Please rejudge the runs by running the following command:"
echo "bash scripts/experiments/rejudge/rejudge_ace_hiexp.sh runs/bcp/model_name/hiexp_e2_e1"