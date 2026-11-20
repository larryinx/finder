#!/bin/bash
# ACE on BCP — offline mode, ground-truth to Reflector, Qwen3-Embedding-8B retriever.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "$REPO_DIR"
source .venv/bin/activate

# TODO: add your own modelm (default is Qwen/Qwen3.6-35B-A3B)
# export AGENT_LLM_MODEL=""
# TODO: add your own base URL (default is http://127.0.0.1:8000/v1)
# export AGENT_LLM_BASE_URL="https://openrouter.ai/api/v1
# TODO: add your own API key
# export AGENT_LLM_API_KEY=""

SAVE_PATH="runs/ace/model_name"

python scripts/ace/run.py \
    --benchmark bcp \
    --retriever qwen3-8b \
    --mode offline \
    --save-path "${SAVE_PATH}" \
    -o data.splits.train=data/BCP/topics-qrels-ttl-train100/queries.tsv \
       data.splits.val=data/BCP/topics-qrels-ttl-val100/queries.tsv \
       ace.test_workers=16 \
       ace.eval_steps=100 ace.save_steps=25
       # TODO: Please uncomment this if you are using a model that supports responses API
       # llm.api_flavor=responses

echo "Please rejudge the runs by running the scripts/experiments/rejudge/rejudge_ace_hiexp.sh"