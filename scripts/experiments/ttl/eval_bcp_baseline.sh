#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "$REPO_DIR"
[[ -d .venv ]] && source .venv/bin/activate

# TODO: add your own modelm (default is Qwen/Qwen3.6-35B-A3B)
# export AGENT_LLM_MODEL=""
# TODO: add your own base URL (default is http://127.0.0.1:8000/v1)
# export AGENT_LLM_BASE_URL="https://openrouter.ai/api/v1
# TODO: add your own API key
# export AGENT_LLM_API_KEY=""

export EVAL_JUDGE_MODEL="openai/gpt-5.4-mini"
export EVAL_JUDGE_BASE_URL="https://openrouter.ai/api/v1"
export EVAL_JUDGE_PROVIDER="openrouter"
# TODO: add your own API key
export EVAL_JUDGE_API_KEY=""
export OPENROUTER_PROVIDER="openai"
export OPENROUTER_PROVIDER_FALLBACKS=false

SAVE_PATH="runs/ttl/model_name"
EXP_PATH="${SAVE_PATH}/experience/baseline_empty.md"
export LANGSMITH_TRACING=false

mkdir -p "$(dirname "${EXP_PATH}")"
printf '(none yet)\n' > "${EXP_PATH}"

python scripts/ttl/eval_experience.py \
    --experience-path "${EXP_PATH}" \
    --split test \
    --tag baseline_test \
    --save-path "${SAVE_PATH}" \
    --benchmark bcp \
    --retriever qwen3-8b \
    -o eval.concurrency=16 \
    agent.max_search_limit=50 \
    agent.max_turn=50 \
    data.splits.test=data/BCP/topics-qrels-ttl-test630/queries.tsv
