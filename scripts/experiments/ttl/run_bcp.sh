#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "$REPO_DIR"
[[ -d .venv ]] && source .venv/bin/activate

# TODO: add your own modelm (default is Qwen/Qwen3.6-35B-A3B)
# export AGENT_LLM_MODEL=""
# TODO: add your own base URL (default is http://127.0.0.1:8000/v1)
# export AGENT_LLM_BASE_URL="https://openrouter.ai/api/v1"
# TODO: add your own API key
# export AGENT_LLM_API_KEY=""

export LANGSMITH_TRACING=true

SAVE_PATH="runs/ttl/model_name"

python scripts/ttl/run.py \
    --benchmark bcp \
    --retriever qwen3-8b \
    --save-path "${SAVE_PATH}" \
    -o data.train_batch_size=4 \
       rollout.n=4 \
       rollout.concurrency=16 \
       actor.mini_batch_size=16 \
       actor.micro_batch_size=4 \
       trainer.total_epochs=1 \
       trainer.save_freq=5 \
       data.splits.train=data/BCP/topics-qrels-ttl-train100/queries.tsv \
       data.splits.val=data/BCP/topics-qrels-ttl-val100/queries.tsv \
       eval.eval_every_n_steps=5 \
       reflector.enable_thinking=true \
       curator.enable_thinking=true \
       # TODO: Please uncomment this if you are using a model that supports responses API
       # agent.llm_api_flavor=responses
       # reflector.llm_api_flavor=responses
       # curator.llm_api_flavor=responses

export EVAL_JUDGE_MODEL="openai/gpt-5.4-mini"
export EVAL_JUDGE_BASE_URL="https://openrouter.ai/api/v1"
export EVAL_JUDGE_PROVIDER="openrouter"
# TODO: add your own API key
export EVAL_JUDGE_API_KEY=""
export OPENROUTER_PROVIDER="openai"
export OPENROUTER_PROVIDER_FALLBACKS=false

EXP_PATH="${SAVE_PATH}/experience/final.md"
export LANGSMITH_TRACING=false

python scripts/ttl/eval_experience.py \
    --experience-path "${EXP_PATH}" \
    --split test \
    --tag final_test \
    --save-path "${SAVE_PATH}" \
    --benchmark bcp \
    --retriever qwen3-8b \
    -o eval.concurrency=16 \
       agent.max_search_limit=50 \
       agent.max_turn=50 \
       data.splits.test=data/BCP/topics-qrels-ttl-test630/queries.tsv
       # TODO: Please uncomment this if you are using a model that supports responses API
       # agent.llm_api_flavor=responses
       # reflector.llm_api_flavor=responses
       # curator.llm_api_flavor=responses