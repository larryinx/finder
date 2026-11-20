#!/usr/bin/env bash
#
# Re-judge already-collected ACE / HiExp per-query run dirs with a FAIRER judge model

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "$REPO_DIR"
[[ -d .venv ]] && source .venv/bin/activate

export EVAL_JUDGE_MODEL="openai/gpt-5.4-mini"
export EVAL_JUDGE_BASE_URL="https://openrouter.ai/api/v1"
export EVAL_JUDGE_PROVIDER="openrouter"
# TODO: add your own API key
export EVAL_JUDGE_API_KEY=""

export OPENROUTER_PROVIDER="openai"
export OPENROUTER_PROVIDER_FALLBACKS=false

CONCURRENCY="${CONCURRENCY:-8}"

if [[ $# -gt 0 ]]; then
    RUN_DIRS=("$@")
else
    # TODO: add your own run directories
    RUN_DIRS=(
        "runs/ttl/model_name"
    )
fi

for d in "${RUN_DIRS[@]}"; do
    echo "================================================================"
    echo " Re-judge: $d  (judge=${EVAL_JUDGE_MODEL} via ${EVAL_JUDGE_PROVIDER})"
    echo "================================================================"
    python scripts/ttl/rejudge_runs.py "$d" \
        --reward search_agent_lasj \
        --concurrency "$CONCURRENCY"
    echo ""
done

echo "================================================================"
echo " All re-judged. Old metrics/rewards are in each dir's"
echo " rejudge_backup_<timestamp>/ subfolder."
echo "================================================================"
