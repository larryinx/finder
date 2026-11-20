#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

PY="$PROJECT_ROOT/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
    echo "[ERROR] $PY not found. Run 'uv sync' from $PROJECT_ROOT first."
    exit 1
fi

TRAIN_OUT="$PWD/train"
EVAL_OUT="$PWD/eval"
BCP_OUT="$PROJECT_ROOT/data/BCP"

mkdir -p "$TRAIN_OUT" "$EVAL_OUT" "$BCP_OUT"

IFS=' ' read -ra TRAIN_EXTRA <<<"${HIEXP_DATA_PREP_TRAIN_FLAGS:-}"
IFS=' ' read -ra EVAL_EXTRA <<<"${HIEXP_DATA_PREP_EVAL_FLAGS:-}"

echo "── HEK training data (R1-Searcher stage_2) ──"
"$PY" "$PROJECT_ROOT/scripts/data_prep/hiexp/build_train.py" \
    --out-dir "$TRAIN_OUT" \
    "${TRAIN_EXTRA[@]}"

echo ""
echo "── HotpotQA eval split (paper §A: 1000 samples, seed 42) ──"
"$PY" "$PROJECT_ROOT/scripts/data_prep/hiexp/build_eval_hotpotqa.py" \
    --out-dir "$EVAL_OUT" \
    "${EVAL_EXTRA[@]}"

echo ""
echo "── BCP-native layout (data/BCP/) ──"
if [[ -f "$PROJECT_ROOT/third_party/BrowseComp-Plus/data/browsecomp_plus_decrypted.jsonl" ]]; then
    "$PY" "$PROJECT_ROOT/scripts/data_prep/hiexp/build_bcp_data.py" \
        --out-root "$BCP_OUT"
else
    echo "[skip] third_party/BrowseComp-Plus/data/browsecomp_plus_decrypted.jsonl missing —"
    echo "       run 'bash scripts/data_prep/prepare.sh browsecomp-plus' first"
fi

echo ""
echo "[OK] HiExp data ready under $PWD/{train,eval}/ and $BCP_OUT/"
