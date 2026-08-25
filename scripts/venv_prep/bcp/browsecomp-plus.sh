#!/bin/bash
#
# browsecomp-plus.sh — build the `bcp` conda env for BrowseComp-Plus.
#
# Everything lives in ONE conda env, so a single `conda activate bcp`
# gives you python, java and faiss:
#   1. `conda create -n bcp python=3.11`
#   2. `conda install` openjdk 21 (Lucene / Pyserini BM25 needs java 21,
#      system java on the cluster is 11), faiss-gpu and uv — conda-forge
#   3. `uv pip install -r requirements.txt` into that env
#
# requirements.txt is a freeze of the working env (python 3.11,
# torch 2.7.0 cu126, tevatron pinned to a commit). flash-attn is not in
# it: searcher/searchers/faiss_searcher.py no longer defaults to
# flash_attention_2, so it is optional.
#
# prepare.sh runs this with cwd = third_party/BrowseComp-Plus, but
# nothing here depends on cwd; requirements.txt is found next to this file.
#
# Login nodes have no GPU, so conda cannot detect a CUDA driver and would
# refuse the faiss-gpu build. CONDA_OVERRIDE_CUDA tells the solver which
# driver to assume; 12.6 matches the nvidia-*-cu12 wheels in
# requirements.txt (conda still picks the newest compatible faiss build).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQ_FILE="$SCRIPT_DIR/requirements.txt"
ENV_NAME="bcp"
PY_VERSION="3.11"
export CONDA_OVERRIDE_CUDA="${CONDA_OVERRIDE_CUDA:-12.6}"

if ! command -v conda >/dev/null 2>&1; then
    echo "[bcp] ERROR: conda not found on PATH — bootstrap miniconda/miniforge first." >&2
    exit 1
fi

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "[bcp] conda env '$ENV_NAME' already exists — reusing it."
else
    echo "[bcp] creating conda env '$ENV_NAME' (python $PY_VERSION)..."
    conda create -n "$ENV_NAME" -c conda-forge "python=$PY_VERSION" -y
fi

echo "[bcp] installing openjdk 21, faiss-gpu, uv from conda-forge..."
conda install -n "$ENV_NAME" -c conda-forge openjdk=21 faiss-gpu uv -y

# Resolve the env prefix instead of `conda activate`: no shell hook needed,
# and the explicit --python keeps uv from targeting whatever VIRTUAL_ENV
# happens to be active in the caller's shell.
ENV_PREFIX="$(conda env list | awk -v n="$ENV_NAME" '$1 == n {print $NF}')"
echo "[bcp] installing $REQ_FILE into $ENV_PREFIX..."
"$ENV_PREFIX/bin/uv" pip install --python "$ENV_PREFIX/bin/python" -r "$REQ_FILE"

echo "[bcp] done — activate with: conda activate $ENV_NAME"
