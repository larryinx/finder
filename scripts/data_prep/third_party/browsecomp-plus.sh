#!/bin/bash

set -euo pipefail
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate bcp
set -u

python scripts_build_index/decrypt_dataset.py \
    --output data/browsecomp_plus_decrypted.jsonl \
    --generate-tsv topics-qrels/queries.tsv

python - <<'PY'
from datasets import load_dataset
ds = load_dataset("Tevatron/browsecomp-plus-corpus", split="train")
print(f"[corpus] cached {len(ds)} docs at {ds.cache_files[0]['filename'] if ds.cache_files else '<unknown>'}")
PY

bash scripts_build_index/download_indexes.sh

python scripts_build_index/split_topics_qrels.py
