#!/bin/bash
#
# prepare.sh — Fetch / decrypt / build datasets for Preliminary-stage projects.
#
# Usage:
#   bash scripts/data_prep/prepare.sh                      # prepare all
#   bash scripts/data_prep/prepare.sh all                  # prepare all
#   bash scripts/data_prep/prepare.sh browsecomp-plus      # just one
#   bash scripts/data_prep/prepare.sh ace search-r1        # multiple

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ── Registry ────────────────────────────────────────────────────────
# Each entry: name|script|data_dir (relative to PROJECT_ROOT)
# The dispatcher `cd`s into data_dir and runs the script there, so
# per-project scripts can assume CWD == data_dir (handy when upstream
# scripts hardcode relative paths).
ENTRIES=(
    "browsecomp-plus|scripts/data_prep/third_party/browsecomp-plus.sh|third_party/BrowseComp-Plus"
    "hiexp|scripts/data_prep/third_party/hiexp.sh|data/hiexp"
    "ttl|scripts/data_prep/ttl/prepare.sh|data/BCP"
)
# ────────────────────────────────────────────────────────────────────

get_field() { echo "$1" | cut -d'|' -f"$2"; }

lookup_entry() {
    local target="$1"
    for entry in "${ENTRIES[@]}"; do
        local name
        name=$(get_field "$entry" 1)
        if [[ "$name" == "$target" ]]; then
            echo "$entry"
            return 0
        fi
    done
    return 1
}

prepare_one() {
    local entry="$1"
    local name script data_dir
    name=$(get_field "$entry" 1)
    script=$(get_field "$entry" 2)
    data_dir=$(get_field "$entry" 3)

    local abs_script="$PROJECT_ROOT/$script"
    local abs_data_dir="$PROJECT_ROOT/$data_dir"

    if [[ ! -f "$abs_script" ]]; then
        echo "[ERROR] Script not found: $abs_script"
        return 1
    fi

    echo "================================================================"
    echo " Preparing: $name"
    echo " Script:    $script"
    echo " Data dir:  $data_dir"
    echo "================================================================"

    mkdir -p "$abs_data_dir"
    (cd "$abs_data_dir" && bash "$abs_script")

    echo ""
    echo "[OK] $name data prepared."
    echo ""
}

# ── Main ────────────────────────────────────────────────────────────
targets=("$@")
if [[ ${#targets[@]} -eq 0 ]] || [[ "${targets[0]}" == "all" ]]; then
    targets=()
    for entry in "${ENTRIES[@]}"; do
        targets+=("$(get_field "$entry" 1)")
    done
fi

all_names=()
for entry in "${ENTRIES[@]}"; do
    all_names+=("$(get_field "$entry" 1)")
done

failed=()
for t in "${targets[@]}"; do
    entry=$(lookup_entry "$t") || {
        echo "[ERROR] Unknown project: '$t'. Available: ${all_names[*]}"
        failed+=("$t")
        continue
    }
    prepare_one "$entry" || failed+=("$t")
done

echo "================================================================"
if [[ ${#failed[@]} -eq 0 ]]; then
    echo " All done. Prepared: ${targets[*]}"
else
    echo " Finished with errors. Failed: ${failed[*]}"
    exit 1
fi
echo "================================================================"
