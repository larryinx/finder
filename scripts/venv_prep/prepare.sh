#!/bin/bash
#
# prepare.sh — Install venvs for Preliminary-stage projects.
#
# Usage:
#   bash scripts/venv_prep/prepare.sh                      # install all
#   bash scripts/venv_prep/prepare.sh all                  # install all
#   bash scripts/venv_prep/prepare.sh browsecomp-plus      # install only browsecomp-plus
#   bash scripts/venv_prep/prepare.sh ace search-r1        # install ace and search-r1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ── Registry ────────────────────────────────────────────────────────
# Each entry: name|env_script|working_dir (relative to PROJECT_ROOT)
ENTRIES=(
    "browsecomp-plus|scripts/venv_prep/bcp/browsecomp-plus.sh|third_party/BrowseComp-Plus"
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

install_one() {
    local entry="$1"
    local name script workdir
    name=$(get_field "$entry" 1)
    script=$(get_field "$entry" 2)
    workdir=$(get_field "$entry" 3)

    local abs_script="$PROJECT_ROOT/$script"
    local abs_workdir="$PROJECT_ROOT/$workdir"

    if [[ ! -f "$abs_script" ]]; then
        echo "[ERROR] Script not found: $abs_script"
        return 1
    fi
    if [[ ! -d "$abs_workdir" ]]; then
        echo "[ERROR] Working dir not found: $abs_workdir"
        return 1
    fi

    echo "================================================================"
    echo " Installing: $name"
    echo " Script:     $script"
    echo " Working dir: $workdir"
    echo "================================================================"

    (cd "$abs_workdir" && bash "$abs_script")

    echo ""
    echo "[OK] $name installed."
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
    install_one "$entry" || failed+=("$t")
done

echo "================================================================"
if [[ ${#failed[@]} -eq 0 ]]; then
    echo " All done. Installed: ${targets[*]}"
else
    echo " Finished with errors. Failed: ${failed[*]}"
    exit 1
fi
echo "================================================================"
