#!/bin/bash
# Batch driver for run-bestofn.sh across (method, dataset) combos.
#
# Usage:
#   bash evaluation/run-bestofn-batch.sh [gpus] [n_max]
# Defaults:
#   gpus=0,1,2,3,4,5,6,7  n_max=32

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

gpus="${1:-0,1,2,3,4,5,6,7}"
n_max="${2:-32}"

methods=(base dpo inpo spo)
datasets=(geneval)

for method in "${methods[@]}"; do
    for dataset in "${datasets[@]}"; do
        echo ""
        echo "############################################"
        echo "# method=${method}  dataset=${dataset}"
        echo "############################################"
        bash "${SCRIPT_DIR}/run-bestofn.sh" "${gpus}" "${method}" "${dataset}" "${n_max}"
    done
done

echo ""
echo "All batch runs completed."
