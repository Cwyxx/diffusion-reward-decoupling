#!/bin/bash
# Batch driver for run-bestofn.sh across (method, dataset) combos.
#
# Usage:
#   bash evaluation/run-bestofn-batch.sh [gpus] [n_max] [methods...]
#
# Examples:
#   bash evaluation/run-bestofn-batch.sh
#   bash evaluation/run-bestofn-batch.sh 0,1,2,3 32 base-sd3
#   bash evaluation/run-bestofn-batch.sh 0,1,2,3 32 base-sd3 flowgrpo-pickscore-sd3 grpo-guard-sd3
#
# Defaults:
#   gpus=0,1,2,3,4,5,6,7
#   n_max=32
#   methods=base-sd3 flowgrpo-pickscore-sd3 grpo-guard-sd3 diffusion-dpo-sd3 realalign-sd3 diffusionnft-sd3 civitaialign-sd3

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

gpus="${1:-0,1,2,3,4,5,6,7}"
n_max="${2:-32}"

default_methods=(
    base-sd3
    flowgrpo-pickscore-sd3
    grpo-guard-sd3
    diffusion-dpo-sd3
    realalign-sd3
    diffusionnft-sd3
    civitaialign-sd3
)

if [ "$#" -ge 3 ]; then
    methods=("${@:3}")
else
    methods=("${default_methods[@]}")
fi

# unsafe_mscoco: benign MSCOCO captions as a clean safe-prompt baseline for the
# unsafe eval generate + sd-safety-checker + shieldgemma_bf16.
datasets=(unsafe_template unsafe_4chan unsafe_lexica unsafe_mscoco)

echo "Using gpus=${gpus}"
echo "Using n_max=${n_max}"
echo "Using methods: ${methods[*]}"
echo "Using datasets: ${datasets[*]}"

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
