#!/usr/bin/env bash
set -eo pipefail

# Usage: ./eval_fire_risk.sh MONTH [OUTPUT_DIR]

output_dir="${2:-./angstrom_fire_output}"

source /home/kris/miniconda3/etc/profile.d/conda.sh
conda activate analytics

set -u

run_timestamp="$(date +%Y-%m-%d_%H-%M-%S)"
mkdir -p "$output_dir"
mkdir -p logs

for year in {2002..2022}; do
    echo "========================================"
    echo "Starting wildfire backtest for Year ${year}"
    echo "========================================"
    for month in {1..12}; do
        echo "========================================"
        echo "Starting wildfire backtest for month ${month}"
        echo "========================================"

        python ./get_angstrom_index.py \
            --surface-model-dir /mnt/c/Users/Kris/Documents/amazonforecast/backup/monthly \
            --fcst-init-dat "$year" "$month" \
            --output-dir "$output_dir" \
            > "./logs/angstrom_index_${year}_${month}_${run_timestamp}.log" 2>&1

        echo "Finished angstrom fire index for ${year} ${month}"
    done
done


# nohup python ./get_angstrom_index.py \
#     --surface-model-dir /mnt/c/Users/Kris/Documents/amazonforecast/backup/monthly \
#     --fcst-init-date 2022 03 \
#     --output-dir "$output_dir" \
#     > "./logs/angstrom_index_${run_timestamp}.log" 2>&1