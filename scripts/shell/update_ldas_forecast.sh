#!/usr/bin/env bash
set -eo pipefail

# start working venv for the updater
source /home/local/WIN/qsu4/miniconda3/etc/profile.d/conda.sh
conda activate analytics

set -u

current_date_time="$(date +%Y-%m-%d)"

# This script lives in scripts/shell; the Python runner lives in scripts/python.
# Resolve from the script's location rather than shell's working directory.
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
scripts_dir="$(cd -- "$script_dir/.." && pwd)"
runner="$scripts_dir/python/updater.py"

# Read monthly-forecast from mounted drive.
fcst_dir="/mnt/vast/prakrut/backup/lis_runs/malaria_amazon/forecast/monthly"

# Use the active environment's Python, via PYTHON
python_bin="${PYTHON:-python}"

mkdir -p logs

"$python_bin" "$runner" \
    --surface-model-dir "$fcst_dir" \
    --hcst-start-year 2001 \
    --hcst-end-year 2020 \
    --dask-workers 4 \
    --worker-memory "32GiB" \
    > ./logs/forecast_update_ver_${current_date_time}.log 2>&1 

# push new data to github data repo
git push

echo "Finished Processing"
