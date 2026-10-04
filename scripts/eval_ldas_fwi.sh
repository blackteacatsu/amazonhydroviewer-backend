#!/usr/bin/env bash
# Run daily FWI using dates stored in the input file. 
# Does not depend on the current directory.
# Usage: PYTHON=/path/to/python bash scripts/eval_ldas_fwi.sh [INPUT.nc] [OUTPUT_DIR]

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$script_dir/.." && pwd)"
input_file="${1:-/Users/kris/Documents/hydroclim/amazonforecast/data/202301/LIS_HIST_2023_Jan.nc}"
output_dir="${2:-$repo_dir/output/fwi}"
python_bin="${PYTHON:-python}"

# Use your activated analytics environment, or set PYTHON explicitly.
export PYTHONDONTWRITEBYTECODE=1

exec "$python_bin" "$script_dir/python/get_fireW_index.py" \
    --input "$input_file" \
    --initial-ffmc 85 --initial-dmc 6 --initial-dc 15 \
    --output-dir "$output_dir" "${@:3}"
