#!/usr/bin/env bash
# Run from any directory, inside your compute-node allocation, with analytics active.
# Only source code lives in the repository. Outputs and temporary data live on /mnt/vast.
# Example: bash scripts/shell/eval_ldas_fwi_01_25.sh --dask-workers 4 --worker-memory 4GiB
# Override locations with OUTPUT_DIR=/absolute/path WORK_DIR=/absolute/scratch/path.
# START_YEAR/END_YEAR select a year range; later years require the preceding daily output.

# Stop on a failed command, an unset variable, or a failure anywhere in a pipeline.
set -euo pipefail

# This script lives in scripts/shell; the Python runner lives in scripts/python.
# Resolve from the script's location rather than your terminal's working directory.
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
scripts_dir="$(cd -- "$script_dir/.." && pwd)"
runner="$scripts_dir/python/get_fireW_index.py"

# Read-only annual input archives. ARCHIVE_DIR can override this default.
archive_dir="${ARCHIVE_DIR:-/mnt/vast/prakrut/backup/lis_runs/malaria_amazon/retrospective}"
output_dir="${OUTPUT_DIR:-/mnt/vast/ksu/amazon_wildfire/lis_fiw}"
work_dir="${WORK_DIR:-$output_dir/.work}" # Temporary annual extraction folders; use node-local scratch.

# Use the active environment's Python, or an explicit interpreter supplied via PYTHON.
python_bin="${PYTHON:-python}"

# A fresh run covers 2001–2025; 
# use START_YEAR to resume at a completed year boundary.
start_year="${START_YEAR:-2001}"
end_year="${END_YEAR:-2025}"

# Validate settings before creating directories or extracting large archives.
[[ "$start_year" =~ ^[0-9]{4}$ && "$end_year" =~ ^[0-9]{4}$ ]] || {
    echo "START_YEAR and END_YEAR must be four-digit years." >&2; exit 2;
}
(( start_year >= 2001 && end_year <= 2025 && start_year <= end_year )) || {
    echo "Require 2001 <= START_YEAR <= END_YEAR <= 2025." >&2; exit 2;
}
# Absolute data paths prevent accidental creation relative to the repository.
for location in "$archive_dir" "$output_dir" "$work_dir"; do
    [[ "$location" == /* ]] || { echo "Use an absolute data path: $location" >&2; exit 2; }
done
[[ -f "$runner" ]] || { echo "Missing Python runner: $runner" >&2; exit 1; }

# Keep Python bytecode and Dask spill files out of the code checkout.
export PYTHONDONTWRITEBYTECODE=1
export DASK_TEMPORARY_DIRECTORY="$work_dir/dask"
mkdir -p "$output_dir" "$work_dir" "$DASK_TEMPORARY_DIRECTORY"

# Process years sequentially: each year depends on the previous year's moisture codes.
for ((year=start_year; year<=end_year; year++)); do
    archive="$archive_dir/SURFACEMODEL_${year}.tar.gz"
    [[ -f "$archive" ]] || { echo "Missing archive: $archive" >&2; exit 1; }

    # Create a uniquely named directory owned by this run, so cleanup cannot remove
    # a pre-existing extraction folder. Retain it if extraction or calculation fails.
    batch="$(mktemp -d "$work_dir/fwi-${year}.XXXXXX")"
    echo "Extracting $archive into $batch"
    tar -xzf "$archive" -C "$batch"

    # Apply FFMC=85, DMC=6, DC=15 only at the beginning of the 2001 series.
    # Later years automatically load the preceding day's daily FWI output.
    init_args=()
    if [[ "$year" == 2001 ]]; then init_args=(--initialize); fi

    # Scan extracted daily files, write permanent results under OUTPUT_DIR/year,
    # and search all output years for the previous day's moisture codes.
    # "$@" forwards options such as --dask-workers and --worker-memory unchanged.
    "$python_bin" "$runner" \
        --input-dir "$batch/outputs/retrospective/SURFACEMODEL" \
        --output-dir "$output_dir/$year" \
        --restart-root "$output_dir" \
        "${init_args[@]}" "$@"

    # Reached only after a successful Python run. Delete this temporary extraction;
    # retain the original tar.gz archives and all daily output files.
    rm -rf -- "$batch"
done
