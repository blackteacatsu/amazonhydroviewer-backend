#!/usr/bin/env bash

# -----------------------------------------------
# Only source code lives in the repository.
# Outputs and temporary data live on /mnt/vast.
# Override locations with:
    # OUTPUT_DIR=/absolute/path 
    # WORK_DIR=/absolute/scratch/path.
# START_YEAR/END_YEAR select a year range; 
# later years require the preceding daily output.
# -----------------------------------------------

# Stop on a failed command, an unset variable,
# or a failure anywhere in a pipeline.
set -eou pipefail

# Load Conda for noninteractive jobs and use the dependencies
source /home/local/WIN/qsu4/miniconda3/etc/profile.d/conda.sh
conda activate analytics


# This script lives in scripts/shell/fwi; the Python runner lives in scripts/python.
# Resolve from the script's location rather than your terminal's working directory.
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
runner="$scripts_dir/../../python/get_fireW_index.py"


# Use the active environment's Python, 
# or an explicit interpreter supplied via PYTHON.
python_bin="${PYTHON:-python}"

# Read-only annual input archives. retro_dir can override this default.
retro_dir="${RETRO_DIR:-/mnt/vast/prakrut/backup/lis_runs/malaria_amazon/retrospective}"


dc_method="${DC_METHOD:-original}"
# Pass the selected method to Python and keep its outputs separate.
method_args=(--dc-method "$dc_method")

case "$dc_method" in 
    original) method_dir=original_dc ;;
    SoilMoist_inst)
        method_dir=soilMoist_dc
        : "${SOIL_SATURATION:? Set SOIL_SATURATION to the MAXSMC NetCDF used by retrospective}"

        [[ "$SOIL_SATURATION" == /* && -f "$SOIL_SATURATION" && -r "$SOIL_SATURATION" ]] || {
            echo "SOIL_SATURATION must be an absolute path to a readable NetCDF file." >&2
            exit 2
        }
        method_args+=(--soil-saturation "$SOIL_SATURATION")
        ;;
    *) echo "Unsupported DC_METHOD : $dc_method" >&2; 
    exit 2
    ;;
esac
output_root="${OUTPUT_DIR:-/mnt/vast/ksu/lis_products/fwi_amazon/retrospective/daily}"
output_dir="$output_root/$method_dir"

work_dir="${WORK_DIR:-$output_dir/.work}" # Temporary annual extraction folders; use node-local scratch.


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
for location in "$retro_dir" "$output_dir" "$work_dir"; do
    [[ "$location" == /* ]] || { echo "Use an absolute data path: $location" >&2; exit 2; }
done
[[ -f "$runner" ]] || { echo "Missing Python runner: $runner" >&2; exit 1; }

# Keep Python bytecode and Dask spill files out of the code checkout.
export PYTHONDONTWRITEBYTECODE=1
export DASK_TEMPORARY_DIRECTORY="$work_dir/dask"
mkdir -p "$output_dir" "$work_dir" "$DASK_TEMPORARY_DIRECTORY"

# Process years sequentially: 
# each year depends on the previous year's moisture codes.
for ((year=start_year; year<=end_year; year++)); do
    archive="$retro_dir/SURFACEMODEL_${year}.tar.gz"
    [[ -f "$archive" ]] || { echo "Missing archive: $archive" >&2; exit 1; }

    # Create a uniquely named directory 
    # owned by this run, so cleanup cannot remove
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
        --mode retrospective \
        --input-dir "$batch/outputs/retrospective/SURFACEMODEL" \
        --output-dir "$output_dir/$year" \
        --restart-root "$output_dir" \
        "${method_args[@]}" \
        "${init_args[@]}" "$@"

    # Reached only after a successful Python run. 
    # Delete this temporary extraction;
    # retain the original tar.gz archives and all daily output files.
    rm -rf -- "$batch"

done
