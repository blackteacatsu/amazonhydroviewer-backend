#!/usr/bin/env bash
# Run inside your compute-node allocation with the analytics environment active.
# START_YEAR/END_YEAR allow restarting at a year boundary after a completed year.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
archive_dir="${ARCHIVE_DIR:-/mnt/vast/prakrut/backup/lis_runs/malaria_amazon/retrospective}"
output_dir="${OUTPUT_DIR:-$HOME/fwi_retrospective}"
work_dir="${WORK_DIR:-$HOME/fwi_work}"
python_bin="${PYTHON:-python}"
start_year="${START_YEAR:-2001}"
end_year="${END_YEAR:-2025}"

[[ "$start_year" =~ ^[0-9]{4}$ && "$end_year" =~ ^[0-9]{4}$ ]] || exit 2
(( start_year >= 2001 && end_year <= 2025 && start_year <= end_year )) || exit 2

mkdir -p "$output_dir" "$work_dir"
export PYTHONDONTWRITEBYTECODE=1

for ((year=start_year; year<=end_year; year++)); do
    archive="$archive_dir/SURFACEMODEL_${year}.tar.gz"
    [[ -f "$archive" ]] || { echo "Missing archive: $archive" >&2; exit 1; }
    
    # Own a fresh extraction directory; never delete a pre-existing user folder.
    batch="$(mktemp -d "$work_dir/fwi-${year}.XXXXXX")"
    echo "Extracting $archive into $batch"
    tar -xzf "$archive" -C "$batch"
    init_args=()
    if [[ "$year" == 2001 ]]; then init_args=(--initialize); fi
    "$python_bin" "$script_dir/python/get_fireW_index.py" \
        --input-dir "$batch/outputs/retrospective/SURFACEMODEL" \
        --output-dir "$output_dir/$year" \
        --restart-root "$output_dir" \
        "${init_args[@]}" "$@"
    
    # set -e preserves the extracted inputs if extraction or calculation fails.
    rm -rf -- "$batch"
done
