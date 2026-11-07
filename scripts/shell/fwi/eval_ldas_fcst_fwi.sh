#!/usr/bin/env bash
set -eou pipefail

# Load Conda for noninteractive jobs and use the dependencies
source /home/local/WIN/qsu4/miniconda3/etc/profile.d/conda.sh
conda activate analytics

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
runner="$script_dir/../../python/get_fireW_index.py"

# Use the active environment's Python, 
python_bin="${PYTHON:-python}"

: "${INIT_DATE:?Set INIT_DATE to YYYY-MM-DD}"
: "${END_DATE:?Set END_DATE to the last forecast valid date}"

# Define source directories
fcst_dir="${FCST_DIR:-/mnt/vast/prakrut/backup/lis_runs/malaria_amazon/forecast/daily/jul01/SURFACEMODEL}"
retro_dir="${RETRO_DIR:-/mnt/vast/ksu/lis_products/fwi_amazon/retrospective/daily/original_dc}"
dc_method="${DC_METHOD:-original}"
case "$dc_method" in
    original) method_dir=original_dc ;;
    SoilMoist_inst)
        method_dir=soilMoist_dc
        : "${RETRO_DIR:?Set RETRO_DIR to retrospective output using the same soil DC method}"
        : "${SOIL_SATURATION:?Set SOIL_SATURATION to the MAXSMC NetCDF used by retrospective}"
        ;;
    *) echo "Unsupported DC_METHOD : $dc_method" >&2; exit 2;;
esac
output_root="${OUTPUT_DIR:-/mnt/vast/ksu/lis_products/fwi_amazon/forecast/daily}"
output_dir="$output_root/$method_dir/init_$INIT_DATE"

for location in "$fcst_dir" "$retro_dir" "$output_dir"; do
    [[ "$location" == /* ]] || { echo "Use an absolute path: $location" >&2; exit 2; }
done

# Enumerate intersecting months using calender dates, including cross-year runs.
# START_DATE is only needed to resume after a completed forecast checkpoint.
selection="$("$python_bin" - "$INIT_DATE" "${START_DATE:-}" "$END_DATE" "$fcst_dir" <<'PY'
from datetime import date, timedelta
from pathlib import Path
import sys
init = date.fromisoformat(sys.argv[1])
start = date.fromisoformat(sys.argv[2]) if sys.argv[2] else init + timedelta(days=1)
end = date.fromisoformat(sys.argv[3])
if not init < start <= end:
    raise SystemExit('Require INIT_DATE < START_DATE <= END_DATE')
print(start.isoformat())
month = start.replace(day=1)
while month <= end:
    folder = Path(sys.argv[4]) / month.strftime('%Y%m')
    if not folder.is_dir():
        raise SystemExit(f'Missing forecast month: {folder}')
    print(folder)
    month = (month.replace(day=28) + timedelta(days=4)).replace(day=1)

PY
)"

inputs=()
# Bash 3-compatible array construction also preserves spaces in directory names.
while IFS= read -r line; do inputs+=("$line"); done <<< "$selection"
start_date="${inputs[0]}"
inputs=("${inputs[@]:1}")

# Resolve only the first checkpoint from retrospective. Later batches search
# within this initialization's output directory, preventing cross-run resumes.
checkpoint_args=(--restart-root "$output_dir")
first_date="$("$python_bin" -c 'from datetime import date,timedelta; import sys; print(date.fromisoformat(sys.argv[1])+timedelta(days=1))' "$INIT_DATE")"
if [[ "$start_date" == "$first_date" ]]; then
    checkpoint="$("$python_bin" - "$retro_dir" "$INIT_DATE" <<'PY'
from pathlib import Path
import sys
retro_dir = Path(sys.argv[1])
date = sys.argv[2]
names = [f'LIS_FWI_{date.replace("-", "_")}.nc', f'fwi_{date.replace("-", "")}.nc']
matches = sorted({p.resolve() for name in names for p in retro_dir.rglob(name) if p.is_file()})
if len(matches) != 1:
    raise SystemExit(f'Expected one retrospective checkpoint for {date}, found {len(matches)}')

print(matches[0])  
PY
)"
    checkpoint_args=(--initial-state "$checkpoint")
fi
extra_args=(--dc-method "$dc_method")
if [[ "$dc_method" == SoilMoist_inst ]]; then
    extra_args+=(--soil-saturation "$SOIL_SATURATION")
fi
# Positional flags are compute settings, RH policy or an explicit overwrite.
# Keep run identity and data selection controlled by the environment above.
for option in "$@"; do
    case "$option" in 
        --input*|--output-dir*|--restart-root*|--mode*|--init*|--start-date*|--end-date*|--dc-method*|--soil-saturation*)
        echo "Configure runs selection through environment variables :$option" >&2; exit 2 ;;
    esac
done

# Keep Python bytecode and Dask spill files out of the code checkout.
export PYTHONDONTWRITEBYTECODE=1

export DASK_TEMPORARY_DIRECTORY="${WORK_DIR:-$output_dir/.work/dask}"
mkdir -p "$DASK_TEMPORARY_DIRECTORY"
exec "$python_bin" "$runner" \
    --mode forecast \
    --init-date "$INIT_DATE" \
    --start-date "$start_date" \
    --end-date "$END_DATE" \
    --input-dir "${inputs[@]}" \
    --output-dir "$output_dir" \
    "${checkpoint_args[@]}" \
    "${extra_args[@]}" "$@"