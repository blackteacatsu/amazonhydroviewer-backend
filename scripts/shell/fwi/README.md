# The Canadian Fire Weather Index (FWI)

This section documents the proper use of codes

using Land Data Assimilation System (LDAS) retrospective and forecast datasets on the `eps-monsoon` server.

## Retrospective FWI (starting from 2001)

The retrospective workflow calculates daily FWI values from historical LDAS output for the period of 2001-2025.

In order to use `eval_ldas_retro_fwi.sh` for daily retrospective from `2001` to `2025`, configure/check `eval_ldas_retro_fwi.sh` accordingly,

```sh
...
# Read-only annual input archives. ARCHIVE_DIR can override this default.
archive_dir="${ARCHIVE_DIR:-/mnt/vast/prakrut/backup/lis_runs/malaria_amazon/retrospective}"
output_root="${OUTPUT_DIR:-/mnt/vast/ksu/lis_products/fwi_amazon/retrospective/daily}"
# The wrapper appends original_dc or soilMoist_dc, then the year.
output_dir="$output_root/$method_dir"

# Temporary annual extraction folders; use node-local scratch.
work_dir="${WORK_DIR:-$output_dir/.work}"
...

# RN a complete run of retrospective FWI should covers 2001–2025;
# You could use START_YEAR to resume at a completed year boundary.
start_year="${START_YEAR:-2001}"
end_year="${END_YEAR:-2025}"
...

```

This script finds the FWI index from the retrospective LDAS data on `eps-monsoon`. Run the following code from the repository root:

```shell
nohup bash scripts/shell/fwi/eval_ldas_retro_fwi.sh \
    --dask-workers 4 \
    --worker-memory 32GiB \
    > /mnt/vast/ksu/lis_products/fwi_amazon/retrospective/logs/fwi_2001_2025_original_dc.log \
    2>&1 < /dev/null &

echo "Job PID: $!"
```

To monitor the processing progress type the following:

```bash
tail -f /mnt/vast/ksu/lis_products/fwi_amazon/retrospective/logs/fwi_2001_2025_original_dc.log
```

## Forecast FWI using LDAS

To derive FWI from LIS forecast runs, use script `eval_ldas_fcst_fwi.sh`.

Check the structure of the forecast directory before initiating the runner script, make sure it conforms with the requirement of the runner. Daily lis-forecast runs are stored under: `/mnt/vast/prakrut/backup/lis_runs/malaria_amazon/forecast/daily/jul01/SURFACEMODEL`.

> [!NOTE]
> Since right now there are only daily forecasts initialized on July 1st between 2001 to 2024 year on the `/mnt/vast/...` location. In the future, data initialized on other days will be added, and the forecast directory should will be organized as `/mnt/vast/.../daily/{init_date}/...`

The July 1st initialized data takes the following structure under the `~/jul01/SURFACEMODEL/` directory should looks like,

```bash
forecast/daily/jul01/SURFACEMODEL/
├── 200107/
│   ├── LIS_HIST_200107020000.d01.nc
│   ├── LIS_HIST_200107030000.d01.nc
│   ├── ...
│   └── LIS_HIST_200107310000.d01.nc
├── 200108/
│   ├── LIS_HIST_200108010000.d01.nc
│   └── ...
├── ...
├── 202407/
│   ├── LIS_HIST_202407020000.d01.nc
│   └── ...
└── 202412/
    ├── LIS_HIST_202412010000.d01.nc
    ├── ...
    └── LIS_HIST_202412310000.d01.nc
```

LIS uses the July 1st data to forecast starting from the day after it, therefore every year's hindcast starts from July 2nd.
So, in our FWI workflow, we will need to first read the FWI of July 1st derived from the LIS retrospective data of that specific year to spin up the algorithm.

The FWI calculated from retrospective LDAS data is stored under: `/mnt/vast/ksu/lis_products/fwi_amazon/retrospective/daily/original_dc/`

### Run one year's initialization first to test the method.

Run one initialization per invocation. Set `END_DATE`to the last day file expected in that run. The wrapper validates both boundaries and the runner ejects gaps in between.

```shell
mkdir -p /mnt/vast/ksu/lis_products/fwi_amazon/forecast/logs

INIT_DATE=2001-07-01 \ # <-- chane this to other init date
END_DATE=2001-12-31 \ # <-- every init forecast 6-month into the future.

nohup bash scripts/shell/fwi/eval_ldas_fcst_fwi.sh \
    --dask-workers 4 \
    --worker-memory 32GiB \
    > /mnt/vast/ksu/lis_products/fwi_amazon/forecast/logs/fwi_fcst_init_2001-07-01_original_dc.log \
    2>&1 < /dev/null &

echo "Job PID: $!"
```

The default
`OUTPUT_DIR` is `/mnt/vast/ksu/lis_products/fwi_amazon/forecast/daily`.

Output go under `OUTPUT_DIR/original_dc/init_{INIT_DATE}/`.

Override `FCST_DIR` in `eval_ldas_fcst_fwi.sh` with the initialization-specific `SURFACEMODEL`
directory when using an archive other than the documented `jul01` archive.

To resume after a completed forecast day, add `START_DATE` to the above snippet and set it to the next date,
keeping `INIT_DATE`, configuration, and `OUTPUT_DIR` the same. The preceding
forecast output must be present. Existing daily output require an explicit `--overwrite` to be replaced.

For soil-moisture DC, set `DC_METHOD=SoilMoist_inst`, `RETRO_DIR` to matching
retrospective outputs, and `SOIL_SATURATION` to their MAXSMC saturation file.
The current saturation loader expects `MAXSMC` with `lat` and `lon` dimensions.

<!-- Older preprocessing examples below produce a different schema (`SM_sat`),
which requires separate reconciliation before use with the current loader. -->

> [!CAUTION]
> Use of `SoilMoist_inst` in Drought Code are included for experimental purposes only, see section below for more information. Use with caution.

### Processing the entire batch

Run each year's July 1st initialization sequentially, using their retrospective checkpoint:

```bash
# Run from the repository directory.
log_dir=/mnt/vast/ksu/lis_products/fwi_amazon/forecast/logs
# mkdir -p "$log_dir" # <= include this if running for the first time

# Timestamp distinguishes this batch from earlier attempts.
batch_id="$(date +%Y%m%d_%H%M%S)"

nohup bash -c '
    set -euo pipefail
    log_dir="$1"
    batch_id="$2"

    for year in {2001..2024}; do
        log_file="$log_dir/fwi_init_${year}-07-01_${batch_id}.log"
        echo "$(date -Is) Starting ${year}-07-01 — $log_file"

        if INIT_DATE="${year}-07-01" \
           END_DATE="${year}-12-31" \
           PYTHONUNBUFFERED=1 \
           bash scripts/shell/fwi/eval_ldas_fcst_fwi.sh \
               --dask-workers 4 \
               --worker-memory 32GiB \
               > "$log_file" 2>&1
        then
            echo "$(date -Is) Completed ${year}-07-01"
        else
            echo "$(date -Is) FAILED ${year}-07-01 — see $log_file"
            exit 1
        fi
    done
' bash "$log_dir" "$batch_id" \
    > "$log_dir/batch_${batch_id}.log" 2>&1 < /dev/null &

echo "Batch PID: $!"
echo "Batch log: $log_dir/batch_${batch_id}.log"s
```

Monitor overall progress:

```bash
tail -f "$log_dir/batch_${batch_dir}.log"
```

Monitor a particular year's daily processing:

```bash
tail -f "$log_dir/fwi_init_2001-07-01_${batch_id}.log"
```

## Experimental FWI using simulated soil moisture

Exploring methods and parameters used for the FWI system, ranging from fine tunning the system to better work with designated AOI or incorporating simulated surface data.

### Using a different method for Drought Code, i.e. `SoilMoist_inst` (Yang et al. 2015)

Yang et al. (2015) proposed a new DC methods that uses Soil Moisture content (in m^3 / m^3)

$$DC = 400 \ln{\frac{\text{Saturated Soil Moisture}}{\text{Soil Moisture}}} $$

```bash
# Create the log directory before shell redirection opens the log file.
mkdir -p /mnt/vast/ksu/lis_products/fwi_amazon/retrospective/logs

DC_METHOD=SoilMoist_inst \
SOIL_SATURATION=/mnt/vast/ksu/lis_products/inputs/AmazonLDAS_MAXSMC.nc \
START_YEAR=2001 \
END_YEAR=2025 \

nohup bash scripts/shell/fwi/eval_ldas_retro_fwi.sh \
    --dask-workers 4 \
    --worker-memory 32GiB \
    > /mnt/vast/ksu/lis_products/fwi_amazon/retrospective/logs/fwi_2001_2025_soilMoist_dc.log \
    2>&1 < /dev/null &

echo "Job PID: $!"
```

The retrospective wrapper defaults to `DC_METHOD=original`. Soil mode requires
`DC_METHOD=SoilMoist_inst` and an absolute, readable `SOIL_SATURATION` path.
`OUTPUT_DIR` is a base directory: results go to `OUTPUT_DIR/original_dc/YEAR`
or `OUTPUT_DIR/soilMoist_dc/YEAR`; do not include the method directory in the override.
Restarts search only the selected method tree. Resume with `START_YEAR` after a
completed year, keeping the same saturation file and method. Pass compute flags
(e.g. `--dask-workers`, `--worker-memory`), RH policy, and `--overwrite` as arguments;
configure input/output paths, method, and year selection through environment variables.
