# The Canadian Fire Weather Index (FWI)

This section documents the proper use of codes

using Land Data Assimilation System (LDAS) retrospective and forecast datasets on the `eps-monsoon` server.

## Retrospective FWI (starting from 2001)

The retrospective workflow calculates daily FWI values from historical LDAS output for the period of 2001-2025.

How to use `eval_ldas_*_fwi.sh` for daily retrospective from `2001` to `2025`

Configure `eval_ldas_retro_fwi.sh` accordingly,

```sh

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

Check the following before initiating the runner script, do a quick check on the structure of the forecast directory, make sure it conforms with the requirement of the runner.

For single day forecast, daily lis-forecast runs are stored under: `/mnt/vast/prakrut/backup/lis_runs/malaria_amazon/forecast/daily/jul01/SURFACEMODEL`.

> [!NOTE]
> Since right now there are only daily forecasts initialized on July 1st of each year on the `/mnt/vast/...` location. In the future, data initialized on other days will be added, and the forecast directory should will be organized as `/mnt/vast/.../daily/{init_date}/...`

Under the `~/SURFACEMODEL` directory of July 1st initialized data takes the following structure:

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
So, during our FWI workflow, we will need to first read the FWI of July 1st derived from the LIS retrospective data of that year to spin up the algorithm.

The FWI calculated from retrospective LDAS data is stored under: `/mnt/vast/ksu/lis_products/fwi_amazon/retrospective/daily/original_dc/`

Run one initialization per invocation with the Python environment activated.
Set `END_DATE`to the last day file expected in that run. The wrapper validates both boundaries and the runner ejects gaps in between.

```shell
mkdir -p /mnt/vast/lis_products/fwi_amazon/forecast/logs
INIT_DATE=2001-07-01 \
END_DATE=2001-12-31 \
nohup bash scripts/shell/fwi/eval_ldas_fcst_fwi.sh \
    --dask-workers 4 \
    --worker-memory 32GiB \
    > /mnt/vast/ksu/lis_products/fwi_amazon/forecast/logs/fwi_fcst_init_2001-07-01_original_dc.log \
    2>1 < /dev/null &
echo "Job PID: $!"
```

Output go under `OUTPUT_ROOT/original_dc/init_2001-07-01/`. The default
`OUTPUT_ROOT` is `/mnt/vast/ksu/lis_products/forecast/fwi_amazon/daily`.

Override `FORECAST_ROOT` in the `.sh` script with the initialization-specific `SURFACEMODEL`
directory when using an archive other than the documented `jul01` archive.

To resume after a completed forecast day, set `START_DATE` to the next date,
keeping `INIT_DATE`, configuration, and `OUTPUT_ROOT` the same. The preceding
forecast output must be present. Existing daily output require

For soil-moisture DC, set `DC_METHOD=SoilMoist_inst`, `RETRO_ROOT` to matching
retrospective outputs, and `SOIL_SATURATION` to their MAXSMC saturation file.
The current saturation loader expects `MAXSMC` with `lat` and `lon` dimensions.
Older preprocessing examples below produce a different schema (`SM_sat`),
which requires separate reconciliation before use with the current loader.

Run each year's July 1st initialization sequentially, using their retrospective checkpoint:

```bash
# Run from the repository root.
(
    set -euo pipefail

    for year in {2001..2024}; do
        echo "Processing initialization ${year}-07-01"

        INIT_DATE="${year}-07-01" \
        END_DATE="${year}-12-31" \
        bash scripts/shell/fwi/eval_ldas_fcst_fwi.sh \
            --dask-workers 4 \
            --worker-memory 32GiB
    done
)
```

## Experimental FWI

Exploring methods and parameters used for the FWI system, ranging from fine tunning the system to better work with designated AOI or incorporating simulated surface data.

### Using a different method `drought_code` , i.e. `SoilMoist_inst` (Yang et al. 2015)

As `Yang et al. (2015)` propose a new DC methods that uses Soil Moisture content (m^3 / m^3)

$$DC = 400 \ln{\frac{\text{Saturated Soil Moisture}}{\text{Soil Moisture}}} $$

```bash
DC_METHOD=SoilMoist_inst \
SOIL_SATURATION=/mnt/vast/ksu/hihydrosoil_250_WCsat/LIS_formatted_WCsat/SM_Vsat_0_40.nc \
OUTPUT_DIR=/mnt/vast/ksu/fwi_retrospective_amazon/daily_soilMoist_dc \
START_YEAR=2001 \
END_YEAR=2025 \
nohup bash scripts/shell/fwi/eval_ldas_retro_fwi.sh \
    --dask-workers 4 \
    --worker-memory 32GiB \
    > /mnt/vast/ksu/fwi_retrospective_amazon/logs/fwi_2001_2025_soilMoist_dc.log \
    2>&1 < /dev/null &

echo "Job PID: $!"
```
