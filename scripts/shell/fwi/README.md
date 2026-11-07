# The Canadian Fire Weather Index

## How to use `eval_ldas_*_fwi.sh` for daily retrospective from `2001` to `2025`

Configure `eval_ldas_retro_fwi.sh` accordingly,

```sh

```

This script finds the FWI index from the retrospective LDAS data on `eps-monsoon`. Run the following code from the repository root:

```shell
nohup bash scripts/shell/eval_ldas_retro_fwi.sh \
    --dask-workers 4 \
    --worker-memory 32GiB \
    > /mnt/vast/ksu/lis/retrospective/fwi_amazon/logs/fwi_2001_2025_original_dc.log \
    2>&1 < /dev/null &

echo "Job PID: $!"
```

To monitor the processing progress type the following:

```bash
tail -f /mnt/vast/ksu/amazon_wildfire/logs/fwi_2001_2025_original_dc.log
```

## Forecasting FWI w/ LDAS

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

The FWI calculated from retrospective LDAS data is stored under: `/mnt/vast/ksu/lis_products/retrospective/fwi_amazon/daily/original_dc/`

Once all checked, copy the following code snippet into terminal and run,

```shell
nohup bash scripts/shell/fwi/eval_ldas_fcst_fwi.sh \
    --dask-workers 4 \
    --worker-memory 32 GiB \
    > /mnt/vast/ksu/lis/retrospective/fwi_amazon/logs/fwi_2001_2025
echo
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
nohup bash scripts/shell/eval_ldas_retro_fwi.sh \
    --dask-workers 4 \
    --worker-memory 32GiB \
    > /mnt/vast/ksu/fwi_retrospective_amazon/logs/fwi_2001_2025_soilMoist_dc.log \
    2>&1 < /dev/null &

echo "Job PID: $!"
```
