# `README` for `.sh` directory

This directory contains bash scripts & command lines to run `python` runner scripts under `scripts/python/.py`.

Directories are configure according to comply with the `eps-redwood` server. Therefore, use with extra caution when transplanting to other machine/environment.

> [!INFO] This snippet are used as cronjob on the research group server `eps-redwood`.

## LDAS S2S Forecast Updater

This repo contains code for updating the monthly LDAS forecast. To initiate

<!-- to be completed -->

Run the code below to launch monthly updater, (`updater.py`)

```shell
nohup bash scripts/shell/update_ldas_forecast.sh \
    > /dev/null \
    2>&1 < /dev/null &
```

## The Canadian Fire Weather Index

### How to use `eval_ldas_*_fwi.sh` for daily retrospective from `2001` to `2025`

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

In order to monitor the processing progress type the following:

```bash
tail -f /mnt/vast/ksu/amazon_wildfire/logs/fwi_2001_2025_original_dc.log
```

### Forecasting FWI w/ LDAS

Use `eval_ldas_fcst_fwi.sh`

````shell
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
````
