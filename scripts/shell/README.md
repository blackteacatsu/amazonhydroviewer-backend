# `README` for `.sh` directory

## LDAS S2S Forecast Updater

This repo contains code for updating the monthly LDAS forecast. To initiate 

> [!INFO] This snippet are used as cronjob on the research group server `eps-redwood`. 

<!-- to be completed -->
```bash
nohup bash update_ldas_forecast.sh \ 
    > /mnt/vast/ksu
    2>&1 < /dev/null &
```


## FWI from retrospective for the time period `2001` to `2025`



### How to use `eval_ldas_retro_fwi.sh`

This script finds the FWI index from the retrospective LDAS data on `eps-monsoon`. Run the following code to 
```bash
nohup bash scripts/shell/eval_ldas_retro_fwi.sh \
    --dask-workers 4 \
    --worker-memory 32GiB \
    > /mnt/vast/ksu/fwi_retrospective_amazon/logs/fwi_2001_2025.log \
    2>&1 < /dev/null &

echo "Job PID: $!"
```

Run this from the repository root with your Python environment activated.

```bash
tail -f /mnt/vast/ksu/amazon_wildfire/logs/fwi_2001_2025.log
```
