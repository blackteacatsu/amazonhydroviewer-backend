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
