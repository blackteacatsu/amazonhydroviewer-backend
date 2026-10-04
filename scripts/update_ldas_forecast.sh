#!/usr/bin/env bash
set -e 

# start working venv for the updater
source /home/local/WIN/qsu4/miniconda3/etc/profile.d/conda.sh
conda activate analytics

current_date_time="$(date +%Y-%m-%d)"

mkdir -p logs

nohup python ./python/updater.py \
    --surface-model-dir /mnt/vast/prakrut/backup/lis_runs/malaria_amazon/forecast/monthly \
    --hcst-start-year 2001 \
    --hcst-end-year 2020 \
    --fcst-init-date 2022 08 \
    > ./logs/forecast_update_ver_${current_date_time}.log 2>&1 

# push new data to github data repo
git push

echo "Finished Processing"