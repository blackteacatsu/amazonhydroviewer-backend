#!/usr/bin/env bash
set -eo pipefail

# Usage: ./eval_fire_risk.sh MONTH [OUTPUT_DIR]
month="${1:?Usage: $0 MONTH [OUTPUT_DIR]}"
output_dir="${2:-./wildfire_backtest_output}"

source /home/kris/miniconda3/etc/profile.d/conda.sh
conda activate analytics

set -u

run_timestamp="$(date +%Y-%m-%d_%H-%M-%S)"
mkdir -p "$output_dir"
mkdir -p logs

# DEFAULT HINDCAST RANGE 2001-2020
# nohup python ./hydroviewer-backend/init_firerisk.py \
#     --surface-model-dir /mnt/c/Users/Kris/Documents/amazonforecast/backup/monthly \
#     --hcst-start-year 2001 \
#     --hcst-end-year 2020 \
#     --month "$month" \
#     --fire-risk-method both \
#     --output-dir "$output_dir" \
#     > "./logs/fire_risk_backtest_${run_timestamp}.log" 2>&1

# echo "Finished wildfire backtest for month ${month}"


for month in {1..12}; do
    echo "========================================"
    echo "Starting wildfire backtest for month ${month}"
    echo "========================================"

    python ./hydroviewer-backend/init_firerisk.py \
        --surface-model-dir /mnt/c/Users/Kris/Documents/amazonforecast/backup/monthly \
        --hcst-start-year 2002 \
        --hcst-end-year 2020 \
        --month "$month" \
        --fire-risk-method both \
        --output-dir "$output_dir" \
        > "./logs/fire_risk_backtest_month_${month}_${run_timestamp}.log" 2>&1

    echo "Finished wildfire backtest for month ${month}"
done

echo "Finished wildfire backtest for all 12 months."



# #!/usr/bin/env bash
# set -eo pipefail

# # Usage: ./eval_fire_risk.sh MONTH [OUTPUT_DIR]
# month="${1:?Usage: $0 MONTH [OUTPUT_DIR]}"
# output_dir="${2:-./wildfire_backtest_output}"

# set -u

# # source /home/local/WIN/qsu4/miniconda3/etc/profile.d/conda.sh
# source /home/kris/miniconda3/etc/profile.d/conda.sh
# conda activate analytics

# run_timestamp="$(date +%Y-%m-%d_%H-%M-%S)"
# mkdir -p logs

# nohup python ./init_firerisk.py \
#     --surface-model-dir /mnt/c/Users/Kris/Documents/amazonforecast/backup/monthly \
#     --hcst-start-year 2001 \
#     --hcst-end-year 2020 \
#     --month "$month" \
#     --fire-risk-method both \
#     --output-dir "$output_dir" \
#     > "./logs/fire_risk_backtest_${run_timestamp}.log" 2>&1

# echo "Finished wildfire backtest for month ${month}"
