#!/usr/bin/env python
# coding: utf-8

"""Monthly Update Script
Description: this is the master workflow script, 
run the following cells to initialize data for visualization.
"""

from __future__ import annotations
from pathlib import Path
from git import Repo
from datetime import datetime
from dask.distributed import Client, LocalCluster

import argparse
import os

import modules.utils as utils
import modules.data_subsampler as subsampler 
import modules.get_zonal_stats as zonal
import modules.get_prob_fcst as prob


clustr = LocalCluster(
    n_workers=4,
    threads_per_worker=4
)


# Current working directory
CWD = Path(os.getcwd())

# Variable definitions with both long & short names
LIST_OF_VARIABLES = {
    'Rainf_tavg': 'Average precipitation', 
    'Qair_f_tavg': 'Specific humidity',
    'Qs_tavg': 'Surface runoff',
    'Evap_tavg': 'Evapotranspiration',
    'Tair_f_tavg': 'Avg. air temperature',
    'SoilMoist_inst': 'Soil moisture',
    'SoilTemp_inst': 'Soil temperature',
    'Streamflow_tavg': 'Stream flow'
}

# Data bounds of AOI used for Sub-sampling
DATA_BOUNDS = {'lon_min': -81.975, 
               'lon_max': -49.025, 
               'lat_min': -20.975, 
               'lat_max': 5.975}
# DATA_BOUNDS = [
#     -81.975,
#     -49.025,
#     -20.975,
#     5.975
# ]

# Data directory
SURFACE_MODEL_DIR = r"/mnt/vast/prakrut/backup/lis_runs/malaria_amazon/forecast/monthly"

# 
RIVER_NETWORK_FILE = CWD / "static" / "annual_mean_50cumecs_river_network.nc"

# AOI shapes - labelled with PFAF_IDs 
AOI_MASK_POLYGON = CWD / "static" / "hybas_sa_lev05_aoi.geojson"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cwd", type=Path, default=CWD)
    parser.add_argument("--surface-model-dir", type=Path, default=Path(SURFACE_MODEL_DIR), required=True)
    parser.add_argument("--hcst-start-year", type=int, default=2001)
    parser.add_argument("--hcst-end-year", type=int, default=2020)
    parser.add_argument(
        "--fcst-init-date",
        nargs=2,
        type=int,
        metavar=("YEAR", "MONTH"),
        help="Forecast initialization month; defaults to the latest available.",
    )
    parser.add_argument("--variables", nargs="+", choices=LIST_OF_VARIABLES, default=list(LIST_OF_VARIABLES))
    parser.add_argument(
        "--data-bounds", 
        type=float, 
        nargs=4, 
        default=[
            DATA_BOUNDS["lon_min"],
            DATA_BOUNDS["lon_max"],
            DATA_BOUNDS["lat_min"],
            DATA_BOUNDS["lat_max"]
        ], 
        metavar=("lon-min", "lon-max", "lat-min", "lat-max")
    )
    parser.add_argument("--river-mask-file", type=Path, default=RIVER_NETWORK_FILE)
    parser.add_argument("--aoi-polygon-file", type=Path, default=AOI_MASK_POLYGON)
    
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.hcst_start_year > args.hcst_end_year:
        raise SystemExit("Hindcast start year must not exceed end year.")

    requested_init = None
    if args.fcst_init_date:
        try:
            requested_init = datetime(*args.fcst_init_date, 1)
        except ValueError as exc:
            raise SystemExit(f"Invalid forecast initialization month: {exc}") from exc
        if args.hcst_end_year > requested_init.year:
            raise SystemExit(
                "Hindcast end year must not exceed forecast initialization year."
            )

    selected_variables = {
        name : LIST_OF_VARIABLES[name]
        for name in args.variables
    }

    if isinstance(args.data_bounds, list):
        args.data_bounds = dict(
            zip(
                ("lon_min", "lon_max", "lat_min", "lat_max"),
                args.data_bounds
            )
        )

    # Validate repository ownership before initialization performs output cleanup.
    repo = Repo(
        Path(args.cwd).expanduser().resolve(),
        search_parent_directories=True,
    )

    if repo.index.diff("HEAD"):
        raise RuntimeError(
            "Git index already contains staged changes; refusing to include them "
            "in the automated forecast commit."
        )
    
    conditions = utils.initialize_working_cond(
        cwd=args.cwd,
        surface_model_dir = args.surface_model_dir,
        hcst_start_year = args.hcst_start_year,
        hcst_end_year = args.hcst_end_year,
        fcst_init_date = requested_init,
    )

    # Step 1 Generate Probabilistic Forecast Data Using Hindcast
    prob.mainloop(
        variables= selected_variables,
        fcst_file=conditions.fcst_file,
        hdct_files=conditions.hcst_files,
        prob_fcst_cache_dir=conditions.prob_fcst_cache_dir,
        init_date=conditions.init_date,
        river_mask_file=args.river_mask_file
    )


    # Step 2 Apply Sub-Sampler For Web Use
    subsampler.subsample_updates(
        cache_dir=conditions.prob_fcst_cache_dir,
        target_dir=conditions.prob_fcst_subsampled_dir,
        data_bounds=args.data_bounds,
        init_date=conditions.init_date
    )


    # Step 3 Build forecast & climatology tables for each PFAF region
    zonal.get_zonal_tables(
        variables = selected_variables,
        fcst_file = conditions.fcst_file,
        hdct_files = conditions.hcst_files,
        aoi_polygon_file = args.aoi_polygon_file,
        zonal_avg_fcst_tab_dir = conditions.zonal_avg_fcst_tab_dir,
        zonal_avg_climatology_tab_dir = conditions.zonal_avg_climatology_tab_dir,
        climatology_cache_dir = conditions.climatology_cache_dir,
        init_date=conditions.init_date
    )


    # Repository ownership stays in the updater because staging and committing are
    # workflow side effects, not filesystem utilities.
    repository_root = Path(repo.working_tree_dir).resolve()
    output_paths = (
        conditions.prob_fcst_subsampled_dir,
        conditions.zonal_avg_fcst_tab_dir,
        conditions.zonal_avg_climatology_tab_dir
    )

    relative_output_paths = [
        str(path.relative_to(repository_root)) for path in output_paths
    ]
    repo.git.add("-A", "--", *relative_output_paths)
    if not repo.index.diff("HEAD"):
        print("No forecast output changes to commit.")
        return
    repo.index.commit(
        f"updated forecast anomaly data - {conditions.init_date}"
    )


if __name__ == "__main__":
    main()
