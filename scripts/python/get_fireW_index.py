#!/usr/bin/env python3
"""Calculate daily FWI from the dates in one or more daily LDAS files (1985 equations).

Daily-mean pilot: T/RH/wind are not verified noon observations. Rain rates are
assumed to represent the 86400 seconds associated with each source date label.
Uses the original Canadian month factors, not an Amazon-calibrated adaptation.
"""
from __future__ import annotations

import sys
import json
import argparse
import hashlib
import numpy as np
import xarray as xr
from pathlib import Path
from datetime import datetime, timezone
# from collections.abc import Any


# Import the new src tree, never a leftover 
# repository-root modules directory.
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / 'src'
if not SRC.is_dir():
    raise RuntimeError(f'Missing source tree: {SRC}')
sys.path.insert(0, str(SRC))

from modules.fireRisk import angstrom
from modules.fireRisk.fwi import core, utils, soilMoist

# Define variables short name used for calculating FwI
VARIABLES = (
    'Tair_f_tavg',
    'Qair_f_tavg',
    'Psurf_f_tavg',
    'Wind_f_tavg',
    'Rainf_tavg',
)
CODES = (
    'ffmc',
    'dmc',
    'dc',
    'isi',
    'bui',
    'fwi',
    'dsr',
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir', '--input', dest='input_dir', type=Path, nargs='+', required=True, help='Daily LDAS NetCDF files or folders (if folders then search for LIS_HIST*.nc).')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--restart-root', type=Path, help='Output tree for preceding daily FWI file; default to `output_dir.`')
    p.add_argument('--dask-workers', type=int, default=0, help='Optional local worker processes (0 runs NumPy directly).')
    p.add_argument('--worker-memory', default='4GiB', help='Memory limit per Dask worker; stay within job allocation.')
    p.add_argument('--chunk-size', type=int, default=270, help='Spatial chunk edge length for Dask.')

    # Define constants used for initializing FWI runs before the first day of available data.
    start = p.add_mutually_exclusive_group()
    start.add_argument('--initial-state', type=Path, help='Explicit checkpoint from the preceding day(override auto search).')
    start.add_argument('--initialize', action='store_true', help='Start a new series using initial constants.')
    p.add_argument('--initial-ffmc', type=float, default=85.0)
    p.add_argument('--initial-dmc', type=float, default=6.0)
    p.add_argument('--initial-dc', type=float, default=15.0)

    p.add_argument('--rh-policy', choices=('cap', 'mask'), default='cap',
                   help='Cap finite RH above 100% (default), or reproduce the strict masking policy.')
    p.add_argument('--dc-method', choices=('SoilMoist_inst', 'original'), default='original',
                   help='Specify the drought code method used, the moisture equivalent originally used in 1987 (default)')
    p.add_argument('--soil-saturation', type=Path, help='Preprocessed SM_sat NetCDF on the LDAS grid, depth 0–40 cm.')
    p.add_argument('--overwrite', action='store_true', help='Replace this run’s output files.')
    args = p.parse_args(argv)
    if (args.dc_method == 'SoilMoist_inst') != (args.soil_saturation is not None):
        p.error('--dc-method SoilMoist_inst requires --soil-saturation; original does not use it')
    if args.dask_workers < 0 or args.chunk_size < 1:
        p.error('Require nonnegative --dask-workers and positive --chunk-size')
    initials = (args.initial_ffmc, args.initial_dmc, args.initial_dc)
    if not all(np.isfinite(v) for v in initials) or not 0 <= initials[0] <= 101 or min(initials[1:]) < 0:
        p.error('Require finite FFMC in [0,101] and finite nonnegative DMC/DC')
    return args


# Run a chronological batch: prepare inputs, carry moisture codes, and save outputs/checkpoints.
def run(args:argparse.Namespace) -> dict:
    # Resolve the daily-output directory to an absolute path; expand ~ to the home dir.
    output_dir = args.output_dir.expanduser().resolve() 
    # Search a shared output tree when months/years use separate output folders.
    restart_root = (args.restart_root or output_dir).expanduser().resolve()
    
    # Inspect inputs and obtain sorted (timestamp, file path, record index) tuples; 
    # (reject time gaps/duplicates)
    # Collect the original timestamps; these retain their time of day.
    soil_method = args.dc_method == 'SoilMoist_inst'
    input_variables = VARIABLES + (('SoilMoist_inst',) if soil_method else ())
    inventory_rows = utils.inventory(args.input_dir, input_variables)
    times = np.array([row[0] for row in inventory_rows])
    dates = times.astype('datetime64[D]')
    inputs = list(dict.fromkeys(row[1] for row in inventory_rows))

    # Load the first day to establish the spatial grid & source metadata.
    # Build YYYYMMDD for the day immediately before this batch, used to find its checkpoints.
    first_day = utils.read_day(inventory_rows[0], input_variables)
    saturation = soilMoist.load_saturation(args.soil_saturation, first_day) if soil_method else None
    saturation_hash = hashlib.sha256(saturation.values.tobytes()).hexdigest() if soil_method else ''
    previous_date = str(dates[0] - np.timedelta64(1, 'D')).replace('-', '')
    # With --initialize use predefined constant; 
    # otherwise choose an explicit checkpoint or the expected preceding-day file
    initial_path = None if args.initialize else utils.find_previous_output(args.initial_state, restart_root, previous_date)
    if initial_path is not None:
        initial_path = initial_path.expanduser().resolve()
        if not initial_path.is_file():
            raise FileNotFoundError(f'Missing checkpoint file at path {initial_path}. ')
    if initial_path is not None:
        with xr.open_dataset(initial_path) as previous:
            previous_policy = previous.attrs.get('rh_policy', 'mask')
            if previous.attrs.get('dc_method', 'original') != args.dc_method:
                raise ValueError('Restart drought-code method differs from this run')
            if soil_method and previous.attrs.get('soil_saturation_sha256') != saturation_hash:
                raise ValueError('Restart saturation field differs from this run')
        if previous_policy != args.rh_policy:
            raise ValueError('Restart RH policy differs from this run. Recompute from initialization into a new output directory.')
    selected = xr.Dataset(coords={'time': times})
    # Prepare one daily output filename for every input date.
    # Name the JSON summary with the first and last day processed.
    paths = [output_dir / utils.daily_output_name(date) for date in dates]
    summary_path = output_dir / f"summary_{str(dates[0]).replace('-', '_')}_{str(dates[-1]).replace('-', '_')}.json"
    for path in paths + [summary_path]:
        if path.exists() and not args.overwrite:
            raise FileExistsError(f'{path} exists; select another directory or use --overwrite')            
    # Validate static metadata before creating any files.
    for n in VARIABLES:
        utils.stdize_units(first_day[n])
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        'source_files': json.dumps([str(path) for path in inputs]),
        'algorithm': 'Van Wagner and Pickett 1985 with Yang 2015 soil-moisture DC' if soil_method else 'Van Wagner and Pickett 1985 written equations',
        'sampling': 'daily_mean_approximation',
        'rh_policy': args.rh_policy,
        'rh_policy_description': 'Finite nonnegative RH capped at saturation after source checks; raw RH retained' if args.rh_policy == 'cap' else 'Reject RH outside 0–100%',
        'rainfall_interval_assumption': '24-hour mean rate associated with each date label; no verified noon-to-noon bounds',
        'day_length_scheme': 'original_Canadian_month_tables_not_regionally_calibrated',
        'initialization': 'previous_daily_output' if initial_path else 'assumed_constants_before_first_day',
        'initial_state_file': str(initial_path) if initial_path else '',
        'initial_ffmc': args.initial_ffmc, 'initial_dmc': args.initial_dmc, 'initial_dc': args.initial_dc, 
        'initial_state_date': str(dates[0] - np.timedelta64(1, 'D')),
        'dc_method': args.dc_method,
        'dc_equation': '400 ln(SM_sat / SM), Yang et al. 2015, adapted to 0–40 cm' if soil_method else 'original',
        'soil_saturation_file': str(args.soil_saturation.resolve()) if soil_method else '',
        'soil_saturation_sha256': saturation_hash,
        'soil_helper_sha256': hashlib.sha256(Path(soilMoist.__file__).read_bytes()).hexdigest(),
        'core_sha256': hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest(),
        'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'humidity_helper_sha256': hashlib.sha256(Path(angstrom.__file__).read_bytes()).hexdigest(),
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'gap_policy': 'invalid forcing invalidates all three carried codes; no automatic reinitialization',
    }
    # For a resumed batch, avoid reporting unused initialization constants.
    # Select the three constant-initialization metadate entries & remove them.
    if initial_path:
        for name in ('initial_ffmc', 'initial_dmc', 'initial_dc'):
            metadata.pop(name)
    if soil_method:
        metadata.pop('initial_dc', None)
        metadata['soil_depth_cm'] = '0-40'
        metadata['soil_invalid_policy'] = 'Mask missing, nonpositive, or above-saturation moisture; FFMC/DMC history remains invalid after gaps'
    # Inherit projection metadata from source file
    # Visit the source projection, grid origin, and spacing attributes.
    for key in ['MAP_PROJECTION', 'SOUTH_WEST_CORNER_LAT', 'SOUTH_WEST_CORNER_LON', 'DX', 'DY']:
        if key in first_day.attrs:
            value = first_day.attrs[key]
            metadata[key] = value.item() if isinstance(value, np.generic) else value
    # Choose day length factors (Le & Lf) from the month of each original timestamp.
    day_l_factors = utils.get_day_length_factors(selected.time)
    # Keep latitude and longitude arrays, when available, for restart-grid validation.
    # Load and validate 
    first_locations = {n : first_day[n] for n in ('lon', 'lat') if n in first_day}
    moisture_code = (utils.load_daily_moisture_codes(initial_path, times[0], first_day.Tair_f_tavg, first_locations)
                     if initial_path else None)
    # Accumulate daily counts and summary statistics for the final JSON file.
    # Process everyday in chronological order, pairing its date with its output filename.
    records = []
    static_locations = None
    for index, (date, path) in enumerate(zip(dates, paths)):
        # Keep memory bounded: load five forcings and locations for ONE day.
        # Reuse the already-loaded first day; subsequent calls open, load and close one day at a time.        
        daily = first_day if index == 0 else utils.read_day(inventory_rows[index], input_variables)
        # Keep file I/O in the parent process; distribute only the spatial calculations.
        if args.dask_workers:
            dims = daily.Tair_f_tavg.dims
            daily = daily.chunk({dim: args.chunk_size for dim in dims})
        T, H, W, R, valid, diagnostics = utils.prepare_weather(
            daily, VARIABLES, rh_policy=args.rh_policy, return_diagnostics=True)
        locations = {n: daily[n] for n in ['lat', 'lon'] if n in daily}
        # Require the spatial dimensions and sizes to match the first day.
        if dict(T.sizes) != dict(first_day.Tair_f_tavg.sizes):
            raise ValueError('Spatial dimensions does not match !')

        xr.align(T, first_day.Tair_f_tavg, join='exact')
        if set(locations) != set(first_locations):
            raise ValueError('Latitude/Longitude availability changed between days')
        # Establish the geographic reference on the first day.
        if static_locations is None:
            static_locations = locations
        else:
            for n in locations:
                xr.testing.assert_equal(locations[n], static_locations[n])
        if moisture_code is None:
            moisture_code = [xr.full_like(T, v, dtype=np.float64).where(valid)
                        for v in [args.initial_ffmc, args.initial_dmc, args.initial_dc]]
        # FFMC/DMC always need history. Soil-derived DC itself is memoryless.
        active = valid.copy()
        for previous in moisture_code[:2] if soil_method else moisture_code:
            active = active & np.isfinite(previous)
        if soil_method:
            thickness = daily.attrs.get('SOIL_LAYER_THICKNESSES')
            if thickness is None or not np.allclose(thickness, [10, 30, 60, 100]):
                raise ValueError('Soil method requires LDAS layer thicknesses 10,30,60,100 cm')
            sm = soilMoist.avg_soilMoist_wDepth(daily.SoilMoist_inst)
            soil_dc = soilMoist.get_drought_code(sm, saturation)
            # Retain the existing conservative gap policy for all carried codes.
            active = active & np.isfinite(soil_dc)
        le = xr.DataArray(float(day_l_factors.Le.isel(time=index)))
        lf = xr.DataArray(float(day_l_factors.Lf.isel(time=index)))
        ffmc = core.get_ffmc(T, H, W, R, moisture_code[0]).where(active)
        dmc = core.get_dmc(T, H, R, moisture_code[1], le).where(active)

        if soil_method:
            dc = soil_dc.where(active)
        elif args.dc_method == 'original':
            dc = core.get_drought_code(T, R, moisture_code[2], lf).where(active)

        isi = core.get_isi(W, core.ffmc_to_moisture(ffmc)).where(active)
        bui = core.get_bui(dmc, dc).where(active)
        fwi, dsr = core.get_fwi(isi, bui)

        fields = [ffmc, dmc, dc, isi, bui, fwi, dsr]
        # Evaluate this day's graph once, before validation, statistics, or writing.
        # Carry computed arrays forward so the graph cannot grow across days.
        computed = xr.Dataset(dict(zip(CODES, fields)))
        if soil_method:
            computed['soil_moisture_0_40'] = sm
            computed['soil_moisture_valid'] = np.isfinite(soil_dc)
            computed['soil_above_saturation'] = np.isfinite(sm) & (sm > saturation)
        computed['forcing_valid'] = valid
        computed['state_valid'] = active
        computed['relative_humidity_raw'] = diagnostics['relative_humidity_raw']
        computed['rh_capped'] = diagnostics['rh_capped']
        computed = computed.compute()
        fields = [computed[name] for name in CODES]
        ffmc, dmc, dc, isi, bui, fwi, dsr = fields
        valid, active = computed.forcing_valid, computed.state_valid
        for n, field in zip(CODES, fields):
            if bool((active & ~np.isfinite(field)).any()):
                raise FloatingPointError(f'{date}: nonfinite {n} at an otherwise valid cell')
        
        # Carry today's full-precision moisture codes into the next iteration; do not reinitialize them.
        moisture_code = [ffmc, dmc, dc]  # Full precision; never reset on each date.
        result = xr.Dataset(dict(zip(CODES, fields))).assign_coords(locations)
        if soil_method:
            for name in ('soil_moisture_0_40', 'soil_moisture_valid', 'soil_above_saturation'):
                result[name] = computed[name] if name == 'soil_moisture_0_40' else computed[name].astype('int8')
        result['forcing_valid'] = valid.astype('int8')
        result['state_valid'] = active.astype('int8')
        result['relative_humidity_raw'] = computed.relative_humidity_raw
        result['rh_capped'] = computed.rh_capped.astype('int8')
        result = result.expand_dims(time=[selected.time.values[index]])
        result.attrs = {**metadata, 'valid_date': str(date), 'Le': float(le), 'Lf': float(lf)}
        utils.write_netcdf(result, path)
        
        # Begin the daily summary with its date and count of valid-weather cells.
        record = {
            'date': str(date), 'forcing_valid_cells': int(valid.sum()),
            'rh_capped_cells': int(computed.rh_capped.sum()),
            'state_valid_cells' : int(active.sum()), 'file' : path.name
        }
        if soil_method:
            record['soil_valid_cells'] = int(computed.soil_moisture_valid.sum())
            record['soil_above_saturation_cells'] = int(computed.soil_above_saturation.sum())
        # Visit each named output array.
        for n, field in zip(CODES, fields):
            values = field.values[np.isfinite(field.values)]
            record[n] = ({
                'min': float(values.min()),
                'mean': float(values.mean()),
                'max': float(values.max()),
            } if values.size else None)
        records.append(record)
        print(f'{date}: {record["state_valid_cells"]:,} valid cells -> {path.name}', flush=True)
    summary = {**metadata, 'days_completed': len(records), 'restart_file': str(paths[-1]), 'daily': records}
    temporary = summary_path.with_suffix('.json.partial')
    temporary.write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    temporary.replace(summary_path)
    print(f'Complete: {len(records)} daily files, restart state and {summary_path.name}', flush=True)
    return summary


if __name__ == '__main__':
    args = parse_args()
    if args.dask_workers:
        from dask.distributed import Client, LocalCluster

        # Local workers use this node only; 
        # request CPUs/RAM through the job scheduler first.
        with LocalCluster(n_workers=args.dask_workers, threads_per_worker=1,
                          processes=True, memory_limit=args.worker_memory,
                          dashboard_address=None) as cluster:
            with Client(cluster):
                run(args)
    else:
        run(args)
