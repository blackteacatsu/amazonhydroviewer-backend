#!/usr/bin/env python3
"""Prepare a reusable 0–40 cm HiHydroSoil saturation map on a daily LIS grid."""
import argparse
import json
import sys
from pathlib import Path
import xarray as xr
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from modules.fireRisk.fwi import soilMoist, utils


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ldas-template', type=Path, required=True)
    parser.add_argument('--wcsat', type=Path, nargs=4, required=True,
                        help='GeoTIFFs in order: 0–5, 5–15, 15–30, 30–60 cm')
    parser.add_argument('--scale', type=float, default=0.0001,
                        help='Raw integer WCsat: 0.0001; already scaled rasters: 1')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    
    if args.output.exists():
        raise FileExistsError(args.output)
    
    with xr.open_dataset(args.ldas_template) as ds:
        day = ds.isel(time=0, drop=True) if 'time' in ds.dims else ds
        result = soilMoist.preprocess_saturation(args.wcsat, day, args.scale)
        
        # Validate the same contract used by the runner before writing.
        if not bool(result.SM_sat.notnull().any()):
            raise ValueError('No valid saturation cells overlap the LDAS grid')
        
        result.attrs['source_files'] = json.dumps([str(p.resolve()) for p in args.wcsat])
        args.output.parent.mkdir(parents=True, exist_ok=True)
        
        utils.write_netcdf(result, args.output)
    
    print(args.output)


if __name__ == '__main__':
    main()
