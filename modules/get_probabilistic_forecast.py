import xarray as xr
import numpy as np
from pathlib import Path
import re
from datetime import datetime
import os

def read_trim_forecast(file_path, va):
    """
    Read forecast data for a specific variable.
    
    Args:
        file_path (str): Path to forecast NetCDF file
        va (str): Variable name to extract

    Returns:
        xarray.DataArray: Forecast data for the variable
    """
    try:
        forecast_data = xr.open_dataset(file_path)[va]
        return forecast_data
    except KeyError:
        print(f'ERROR: Variable {va} not found in dataset {file_path}')
        raise

def read_trim_hindcast(file_path, va):
    """
    Read hindcast data for a specific variable from multiple files.

    Args:
        file_path (str or list): Path(s) to hindcast NetCDF file(s)
        va (str): Variable name to extract

    Returns:
        xarray.DataArray: Hindcast data for the variable, rechunked for quantile ops
    """
    try:
        hindcast_data = xr.open_mfdataset(file_path, join='outer')[va]
        # Rechunk time and ensemble to single chunks for quantile computation
        chunk_dict = {}
        if 'time' in hindcast_data.dims:
            chunk_dict['time'] = -1
        if 'ensemble' in hindcast_data.dims:
            chunk_dict['ensemble'] = -1
        if chunk_dict:
            hindcast_data = hindcast_data.chunk(chunk_dict)
        return hindcast_data
    except KeyError:
        print(f'ERROR: Variable {va} not found in hindcast dataset')
        raise

def get_thresh(icat, quantiles, xrds, dims=['ensemble', 'time']):
    """
    Calculate threshold boundaries for a category based on quantiles.

    Args:
        icat (int): Category index (0, 1, 2 for terciles)
        quantiles (list): Quantile boundaries (e.g., [1/3, 2/3] for terciles)
        xrds (xarray.DataArray): Data array to calculate quantiles from
        dims (list): Dimensions to calculate quantiles over

    Returns:
        tuple: (lower_threshold, upper_threshold) for the category
    """
    if not all(elem in xrds.dims for elem in dims):
        raise Exception(f'Some dimensions in {dims} not present in xr.DataArray {xrds.dims}')

    # Rechunk core dimensions to single chunks for quantile computation with dask
    rechunk_dict = {d: -1 for d in dims if d in xrds.dims}
    if rechunk_dict and hasattr(xrds, 'chunks') and xrds.chunks is not None:
        xrds = xrds.chunk(rechunk_dict)

    if icat == 0:  # Below normal category
        xrds_lo = -np.inf
        xrds_hi = xrds.quantile(quantiles[icat], dim=dims)
    elif icat == len(quantiles):  # Above normal category
        xrds_lo = xrds.quantile(quantiles[icat-1], dim=dims)
        xrds_hi = np.inf
    else:  # Normal category
        xrds_lo = xrds.quantile(quantiles[icat-1], dim=dims)
        xrds_hi = xrds.quantile(quantiles[icat], dim=dims)

    return xrds_lo, xrds_hi

def calculate_probabilities(hcst, fcst, quantiles=[1/3., 2/3.]):
    """
    Calculate tercile category probability exceedance for ensemble forecast.
    
    Uses hindcast to define climatological tercile boundaries (below-normal, 
    normal, above-normal), then calculates probability that forecast ensemble
    members fall into each category.
    
    Args:
        hcst (xarray.DataArray): Hindcast data with dims [time, ensemble, lat, lon]
        fcst (xarray.DataArray): Forecast data with dims [time, ensemble, lat, lon]
        quantiles (list): Category boundaries (default: terciles at [1/3, 2/3])
    
    Returns:
        xarray.DataArray: Probability (0-1) that forecast falls in each category
                         Dims: [category, time, lat, lon]
                         - Category 0 = below normal (< 33rd percentile)
                         - Category 1 = normal (33rd-67th percentile)
                         - Category 2 = above normal (> 67th percentile)
    """
    print('Computing probabilities...')
    numcategories = len(quantiles) + 1  # 3 categories for terciles

    # Mask out 0 values in forecast (assumes 0 = missing/invalid)
    # NOTE: Verify this is appropriate for your data
    fcst_masked = fcst.where(fcst != 0)

    # Rechunk once for the quantile operation and compute all quantile edges once.
    q_dims = [d for d in ['ensemble', 'time'] if d in hcst.dims]
    if not q_dims:
        raise Exception(f"Expected at least one of ['ensemble', 'time'] in hcst dims, got {hcst.dims}")
    if hasattr(hcst, 'chunks') and hcst.chunks is not None:
        hcst = hcst.chunk({d: -1 for d in q_dims})
    q_edges = hcst.quantile(quantiles, dim=q_dims)

    l_probs = []
    for icat in range(numcategories):
        print(f'  category={icat}')
        if icat == 0:
            h_lo = -np.inf
            h_hi = q_edges.sel(quantile=quantiles[0])
        elif icat == len(quantiles):
            h_lo = q_edges.sel(quantile=quantiles[-1])
            h_hi = np.inf
        else:
            h_lo = q_edges.sel(quantile=quantiles[icat - 1])
            h_hi = q_edges.sel(quantile=quantiles[icat])

        # Drop scalar quantile coord to avoid carrying it into outputs.
        if hasattr(h_lo, "coords") and 'quantile' in h_lo.coords:
            h_lo = h_lo.drop_vars('quantile')
        if hasattr(h_hi, "coords") and 'quantile' in h_hi.coords:
            h_hi = h_hi.drop_vars('quantile')

        # Count fraction of ensemble members in this category.
        prob = np.logical_and(fcst_masked > h_lo, fcst_masked <= h_hi).sum('ensemble') / float(fcst_masked.sizes['ensemble'])
        l_probs.append(prob.assign_coords({'category': icat}))
    
    probs = xr.concat(l_probs, dim='category')
    return probs

def _parse_date_from_name(name: str) -> datetime | None:
    """
    Parse initialization date from forecast filename.
    
    Supports formats:
    - ldas_fcst_2024_dec01.nc
    - ldas_fcst_20241201.nc
    """
    # Month name to number mapping
    _MONTHS = {m: i+1 for i, m in enumerate(
        ["jan", "feb", "mar", 
        "apr", "may", "jun", 
        "jul", "aug", "sep", 
        "oct", "nov", "dec"]
    )}

    # Filename patterns to recognize
    _PATTERNS = [
        re.compile(r'^ldas_fcst_(\d{4})_([a-z]{3})(\d{2})\.nc$', re.I),  # ldas_fcst_2024_dec01.nc
        re.compile(r'^ldas_fcst_(\d{4})(\d{2})(\d{2})\.nc$', re.I),      # ldas_fcst_20241201.nc
    ]

    for pat in _PATTERNS:
        m = pat.match(name)
        if not m:
            continue
        if pat is _PATTERNS[0]:
            y = int(m.group(1))
            mon = _MONTHS.get(m.group(2).lower())
            d = int(m.group(3))
        else:
            y, mon, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if mon and 1 <= mon <= 12 and 1 <= d <= 31:
            return datetime(y, mon, d)
    return None

def forecast_init_datetime(fpath: str) -> datetime:
    """Extract initialization datetime from forecast file path."""
    dt = _parse_date_from_name(Path(fpath).name)
    if dt is None:
        raise ValueError(f"Unrecognized forecast filename format: {fpath}")
    return dt

def split_forecast_and_hindcasts(
    dir_path: str,
    prefix: str = "ldas_fcst_",
    recursive: bool = False
):
    """
    Find latest forecast file and all hindcast files of the same month.
    
    Args:
        dir_path (str): Directory containing forecast files
        prefix (str): Filename prefix to match
        recursive (bool): Search subdirectories
        
    Returns:
        tuple: (forecast_path, hindcast_paths_list, forecast_datetime)
    """
    base = Path(dir_path)
    if not base.is_dir():
        raise NotADirectoryError(f"Not a directory: {dir_path}")

    pattern = "**/*.nc" if recursive else "*.nc"
    items = []
    for p in base.glob(pattern):
        if not p.is_file():
            continue
        name = p.name
        if not name.startswith(prefix) or not name.endswith(".nc"):
            continue
        dt = _parse_date_from_name(name)
        if dt is None:
            continue
        items.append((dt, p.stat().st_mtime, name, p))

    if not items:
        raise FileNotFoundError(f"No matching .nc files found in {dir_path} (prefix='{prefix}')")

    # Latest by (date, mtime, name)
    items.sort(key=lambda t: (t[0], t[1], t[2]))
    forecast_path = items[-1][3]
    forecast_dt = items[-1][0]

    # Hindcasts = existing Dec-01 files from earlier years only
    # hindcasts = [
    #     p for (dt, _, _, p) in items
    #     if dt.year < forecast_dt.year and dt.month == 12 and dt.day == 1
    # ]
    hindcasts = [
        p for (dt, _, _, p) in items
        if dt.year < 2021 and dt.month == forecast_dt.month and dt.day == 1
    ]
    # Sort hindcasts by year ascending (oldest → newest)
    hindcasts.sort(key=lambda p: _parse_date_from_name(p.name))

    return str(forecast_path), [str(p) for p in hindcasts], forecast_dt

def purge_old_init(directory: Path, current_init: str):
    import shutil
    for f in list(directory.glob('*')):
        if f.name.endswith(".json"):
            continue
            
        if current_init not in f.name:
            shutil.rmtree(f)
            print(f"Deleted (old init): {f}")