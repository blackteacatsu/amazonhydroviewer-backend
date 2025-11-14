import xarray as xr
import numpy as np


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
        xarray.DataArray: Hindcast data for the variable, chunked by time
    """
    try:
        hindcast_data = xr.open_mfdataset(file_path, join='outer')[va]
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

    l_probs = []
    for icat in range(numcategories):
        print(f'  category={icat}')
        h_lo, h_hi = get_thresh(icat, quantiles, hcst)
        
        # Count fraction of ensemble members in this category
        prob = np.logical_and(fcst_masked > h_lo, fcst_masked <= h_hi).sum('ensemble') / float(fcst_masked.sizes['ensemble'])
        
        # Remove quantile coordinate if present (artifact from threshold calculation)
        if 'quantile' in prob.coords:
            prob = prob.drop_vars('quantile')
        
        l_probs.append(prob.assign_coords({'category': icat}))
    
    probs = xr.concat(l_probs, dim='category')
    return probs

def split_forecast_and_dec_hindcasts(
    dir_path: str,
    prefix: str = "ldas_fcst_",
    recursive: bool = False
):
    """
    Find latest forecast file and all December 1st hindcast files.
    
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
    hindcasts = [
        p for (dt, _, _, p) in items
        if dt.year < forecast_dt.year and dt.month == 12 and dt.day == 1
    ]
    # Sort hindcasts by year ascending (oldest → newest)
    hindcasts.sort(key=lambda p: _parse_date_from_name(p.name))

    return str(forecast_path), [str(p) for p in hindcasts], forecast_dt