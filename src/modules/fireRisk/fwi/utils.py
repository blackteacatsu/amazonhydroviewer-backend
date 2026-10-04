
from __future__ import annotations

import sys
import xarray as xr
import numpy as np

from pathlib import Path


# Import the new src tree, never a leftover 
# repository-root modules directory.
REPO_ROOT = Path(__file__).resolve().parents[4]
SRC = REPO_ROOT / 'src'
if not SRC.is_dir():
    raise RuntimeError(f'Missing source tree: {SRC}')
sys.path.insert(0, str(SRC))


from modules.fireRisk import angstrom


EFFECTIVE_DAY_L_DMC = {
     'JAN' : 6.5, 'FEB' : 7.5, 'MAR' : 9.0,
     'APR' : 12.8, 'MAY' : 13.9, 'JUN' : 13.9,
     'JUL' : 12.4, 'AUG' : 10.9, 'SEP' : 9.4,
     'OCT' : 8.0, 'NOV' : 7.0, 'DEC' : 6.0
}


EFFECTIVE_DAY_L_DC = {
     'JAN' : -1.6, 'FEB' : -1.6, 'MAR' : -1.6,
     'APR' : 0.9, 'MAY' : 3.8, 'JUN' : 5.8,
     'JUL' : 6.4, 'AUG' : 5.0, 'SEP' : 2.4,
     'OCT' : 0.4, 'NOV' : -1.6, 'DEC' : -1.6
}


# def is_date(ds_file : Path) -> bool:
#     time_dim = ['time', 'valid time', 'initialization date']

#     try:
#          ds = xr.open_dataset(ds_file)
#          if names in time_dim in ds.dims:

#     except Exception as exc:
#          raise SystemError from exc


def get_day_length_factors(
        da_time : xr.DataArray
) -> xr.Dataset:
    """Select the original Canadian Le/Lf tables using each record's valid month.

    These are source-report factors, not a validated Amazon regional adaptation.
    Le (DMC) is effective day length; Lf (DC) is a seasonal adjustment.
    """
    months = [
        "JAN", "FEB", "MAR", "APR", "MAY", "JUN",
        "JUL", "AUG", "SEP", "OCT", "NOV", "DEC",
    ]

    tables = xr.Dataset(
        {
            "Le": (
                "month",
                [EFFECTIVE_DAY_L_DMC[m] for m in months]
            ),
            "Lf": (
                "month",
                [EFFECTIVE_DAY_L_DC[m] for m in months]
            )
        },
        # Explicit 1..12 labels: positional 0..11 indexing shifts every month.
        coords={"month": list(range(1, 13))},
    )

    return tables.sel(month=da_time.dt.month)


def stdize_units(da:xr.DataArray):
    value = da.attrs.get('units')
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{da.name}: missing units; refusing to guess')
    return value.lower().replace(' ', '').replace('**', '^')


def select_dates(source:xr.Dataset, variables:tuple[str]) -> tuple[xr.Dataset, np.ndarray]:
    """
    Require a continuous daily sequence, 
    including across file/month boundaries.
    """
    if 'time' not in source.coords or source.time.dims != ('time',):
        raise ValueError('Require a one-dimensional decoded time coordinate')
    missing = set(variables) - set(source.data_vars)
    if missing:
        raise ValueError(f'Missing weather fields: {sorted(missing)}')
    if not np.issubdtype(source.time.dtype, np.datetime64):
        raise ValueError('This runner requires Gregorian datetime64 dates')
    if not source.indexes['time'].is_monotonic_increasing or not source.indexes['time'].is_unique:
        raise ValueError('Time must be sorted and unique')
    times = source.time.values
    if not times.size or np.isnat(times).any():
        raise ValueError('Require nonempty time coordinates without NaT')
    dates = times.astype('datetime64[D]')
    if len(times) > 1 and (not np.all(np.diff(times) == np.timedelta64(1, 'D'))
                          or not np.all(np.diff(dates) == np.timedelta64(1, 'D'))):
        raise ValueError('Require one record per day, exactly 24 hours apart; gaps and overlaps are not allowed')
    selected = source
    dims = selected.Tair_f_tavg.dims
    if 'time' not in dims or len(dims) < 2:
        raise ValueError(f'Expected time and at least one spatial dimension; found {dims}')
    for n in variables:
        if set(selected[n].dims) != set(dims):
            raise ValueError(f'{n}: forcing dimensions must match temperature')
    return selected, dates


def prepare_weather(
        day : xr.Dataset,
        variables : tuple[str]
) -> tuple[
    xr.DataArray,
    xr.DataArray,
    xr.DataArray,
    xr.DataArray, 
    xr.DataArray,
]:
    """
    Normalize the verified daily forcing. No implicit unit defaults.
    """
    units = {n: stdize_units(day[n]) for n in variables}

    # Existing helpers convert temperature, q and p; reject missing units above.
    T = angstrom._temperature_celsius(day.Tair_f_tavg.astype('float64'))
    RH = angstrom.get_QRair(day.Qair_f_tavg.astype('float64'),
                           day.Psurf_f_tavg.astype('float64'), T)
    wind_scales = {'ms-1': 3.6, 'ms^-1': 3.6, 'm/s': 3.6,
                   'kmh-1': 1., 'kmh^-1': 1., 'km/h': 1.}
    rain_scales = {'kgm-2s-1': 86400., 'kgm^-2s^-1': 86400.,
                   'mm/s': 86400., 'mmday-1': 1., 'mm/day': 1., 'mm': 1.}
    if units['Wind_f_tavg'] not in wind_scales or units['Rainf_tavg'] not in rain_scales:
        raise ValueError(f'Unsupported wind/rain units: {units}')
    W = (day.Wind_f_tavg.astype('float64') * wind_scales[units['Wind_f_tavg']]).assign_attrs(units='km h-1')
    R = (day.Rainf_tavg.astype('float64') * rain_scales[units['Rainf_tavg']]).assign_attrs(units='mm')
    finite = np.isfinite(day[list(variables)].to_array()).all('variable')
    valid = (finite & np.isfinite(RH) & (RH >= 0) & (RH <= 100)
             & (W >= 0) & (R >= 0) & (day.Qair_f_tavg >= 0) & (day.Psurf_f_tavg > 0))
    # Specific humidity is a fraction, irrespective of its source units.
    q = day.Qair_f_tavg / (1000 if units['Qair_f_tavg'] in {'g/kg', 'gkg-1', 'gkg^-1'} else 1)
    valid = valid & (q < 1)
    return T, RH, W, R, valid


def write_netcdf(dataset:xr.Dataset, path:Path) -> None:
    """Write atomically; retain float64 moisture state for the next day/month."""
    temporary = path.with_suffix(path.suffix + '.partial')
    encoding = {n: {'zlib': True, 'complevel': 2} for n in dataset.data_vars}
    dataset.drop_encoding().to_netcdf(temporary, engine='h5netcdf', encoding=encoding)
    temporary.replace(path)


def attach_daily_time(ds, variables:tuple[str]):
    """LIS daily files store 2D weather separately from their singleton time."""
    ds = ds.copy(deep=False)
    for name in variables:
        if name not in ds:
            raise ValueError(f'Missing weather field: {name}')
        if 'time' not in ds[name].dims:
            if ds.sizes.get('time') != 1:
                raise ValueError('2D weather requires exactly one file timestamp')
            ds[name] = ds[name].expand_dims(time=ds.time)
    return ds


def inventory(inputs:list[Path], variables:tuple) -> list:
    """
    Read timestamps/units only; 
    close each file before opening the next.
    """
    paths = []
    for value in inputs:
        path = value.expanduser().resolve()
        paths.extend(sorted(path.rglob('LIS_HIST*.nc')) if path.is_dir() else [path])
    if not paths:
        raise ValueError('No input NetCDF files found')
    if len(paths) != len(set(paths)):
        raise ValueError('Duplicate input file paths')
    records, units = [], None
    for path in paths:
        with xr.open_dataset(path) as raw:
            ds, _ = select_dates(attach_daily_time(raw, variables), variables)
            current = {name: stdize_units(ds[name]) for name in variables}
            if units is not None and current != units:
                raise ValueError(f'Forcing units differ between files: {path}')
            units = current
            records.extend((time, path, i) for i, time in enumerate(ds.time.values))
    records.sort(key=lambda row: row[0])
    times = np.array([row[0] for row in records])
    if len(times) > 1 and not np.all(np.diff(times) == np.timedelta64(1, 'D')):
        raise ValueError('Input dates have a gap, duplicate, or non-daily interval')
    return records


def read_day(record, variables:tuple[str]) -> xr.Dataset:
    _, path, index = record
    with xr.open_dataset(path) as raw:
        ds = attach_daily_time(raw, variables)
        names = list(variables) + [n for n in ('lat', 'lon') if n in ds]
        return ds[names].isel(time=index, drop=True).load()


def daily_output_name(date):
    """Use the same daily naming convention for writing and restart discovery."""
    return f"LIS_FWI_{str(np.datetime64(date, 'D')).replace('-', '_')}.nc"


def find_previous_output(explicit_path, root, previous_date):
    """Find exactly one preceding daily file; never guess between duplicate runs."""
    if explicit_path is not None:
        return explicit_path.expanduser().resolve()
    # Recognize both this runner's filenames and the earlier January pilot outputs.
    date = np.datetime64(f'{previous_date[:4]}-{previous_date[4:6]}-{previous_date[6:8]}')
    names = (daily_output_name(date), f'fwi_{previous_date}.nc')
    matches = sorted({path.resolve() for name in names for path in root.rglob(name)
                      if path.is_file()})
    if not matches:
        raise FileNotFoundError(
            f'No preceding daily output ({names}) under {root}. Set --restart-root or '
            '--initial-state; use --initialize only for a new series.')
    if len(matches) != 1:
        raise ValueError(f'Multiple previous outputs found: {matches}. Choose one with --initial-state.')
    return matches[0]


def load_daily_moisture_codes(
        path, 
        first_time, 
        template, 
        locations
) -> list[xr.DataArray]:
    """
    Read FFMC/DMC/DC from one daily output, 
    validating date, grid, and ranges.
    """
    with xr.open_dataset(path) as saved:
        # A daily result must contain exactly one decoded timestamp.
        if 'time' not in saved.coords or saved.time.dims != ('time',) or saved.sizes['time'] != 1:
            raise ValueError('Previous daily output must contain exactly one time record')
        if not np.issubdtype(saved.time.dtype, np.datetime64):
            raise ValueError('Previous output requires a decoded Gregorian timestamp')
        previous_time = saved.time.values[0]
        if np.isnat(previous_time) or previous_time + np.timedelta64(1, 'D') != first_time:
            raise ValueError('Previous output must be exactly one day before the first input')
        for name in ('ffmc', 'dmc', 'dc'):
            if name not in saved or 'time' not in saved[name].dims:
                raise ValueError(f'Previous output missing time-dependent {name}')
        names = ['ffmc', 'dmc', 'dc'] + [n for n in ('lat', 'lon') if n in saved]
        # Drop the single time dimension and load only restart data into memory.
        ds = saved[names].isel(time=0, drop=True).load()
    for name in ('lat', 'lon'):
        if (name in ds) != (name in locations):
            raise ValueError(f'Previous-output grid is missing {name}')
        if name in locations:
            xr.testing.assert_equal(ds[name].reset_coords(drop=True), locations[name].reset_coords(drop=True))
    moisture_codes = []
    for name in ('ffmc', 'dmc', 'dc'):
        field = ds[name].reset_coords(drop=True)
        if dict(field.sizes) != dict(template.sizes):
            raise ValueError(f'Previous-output grid dimensions differ: {name}')
        for dim in template.dims:
            if (dim in field.coords) != (dim in template.coords):
                raise ValueError(f'Previous-output coordinate differs: {dim}')
        xr.align(field, template, join='exact')
        if bool((np.isinf(field) | (field < 0) | ((field > 101) if name == 'ffmc' else False)).any()):
            raise ValueError(f'Invalid saved moisture code: {name}')
        # Preserve NaNs and full precision; never fill missing history with defaults.
        moisture_codes.append(field.astype('float64'))
    return moisture_codes


def load_state(path, first_time, template, locations):
    """Reject stale dates, changed grids, and invalid carried moisture codes."""
    with xr.open_dataset(path) as ds:
        ds = ds.load()
    if 'state_time' not in ds or ds.state_time.ndim != 0:
        raise ValueError('Checkpoint must contain scalar state_time')
    if ds.state_time.values + np.timedelta64(1, 'D') != first_time:
        raise ValueError('Checkpoint timestamp must be exactly one day before the first input')
    for name in ('lat', 'lon'):
        if (name in ds) != (name in locations):
            raise ValueError(f'Checkpoint grid is missing {name}')
        if name in locations:
            xr.testing.assert_equal(ds[name].reset_coords(drop=True), locations[name].reset_coords(drop=True))
    result = []
    for name in ('ffmc', 'dmc', 'dc'):
        field = ds[name].reset_coords(drop=True)
        if dict(field.sizes) != dict(template.sizes):
            raise ValueError(f'Checkpoint grid dimensions differ: {name}')
        for dim in template.dims:
            if (dim in field.coords) != (dim in template.coords):
                raise ValueError(f'Checkpoint coordinate differs: {dim}')
        xr.align(field, template, join='exact')
        if bool((np.isinf(field) | (field < 0) | ((field > 101) if name == 'ffmc' else False)).any()):
            raise ValueError(f'Invalid checkpoint values: {name}')
        result.append(field.astype('float64'))
    return result