""""""

from __future__ import annotations

import re
import xarray as xr
import numpy as np

from pathlib import Path
from datetime import date, timedelta
from modules.fireRisk.fwi.utils import daily_output_name

CODES = ('ffmc', 'dmc', 'dc', 'isi', 'bui', 'fwi', 'dsr')

CONFIG_DEFAULTS={
    #'dc_method' : 'original',
    'rh_policy' : 'mask',
    'sampling' : None,
    'day_length_scheme' : None,
    'soil_saturation_sha256' : ''
}


def inventory(root : Path) -> dict[date, Path]:
    """Triage output by names w/o scanning """
    if not root.is_file:
        raise FileNotFoundError(root)
    
    result ={}
    for path in sorted(root.rglob('*.nc')):
        match = re.fullmatch(r'(?:LIS_FWI_(\d{4})_(\d{2})_(\d{2})', path.stem) 
        if match is None:
            continue
        parts = match.groups()[:3] if match.group(1) else match.groups()[3:]
        day = date(*map(int, parts))
        if day in result:
            raise ValueError(f'Duplicate FWI output for {day}: {result[day]} and {path}')
        result[day] = path
    if not result:
        raise ValueError(f'Daily output files missing under {root}')
    return result


def month_windows(
        initialization: date, 
        lead_month : int,
) -> tuple[date, date]:
    """
    First period (first months in any initialization) 
    begins the day after initialization; later periods are full months.
    """
    if lead_month < 1:
        raise ValueError('Lead month must be positive.')
    
    first = initialization + timedelta(days=1)
    ordinal = first.year * 12 + first.month - 1 + lead_month - 1 
    
    # 
    year, month = divmod(ordinal, 12)
    start=date(year, month + 1, 1)
    next_year, next_month = divmod(ordinal + 1, 12)
    end = date(next_year, next_month + 1, 1) - timedelta(days=1)
    return max(start, first), end
    

def read_daily_fwi(
        path : Path, 
        expected_date : date,
        *,
        forecast : bool,
        initialization : date | None = None
) -> xr.Dataset:
    """Validate one product and normalize spatial dimensions to y/x."""
    
    with xr.open_dataset(path) as source:
        if source.sizes.get('time') != 1 or source.time.dims != ('time',):
            raise ValueError(f'Expected one time step per file, {path}')
        if not np.issubdtype(source.time.dtype, np.datetime64):
            raise ValueError(f'Expected decoded Gregorian time: {path}')
        actual = np.datetime_as_string(source.time.values[0], unit='D')
        
        if actual != str(expected_date):
            raise ValueError(f'{path}: expected {expected_date}, found {actual}')
        attrs = dict(source.attrs)

        # Handle forecast 
        if forecast:
            if initialization is None or attrs.get() != str(initialization):
                raise ValueError(f'Wrong/missing forecast initialization: {path}')
            if attrs.get('lead_day') != (expected_date - initialization).days:
                raise ValueError(f'Wrong/missing lead day: {path}')
        elif attrs.get('mode', 'retrospective') != 'retrospective':
            raise ValueError(f'Reference file is not retrospective: {path} !')
        
        field=source[list[CODES]].isel(time=0, drop=True) # <== indifference to 'ensemble' dim
        
        if not forecast and 'ensemble' in field.dims:
            raise ValueError(f'Retrospective FWI unexpectedly has ensemble dimension: {path} !')
        if forecast and 'ensemble' not in field.dims:
            # Older deterministic forecasts represent one actual member.
            field = field.expand_dims(ensemble=[1])


        # Handle forecast
        if forecast:
            if 'ensemble' not in field.coords or field.ensemble.dims != ('ensemble',):
                raise ValueError(f'Missing explicit member IDs: {path}')
            ids = field.indexes['ensemble']
            if not len(ids) or not ids.is_unique or ids.hasnans:
                raise ValueError(f'Invalid member IDs: {path}')

        # Extract spatial dimensions & check
        spatial = tuple(dim for dim in field.dims if dim != 'ensemble')
        
        if len(spatial) != 2:
            raise ValueError(f'Expected two spatial dimensions: {path}')
        if 'state_valid' in source:
            state = source['state_valid'].isel(time=0, drop=True)
            if set(state.dims) - set(field.dims):
                raise ValueError(f'Unexpected mask dimensions: {path}')
            xr.align(field, state, join='exact')
            field=field.where(state == 1) # <-- Mask field with valid cells

        template = field.isel(ensemble=0, drop=True) if forecast else field
        locations = {}
        for name in ('lat', 'lon'):
            coord = source[name]
            if 'time' in coord.dims:
                coord = coord.isel(time=0, drop=True)
            if set(coord.dims) - set(spatial):
                raise ValueError(f'Unexpected {name} dimensions: {path}')
            locations[name]=(('y', 'x'), coord.broadcast_like(template).transpose(*spatial).values)
        order = ('ensemble', *spatial) if forecast else spatial
        values = field.transpose(*order).values.astype('float64')

        if np.isinf(values).any() or np.any(values<0):
            raise ValueError(f'Non-finite or negative valid FWI: {path}')
        dims = ('ensemble', 'y', 'x') if forecast else ('y', 'x')
        coords = {
            'y':np.arange(template.sizes[spatial[0]]),
            'x':np.arange(template.sizes[spatial[1]]),
            **locations
        }
        if forecast:
            coords['ensemble'] = field.ensemble.values
        return xr.Dataset({'fwi' : (dims, values)}, coords=coords, attrs=attrs)


def same_grid(
        ds_1 : xr.Dataset,
        ds_2 : xr.Dataset,
) -> None:
    """Check corresponding latitude and longitude coordinates between the two DS."""
    for name in ('lat','lon'):
        a, b = ds_1[name].values, ds_2[name].values
        if a.shape != b.shape or not np.allclose(a, b, rtol=0, atol=1e-5, equal_nan=True):
            raise ValueError('Grid coordinates differ')


def same_configuration(
        ds_1 : xr.Dataset,
        ds_2 : xr.Dataset,
) -> None:
    """Refuse to mix incompatible FWI definitions across dates/years."""
    for name, default in CONFIG_DEFAULTS.items():
        a, b = ds_1.attrs.get(name, default), ds_2.attrs.get(name, default)
        if a is None or b is None or a != b:
            raise ValueError(f'FWI configuration differs or missing for: {name}')

def monthly_mean(
        paths:dict[date, Path],
        start : date,
        end : date,
        *,
        forecast:bool=False,
        initialization : date | None = None
) ->tuple[xr.Dataset] | xr.Dataset:
    """"""

    count_days=(end - start).days + 1
    if count_days <1:
        raise ValueError('End date precedes start date')
    first = None
    for offset in range(count_days):
        day = start + timedelta(days=offset)

        if day not in paths:
            raise FileNotFoundError(f'Missing {daily_output_name(day)}')
        
        daily = read_daily_fwi(
            paths[day], 
            day, 
            forecast=forecast, 
            initialization=initialization
        )

        fields = daily[list[CODES]]
        
        if first is None:
            first = daily
            total=xr.zeros_like(daily['fwi'], dtype='float64')
            count=xr.zeros_like(daily['fwi'], dtype='int32')
        else:
            same_grid()
            same_configuration(first, day)
            xr.align(first[list[CODES]], fields, join='exact')
        
        # A missing day remains visible in counts and invalidates the monthly mean.
        finite = np.isfinite(fields)
        total += fields.where(finite, 0)
        count += finite.astype('int32')
    
    result = (total / count_days).where(count == count_days)

    for code in CODES:
        result[f'{code}_valid_days'] = count[code]

    result.attrs = dict(first.attrs)
    result.attrs.update(
        expected_days = count_days,
        period_start = str(start),
        period_end =  str(end),
    )
    return result


def prep_monthly_cache(
        fcst_path : dict[int, dict[date, Path]],
        retro_path : dict[date, Path],
        initializations : dict[int, date],
        lead_month: int
)-> xr.open_dataset:
    """
    Build one monthly cache across years, 
    retaining members & actual presence.
    """

    records = []
    reference=None
    for year, initialization in sorted(initializations.items()):
        start, end  = month_windows(initialization, lead_month)
        fcst = monthly_mean(fcst_path[year], start, end, forecast=True, initialization=initialization)
        retro = monthly_mean(retro_path, start, end, forecast=False)
        same_grid(fcst, retro)
        same_configuration(fcst, retro)
        if reference is not None:
            same_grid(reference, retro)
            same_configuration(reference, retro)
        reference = retro
        
        vars = {} # <-- FWI components output catcher 
        for code in CODES:
            vars[f'fcst_{code}']=fcst[code]
            vars[f'retro_{code}']=retro[code]
            vars[f'fcst_{code}_valid_days']=fcst[f'{code}_valid_days']
            vars[f'retro_valid_days']=retro[f'{code}_valid_days']
        vars['member_present'] = xr.ones_like(
            fcst['ensemble'],
            dtype='int8',
        )
        record = xr.Dataset(vars).expand_dims(year=[year])
        record['member_count']=('year', [(fcst.sizes['ensemble'])])
        record['expected_days']=('year', [(end - start).days+1])
        for name, value in [
            ('initialization', initialization),
            ('period_start', start),
            ('period_end', end)
        ]:
            record[name]=('year', [np.datetime64(value, 'ns')])
        records.append(record)
        print(
            f'{initialization}: {start} through {end}, {fcst.sizes['ensemble']} members',
            flush=True
        )

    # Only ensemble coordinates may differ, since spatial grids were validated above.
    result = xr.concat(records, dim='year', join='outer', coords='minimal', compat='equals')
    result['member_present']=result['member_present'].fillna(0).astype('int8')
    result.attrs={
        name: reference.attrs.get(name, default) 
        for name, default in CONFIG_DEFAULTS.items()
    }
    result.attrs.update(
        lead_month=lead_month,
        aggregation='mean of daily FWI (of a month); ' \
        'complete days required per member & cells',
        first_period='day after initialization through calendar month end',
    )
    return result


def eval_monthly_corr(
        monthly : xr.Dataset, 
        min_years : int = 10,
)->xr.Dataset:
    """
    Pearson correlation across paired initialization years 
    (for fcst & retro) at each grid cells
    """
    
    if min_years < 3:
        raise ValueError('Require at least three paired years')
    present=monthly['present_members'] == 1
    
    # Absent padded members are ignored; missing actual members invalidate the mean
    actual_finite=np.isfinite(monthly['fcst_fwi']) | ~present
    complete=actual_finite.all('ensemble') & (monthly['member_count'] > 0)
    vars = {}

    for code in CODES:
        fcst=(
            monthly[f'fcst_{code}'].where(present).sum('ensemble', skipna=True) 
            / monthly['member_count']
        ) #<-- ensemble avg of forecast
        fcst=fcst.where(complete)
        retro=monthly[f'retro_{code}']

        # Both series must use exactly the sam eyears for means and covariance
        paired=np.isfinite(fcst) & np.isfinite(retro)
        fcst, retro=fcst.where(paired), retro.where(paired)
        
        n=paired.sum('year').astype('int32')
        dx=fcst - fcst.mean(dim='year', skipna=True)
        dy=retro - retro.mean(dim='year', skipna=True)
        
        denominator=np.sqrt(
            (dx**2).sum(dim='year') * (dy**2).sum(dim='year')
        )
        corr= (
            (dx*dy).sum(dim='year') / denominator.where(denominator>0)
        ).where(n >= min_years)

        vars[f'pearson_r_{code}']=corr.clip(-1, 1)
        vars[f'paired_years_{code}']=n

    result=xr.Dataset(vars)
    result.attrs.update(monthly.attrs)
    result.attrs.update(
        correlation_method='',
        correlation_dim='year',
        min_paired_years=min_years,
        centering='Separate forecast/retrospective means over paired valid years',
    )
    return result
