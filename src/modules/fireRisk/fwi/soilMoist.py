"""Yang et al. (2015) soil-moisture DC, adapted to the LDAS 0–40 cm layer."""
from __future__ import annotations

import numpy as np
import xarray as xr


def volumetric(da: xr.DataArray) -> xr.DataArray:
    """
    Reject ambiguous units instead of 
    silently treating water mass as volume.
    """

    unit = da.attrs.get('units', '').lower().replace(' ', '').replace('**', '^')

    if unit not in {'m3/m3', 'm^3/m^3', 'm3m-3', 
                    'm^3m-3', 'm^3m^-3', '1'}:
        raise ValueError(f'{da.name}: expected volumetric soil moisture; got {unit!r}')
    return da.astype('float64')


def avg_soilMoist_wDepth(da: xr.DataArray) -> xr.DataArray:
    """
    Thickness-weight the first two LDAS layers; 
    require both to be valid.
    """

    if da.sizes.get('SoilMoist_profiles') != 4:
        raise ValueError(
            'Require four LDAS SoilMoist_profiles '
            '(0–10, 10–40, 40–100, 100–200 cm)'
        )
    
    with volumetric(da) as da:
        upper = da.isel(SoilMoist_profiles=0, drop=True)
        lower = da.isel(SoilMoist_profiles=1, drop=True)
    valid = (
        np.isfinite(upper) & np.isfinite(lower) & 
        (upper > 0) & (lower > 0) & 
        (upper <= 1) & (lower <= 1)
    )
    return ((upper + 3 * lower) / 4).where(valid).assign_attrs(units='m3 m-3')

def above_saturation(sm : xr.DataArray, sat_sm : xr.DataArray)-> xr.DataArray:
    """
    Native MAXSMC is in float32. Allow the next representable float32 value
    """
    limit =np.nextafter(sat_sm.astype("float32"), np.float32(np.inf))
    return np.isfinite(sm) & np.isfinite(sat_sm) & (sm > limit)

def get_drought_code(
        sm: xr.DataArray, 
        sat_sm: xr.DataArray
) -> xr.DataArray:
    """Equation 6: mask inconsistent saturation rather than clip negative DC."""

    sm, sat_sm = xr.align(sm, sat_sm, join='exact')
    
    valid = (np.isfinite(sm) & np.isfinite(sat_sm) & (sm > 0)
             & (sat_sm > 0) & (sat_sm <= 1) & (sm <= sat_sm))

    valid = (
        np.isfinite(sm) & np.isfinite(sat_sm)
        & (sm > 0) & (sat_sm > 0 ) & (sat_sm <= 1)
        & ~above_saturation(sm, sat_sm)
    )
    # Treat accepted roundoff above saturation as saturated (DC = 0).
    sm_for_dc = np.minimum(sm, sat_sm).where(valid)
    return (400 * np.log(sat_sm.where(valid) / sm_for_dc)).rename('dc')


def load_saturation(path, day: xr.Dataset) -> xr.DataArray:
    """Accept only a saturated sm field on this exact LDAS grid."""

    with xr.open_dataset(path) as ds:
        sat = volumetric(ds['MAXSMC'].transpose("lat", "lon")).load()
        template = day['Tair_f_tavg']
        # Saturation is hared spatial property; keeping its hash member-independent.
        if 'ensemble' in template.dims:
            template = template.isel(ensemble=0, drop=True)
        template=template.transpose('north_south', 'east_west')
        if sat.shape != template.shape:
            raise ValueError('Saturation and LDAS grid differ')    
    
    # Latitude varies by row; longitude varies by column.
    for name, axis in (
        ("lat", sat.lat.values[:, None]),
        ("lon", sat.lon.values[None, :])
    ):
        if name not in day:
            raise ValueError(f"LDAS file is missing {name}")

        actual = day[name].transpose(*template.dims).values
        expected = np.broadcast_to(axis, actual.shape)
        valid = np.isfinite(actual) # LDAS coordinates/data are missing off land.

        if not valid.any() or not np.allclose(
            actual[valid], expected[valid], rtol=0, atol=1e-4
        ):
            raise ValueError(f"MAXSMC and LDAS {name} coordinates differ")

    sat = xr.DataArray(
        sat.data, dims=template.dims, coords=template.coords,
        name="SM_sat", attrs={
            "long_name" : f"{sat.attrs['long_name']}",
            "units" : "m3 m-3"
        }
    )
    return sat


# def preprocess_saturation(
#         paths, 
#         template: xr.Dataset, 
#         scale: float = 0.0001
# ) -> xr.Dataset:
#     """
#     Check 
#     """

#     lat, lon = template.lat, template.lon
    
#     if lat.ndim != 1 or lon.ndim != 1 :
#         raise ValueError('Require LIS 1-D lat/lon arrays')
    
#     spatial_dims = (lat.dims[0], lon.dims[0])
#     ys = np.asanyarray(lat.values, dtype='float64')
#     xs = np.asanyarray(lon.values, dtype="float64")
#     shape = (len(ys), len(xs))

#     if min(len(xs), len(ys)) < 2:
#         raise ValueError('Grid must have at least two rows and columns')
    
#     # Endpoint spacing avoids accumulating float32 coordinate rounding errors.
#     dx = float((xs[-1] - xs[0]) / (len(xs) - 1))
#     dy = float((ys[-1] - ys[0]) / (len(ys) - 1))

#     # Allow float32 coordinate rounding while requiring regular spacing 
#     if (
#         not np.isfinite(xs).all()
#         or not np.isfinite(ys).all()
#         or dx == 0
#         or dy == 0
# 	or not (np.all(np.diff(xs) > 0) or np.all(np.diff(xs) < 0))
# 	or not (np.all(np.diff(ys) > 0) or np.all(np.diff(ys) < 0))
#         or not np.allclose(np.diff(xs), dx, atol=1e-4, rtol=0)
#         or not np.allclose(np.diff(ys), dy, atol=1e-4, rtol=0)
#     ):
#         raise ValueError('Require a regular geographic LDAS grid')
    
#     total = np.zeros(shape, dtype='float64')

#     with xr.open_dataset(path) as source:
#         if source.attrs['grid'] is None or source.count != 1:
#             raise ValueError(f'{path}: require a single-band georeferenced raster')
        
#         # Invalid/missing depth layers propagate instead of changing depth weights.
#         layer = destination * scale
#         layer[(layer <= 0) | (layer > 1) | ~np.isfinite(layer)] = np.nan
#         total += weight * layer / 40
    
#     coords = {
#         dim: template.coords[dim] 
#         for dim in spatial_dims
#         if dim in template.coords
#     }

#     result = xr.Dataset(
#         {
#             'SM_sat': xr.DataArray(
#                 total, 
#                 dims=spatial_dims, 
#                 coords=coords,
#                 attrs={
#                     'units': 'm3 m-3', 
#                     'long_name': '0–40 cm saturated volumetric soil moisture'
#                 }
#             )
#         }
#     )
    
#     result['lat'], result['lon'] = lat, lon
    
#     result.attrs.update(
#         soil_depth_cm='', 
#         source='',
#         source_scale=scale
#     )

#     return result
