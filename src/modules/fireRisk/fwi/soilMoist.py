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
    Thickness-weight the first two 
    LDAS layers; require both to be valid.
    """
    if da.sizes.get('SoilMoist_profiles') != 4:
        raise ValueError('Require four LDAS SoilMoist_profiles (0–10, 10–40, 40–100, 100–200 cm)')
    
    with volumetric(da) as da:
        upper = da.isel(SoilMoist_profiles=0, drop=True)
        lower = da.isel(SoilMoist_profiles=1, drop=True)
    valid = np.isfinite(upper) & np.isfinite(lower) & (upper > 0) & (lower > 0) & (upper <= 1) & (lower <= 1)
    return ((upper + 3 * lower) / 4).where(valid).assign_attrs(units='m3 m-3')


def get_drought_code(
        sm: xr.DataArray, 
        saturation: xr.DataArray
) -> xr.DataArray:
    """Equation 6: mask inconsistent saturation rather than clip negative DC."""
    sm, saturation = xr.align(sm, saturation, join='exact')
    valid = (np.isfinite(sm) & np.isfinite(saturation) & (sm > 0)
             & (saturation > 0) & (saturation <= 1) & (sm <= saturation))
    return (400 * np.log(saturation.where(valid) / sm.where(valid))).rename('dc')


def load_saturation(path, day: xr.Dataset) -> xr.DataArray:
    """Accept only a preprocessed 0–40 cm field on this exact LDAS grid."""
    with xr.open_dataset(path) as ds:
        if ds.attrs.get('soil_depth_cm') != '0-40':
            raise ValueError('Saturation file must declare soil_depth_cm="0-40"')
        sat = volumetric(ds['SM_sat']).load()
        for name in ('lat', 'lon'):
            if name not in ds or name not in day:
                raise ValueError('Saturation and LDAS files must include lat/lon')
            xr.testing.assert_equal(ds[name].reset_coords(drop=True), day[name].reset_coords(drop=True))
    template = day.Tair_f_tavg
    if dict(sat.sizes) != dict(template.sizes):
        raise ValueError('Saturation dimensions differ from LDAS')
    xr.align(sat, template, join='exact')
    if bool((np.isinf(sat) | (sat <= 0) | (sat > 1)).any()):
        raise ValueError('Finite saturation values must be in (0, 1]')
    if not bool(np.isfinite(sat).any()):
        raise ValueError('Saturation has no valid cells')
    return sat.reset_coords(drop=True)


def preprocess_saturation(
        paths, 
        template: xr.Dataset, 
        scale: float = 0.0001
) -> xr.Dataset:
    """Stream four GeoTIFFs onto a regular LDAS grid, then depth-average.

    Rasterio reads source bands in windows, avoiding global arrays in memory.
    Input order is 0–5, 5–15, 15–30, 30–60 cm; the last contributes 10 cm.
    """
    import rasterio
    from rasterio.transform import Affine
    from rasterio.warp import reproject, Resampling

    if len(paths) != 4 or not np.isfinite(scale) or scale <= 0:
        raise ValueError('Require four WCsat rasters and a positive scale')
    lat, lon = template.lat, template.lon
    
    if lat.ndim != 1 or lon.ndim != 1 :
        raise ValueError('Require LIS 1-D lat/lon arrays')
    
    spatial_dims = (lat.dims[0], lon.dims[0])
    ys = np.asanyarray(lat.values, dtype='float64')
    xs = np.asanyarray(lon.values, dtype="float64")
    shape = (len(ys), len(xs))

    if min(len(xs), len(ys)) < 2:
        raise ValueError('Grid must have at least two rows and columns')
    
    # Endpoint spacing avoids accumulating float32 coordinate rounding errors.
    dx = float((xs[-1] - xs[0]) / (len(xs) - 1))
    dy = float((ys[-1] - ys[0]) / (len(ys) - 1))

    # Allow float32 coordinate rounding while requiring regular spacing 
    if (
        not np.isfinite(xs).all()
        or not np.isfinite(ys).all()
        or dx == 0
        or dy == 0
        or not np.allclose(np.diff(xs), dx, atol=1e-5, rtol=0)
        or not np.allclose(np.diff(ys), dy, atol=1e-5, rtol=0)
    ):
        raise ValueError('Require a regular geographic LDAS grid')
    
    transform = Affine(dx, 0, float(xs[0]) - dx / 2, 0, dy, float(ys[0]) - dy / 2)

    total = np.zeros(lat.shape, dtype='float64')

    for path, weight in zip(paths, (5, 10, 15, 10)):
        destination = np.full(shape, dtype='float64')

        with rasterio.open(path) as source:
            if source.crs is None or source.count != 1:
                raise ValueError(f'{path}: require a single-band georeferenced raster')
            
            reproject(
                rasterio.band(source, 1),
                destination,
                src_nodata=source.nodata, 
                dst_transform=transform,
                dst_crs='EPSG:4326', 
                dst_nodata=np.nan,
                resampling=Resampling.average, 
                warp_mem_limit=128
            )
        
        # Invalid/missing depth layers propagate instead of changing depth weights.
        layer = destination * scale
        layer[(layer <= 0) | (layer > 1) | ~np.isfinite(layer)] = np.nan
        total += weight * layer / 40
    
    coords = {
        dim: template.coords[dim] 
        for dim in spatial_dims
        if dim in template.coords
    }

    result = xr.Dataset(
        {
            'SM_sat': xr.DataArray(
                total, 
                dims=spatial_dims, 
                coords=coords,
                attrs={
                    'units': 'm3 m-3', 
                    'long_name': '0–40 cm saturated volumetric soil moisture'
                }
            )
        }
    )
    
    result['lat'], result['lon'] = lat, lon
    
    result.attrs.update(
        soil_depth_cm='0-40', 
        source='HiHydroSoil v2.0 WCsat',
        resampling='average of valid contributing pixels', source_scale=scale
    )

    return result
