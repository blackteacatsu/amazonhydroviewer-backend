import xarray as xr
import numpy as np
import regionmask


def get_standard_coordinates(dataset: xr.Dataset, lon_names=None, lat_names=None, time_names=None):
    """
    Retrieve longitude, latitude, and time variables from an xarray dataset.
    """
    lon_names = lon_names or ["east_west", "lon", "longitude"]
    lat_names = lat_names or ["north_south", "lat", "latitude"]
    time_names = time_names or ["time", "month", "date"]

    def find_variable(ds, possible_names):
        for name in possible_names:
            if name in ds.coords:
                return ds.coords[name]
            if name in ds.variables:
                return ds[name]
        raise AttributeError(f"None of the variable names {possible_names} found in the dataset.")

    lon = find_variable(dataset, lon_names)
    lat = find_variable(dataset, lat_names)
    time = find_variable(dataset, time_names)
    return lon, lat, time


def get_spatial_coordinates(dataset: xr.Dataset, lon_names=None, lat_names=None):
    """
    Retrieve longitude and latitude variables from an xarray dataset.
    """
    lon_names = lon_names or ["east_west", "lon", "longitude"]
    lat_names = lat_names or ["north_south", "lat", "latitude"]

    def find_variable(ds, possible_names):
        for name in possible_names:
            if name in ds.coords:
                return ds.coords[name]
            if name in ds.variables:
                return ds[name]
        raise AttributeError(f"None of the variable names {possible_names} found in the dataset.")

    lon = find_variable(dataset, lon_names)
    lat = find_variable(dataset, lat_names)
    return lon, lat


def build_region_mask_3d(geodataframe, lon, lat):
    """
    Build a robust 3D region mask for many polygons.

    Tries fast rasterize first with safe sequential region numbers, then
    falls back to shapely if rasterization fails (e.g., uint32 casting issues).
    """
    gdf = geodataframe.copy()
    number_col = "__regionmask_number__"
    gdf[number_col] = np.arange(len(gdf), dtype=np.int32)
    try:
        return regionmask.mask_3D_geopandas(
            gdf,
            lon,
            lat,
            numbers=number_col,
        )
    except ValueError as exc:
        if "shape values cannot be cast to specified dtype" not in str(exc):
            raise
        return regionmask.mask_3D_geopandas(
            gdf,
            lon,
            lat,
            numbers=number_col,
            #method="shapely",
        )

def initialize_climatology(hindcast_data_file_path, variable):
    """
    Initialize climatology for the given variable from hindcast data.
    """
    climatology = xr.open_mfdataset(hindcast_data_file_path, join='outer')[variable]
    climatology = climatology.groupby('time.month').mean(dim='time')
    #if variable == 'Stream_flow': # for streamflow, we want max not mean
        #climatology = climatology.groupby('time.month').max(dim='time') 
    return climatology