"""Build forecast & climatology tables aggregated by region"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import regionmask
import xarray as xr

from modules import utils
from tqdm import tqdm


REGION_ID_COL = "PFAF_ID"
REGION_DIM = "pfaf_id"
SPATIAL_DIMS = ("lat", "lon")
STREAMFLOW_VAR = "Streamflow_tavg"


def build_region_mask_3d(
        geodataframe : gpd.GeoDataFrame, 
        lon : xr.DataArray, 
        lat : xr.DataArray,
        region_id_column: str = REGION_ID_COL
) -> xr.DataArray:
    """
    Build a 3D region mask for multiple polygons.

    Tries fast rasterize first with safe sequential region numbers, then
    falls back to shapely if rasterization fails (e.g., uint32 casting issues).

    Args:
        geodataframe (geopandas.GeoDataFrame):
        lon (xarray.DataArray): longitude of the dataset as DataArray
        lat (xarray.DataArray):
    """
    if region_id_column not in geodataframe:
        raise KeyError(f'Regional ID {region_id_column} not found in DataFrame.')
    if geodataframe[region_id_column].duplicated().any():
        raise ValueError(f'Multiple entries of {region_id_column} found in DataFrame. It must be unique!')
    
    aoi = geodataframe.copy()
    number_column = "__regionmask_number__"
    aoi[number_column] = np.arange(len(aoi), dtype=np.int32)
    mask = regionmask.mask_3D_geopandas(
        aoi,
        lon,
        lat,
        numbers=number_column,
    )
    aoi_ids = (
        aoi.iloc[mask["region"].values][region_id_column]
        .to_numpy()
    )
    return mask.assign_coords(region = aoi_ids).rename(region = REGION_DIM)


def _aggregate_variable(
        data: xr.DataArray,
        region_mask: xr.DataArray,
        variable: str,
        average_ensemble: bool = False
) -> xr.DataArray:
    """
    Aggregate one variable over all regions in a vectorized xarray operation

    Args:
        data (xarray.DataArray):
        variable (str):

    Returns:
        xarray.DataArray
    """
    if average_ensemble and "ensemble" in data.dims:
        data = data.mean(dim="ensemble")

    masked = data.where(region_mask)
    if variable == STREAMFLOW_VAR:
        return masked.max(dim=SPATIAL_DIMS, skipna=True)
    return masked.mean(dim=SPATIAL_DIMS, skipna=True)


def _variable_frame(
        aggregated_ds : xr.DataArray,
        variable : str,
        row_dimensions: Sequence[str]
) -> pd.DataFrame:
    """
    Convert an aggregated data array of a variable to table columns, expanding profile levels horizontally.

    Args:
        aggregated_ds (xarray.DataArray): aggregated data array
        variable (str): 
    
    Returns:
        pandas.DataFrame
    """
    profile_dim = [
        dim for dim in aggregated_ds.dims
        if "profile" in dim.lower()
    ]

    if len(profile_dim) > 1:
        raise ValueError(f"Multiple profile dimensions found for {variable}")

    for dim in row_dimensions:
        if dim not in aggregated_ds.dims:
            aggregated_ds = aggregated_ds.expand_dims({dim: [None]})

    if profile_dim:
        profile_dim = profile_dim[0]
        frame = aggregated_ds.to_series().unstack(profile_dim)
        frame.columns = [
            f'{variable}_lvl_{level_idx}'
            for level_idx in range(len(frame.columns))
        ]
        frame = frame.reset_index()

    else:
        frame = aggregated_ds.to_series().rename(variable).reset_index()
    value_columns = [
        column for column in frame.columns if column not in aggregated_ds.dims
    ]
    return frame[[*row_dimensions, *value_columns]]


def _merge_variable_frames(
        frames: Sequence[pd.DataFrame],
        row_dimensions: Sequence[str],
) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame(columns=row_dimensions)

    table = frames[0]
    for frame in frames[1:]:
        table = table.merge(frame, on=list(row_dimensions), 
                            how="outer", validate="1:1")
    return table.sort_values(list(row_dimensions)).reset_index(drop=True)


def fcst_zonal_table(
        fcst_ds : xr.Dataset,
        geodataframe: gpd.GeoDataFrame,
        variables : Sequence[str],
) -> pd.DataFrame:
    """
    Build one forecast table containing every requested hydrological region.
    Args:
        fcst_ds (xarray.Dataset):
        geodataframe (geopandas.GeoDataFrame): AOI files opened as GeoDataFrame
        variables:
    
    Returns:
        pandas.Dataframe:
    """
    lon, lat, _ = utils.get_std_coords(fcst_ds)
    region_mask = build_region_mask_3d(geodataframe, lon, lat)
    row_dimensions = (REGION_DIM, "time", "ensemble")
    frames = []
    missing =[variable for variable in variables if variable not in fcst_ds]
    if missing:
        raise KeyError(
            f"Skipping forecast variable not found: {missing}"
        )
    for variable in variables:
        aggregated = _aggregate_variable(
            fcst_ds[variable], region_mask, variable, average_ensemble=False
        )
        frames.append(_variable_frame(aggregated, variable, row_dimensions))

    table = _merge_variable_frames(frames, row_dimensions)
    if "time" in table:
        table["time"] = pd.to_datetime(table["time"]).map(pd.Timestamp.isoformat)
    return table

 
def get_var_climatology(
        hindcast_files_path : Sequence[str | Path], 
        variable : str
) -> xr.DataArray:
    """
    Initialize climatology for the given variable from hindcast data.
    """
    hindcast = utils.read_trim_hcst(hindcast_files_path, variable)
    grouped = hindcast.groupby('time.month')

    # Handle Streamflow variable as a special case
    if variable == 'Streamflow_tavg': 
        return grouped.max(dim='time') 
    return grouped.mean(dim='time')


def climatology_zonal_table(
        hindcast_files_path : Sequence[str | Path],
        geodataframe: gpd.GeoDataFrame,
        variables: Sequence[str],
        *,
        cache_dir: Path | None = None,
        initialization_date: str | None = None
) -> pd.DataFrame:
    """
    Args:
        hindcast_files
    """
    row_dims = (REGION_DIM, "month")
    region_mask = None
    frames = []
    
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

    for variable in variables:
        climatology = get_var_climatology(hindcast_files_path, variable)
        if region_mask is None:
            lon, lat, _ = utils.get_std_coords(climatology)
            region_mask = build_region_mask_3d(geodataframe, lon, lat)

        if cache_dir is not None:
            if not initialization_date:
                raise ValueError("Initialization Date is required to process the data")
            cache_path = (
                cache_dir
                / f"climatology_{initialization_date}_{variable}"
            )
            if cache_path.exists():
                utils.purge_path(cache_path)
            climatology.to_zarr(cache_path, zarr_format=2, mode='w')
        aggregated = _aggregate_variable(
            climatology, region_mask, variable, average_ensemble=True
        )
        frames.append(_variable_frame(aggregated, variable, row_dims))

    return _merge_variable_frames(frames, row_dims)


def write_zonal_tables(
        table: pd.DataFrame,
        output_dir: Path,
        filename_prefix: str
) -> list[Path]:
    """
    Write one CSV per PFAF region at the created paths.

    Args:
        table (pd.DataFrame):
        output_dir (Path):
        filename_prefix (str): ex. "zonal_climatology" or "zonal_forecast"

    Returns:
        written (list): a list consists of output location for files written 
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for region_id, region_table in table.groupby(REGION_DIM, sort=True):
        output_path = (
            output_dir
            / f"{filename_prefix}_pfaf_{region_id}.csv"
        )
        region_table.to_csv(output_path, index=False)
        written.append(output_path)
    return written


def get_zonal_tables(
        variables : Mapping[str, str] | Sequence[str],
        fcst_file : Path,
        hdct_files : Sequence[str | Path],
        aoi_polygon_file : Path,
        zonal_avg_fcst_tab_dir : Path ,
        zonal_avg_climatology_tab_dir : Path,
        *,
        climatology_cache_dir : Path,
        init_date : str,
) -> tuple[list[Path], list[Path]]:
    """
    Executes the entire Section 3; tabular data for boxplot. 

    Args:
        variables (list):
        fcst_file (Path):
        hdct_file (Path):
    
    Returns:
        fcst_paths (list):
        climatology_paths (list):

    """
    variable_names = list(
        variables.keys() 
        if isinstance(variables, Mapping) else variables
    )
    geodataframe = gpd.read_file(aoi_polygon_file)

    with xr.open_dataset(fcst_file) as fcst:
        fcst_tab = fcst_zonal_table(
            fcst, geodataframe, variable_names
        )
    fcst_paths = write_zonal_tables(
        fcst_tab, zonal_avg_fcst_tab_dir, "zonal_fcst"
    )
    
    climatology_tab = climatology_zonal_table(
        hdct_files,
        geodataframe=geodataframe,
        variables=variable_names,
        cache_dir=climatology_cache_dir,
        initialization_date=init_date
    )
    climatology_paths = write_zonal_tables(
        climatology_tab, 
        zonal_avg_climatology_tab_dir, 
        "zonal_climatology"
    )
    print(
        f"Wrote {len(fcst_paths)} forecast and \n"
        f"{len(climatology_paths)} climatology region tables."
    )
    return fcst_paths, climatology_paths