"""Shared file, dataset, and coordinate helpers for LDAS workflows."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
import shutil

import xarray as xr

# Month tags used in naming LIS runs.
_MONTHS = {
    month: index + 1
    for index, month in enumerate(
        [
            "jan", "feb", "mar", "apr", "may", "jun",
            "jul", "aug", "sep", "oct", "nov", "dec",
        ]
    )
}
_FCST_NAME_PATTERNS = (
    re.compile(r"^ldas_fcst_(\d{4})_([a-z]{3})(\d{2})\.nc$", re.I),
    re.compile(r"^ldas_fcst_(\d{4})(\d{2})(\d{2})\.nc$", re.I),
)


@dataclass(frozen=True)
class WorkingConditions:
    """Input files and output directories used by one updater run."""

    cwd: Path
    fcst_file : Path
    hcst_files : tuple[Path, ...]
    fcst_date : datetime
    init_date : str
    prob_fcst_dir : Path 
    prob_fcst_cache_dir : Path
    prob_fcst_subsampled_dir : Path
    zonal_avg_fcst_tab_dir : Path
    zonal_avg_climatology_dir : Path
    climatology_cache_dir : Path
    zonal_avg_climatology_tab_dir : Path


def _parse_date_from_name(name: str) -> datetime | None:
    """Parse an initialization date from a supported LDAS fcst filename."""
    for index, pattern in enumerate(_FCST_NAME_PATTERNS):
        match = pattern.match(name)
        if not match:
            continue

        if index == 0:
            year = int(match.group(1))
            month = _MONTHS.get(match.group(2).lower())
            day = int(match.group(3))
        else:
            year, month, day = map(int, match.groups())

        if month is None:
            return None
        try:
            return datetime(year, month, day)
        except ValueError:
            return None
    return None


def split_fcst_hcst(
    dir_path: str | Path,
    hcst_start_year : int = 2001,
    hcst_end_year: int = 2020, # CLI args 
    fcst_init_date: datetime | None = None, 
    prefix: str = "ldas_fcst_",
    recursive: bool = False,
) -> tuple[str, list[str], datetime]:
    """
    Return a selected fcst and same-month hindcasts through a cutoff year.

    If ``fcst_init_date`` is omitted, the latest fcst is selected. If it is
    provided, its year and month select a specific initialization month.
    """
    base = Path(dir_path)
    if not base.is_dir():
        raise NotADirectoryError(f"Not a directory: {dir_path}")

    glob_pattern = "**/*.nc" if recursive else "*.nc"
    items: list[tuple[datetime, float, str, Path]] = []
    for path in base.glob(glob_pattern):
        if not path.is_file() or not path.name.startswith(prefix):
            continue
        initialization = _parse_date_from_name(path.name)
        if initialization is not None:
            items.append(
                (initialization, path.stat().st_mtime, path.name, path)
            )

    if not items:
        raise FileNotFoundError(
            f"No matching .nc files found in {dir_path} (prefix='{prefix}')"
        )

    items.sort(key=lambda item: (item[0], item[1], item[2]))

    # The program will select the latest initialization as targeted forecast
    if fcst_init_date is None:
        fcst_date, _, _, fcst_path = items[-1]

    # A special date is requested, over-ride latest initialization
    else:
        requested_items = [
            item
            for item in items
            if item[0].year == fcst_init_date.year
            and item[0].month == fcst_init_date.month
        ]
        if not requested_items:
            available_months = sorted(
                {item[0].strftime("%Y-%m") for item in items},
                reverse=True,
            )
            raise FileNotFoundError(
                "No fcst initialization found for "
                f"{fcst_init_date:%Y-%m} in {dir_path}. Available months: "
                + ", ".join(available_months[:12])
            )
        fcst_date, _, _, fcst_path = requested_items[-1]

    hcsts = [
        path
        for initialization, _, _, path in items
        if hcst_start_year <= initialization.year <= hcst_end_year
        and initialization.month == fcst_date.month
        and initialization.day == 1
        and initialization < fcst_date
    ]
    hcsts.sort(key=lambda path: _parse_date_from_name(path.name))

    return str(fcst_path), [str(path) for path in hcsts], fcst_date


def read_trim_fcst(file_path: str | Path, variable: str) -> xr.DataArray:
    """Open one fcst variable from a NetCDF dataset."""
    try:
        return xr.open_dataset(file_path)[variable]
    except KeyError:
        print(f"ERROR: Variable {variable} not found in dataset {file_path}")
        raise


def read_trim_hcst(
    file_path: str | Path | Sequence[str | Path],
    variable: str,
) -> xr.DataArray:
    """Open one hindcast variable and rechunk its reduction dimensions."""
    try:
        hcst = xr.open_mfdataset(file_path, join="outer")[variable]
    except KeyError:
        print(f"ERROR: Variable {variable} not found in hindcast dataset")
        raise

    chunk_dims = {
        dimension: -1
        for dimension in ("time", "ensemble")
        if dimension in hcst.dims
    }
    return hcst.chunk(chunk_dims) if chunk_dims else hcst


def _find_variable(
    dataset: xr.Dataset | xr.DataArray,
    possible_names: Sequence[str],
) -> xr.DataArray:
    for name in possible_names:
        if name in dataset.coords:
            return dataset.coords[name]
        if isinstance(dataset, xr.Dataset) and name in dataset.variables:
            return dataset[name]
    raise AttributeError(
        f"None of the variable names {list(possible_names)} found in the dataset."
    )


def get_std_coords(
    dataset: xr.Dataset | xr.DataArray,
    lon_names: Sequence[str] | None = None,
    lat_names: Sequence[str] | None = None,
    time_names: Sequence[str] | None = None,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """Return longitude, latitude, and time values under common aliases."""
    lon = _find_variable(dataset, lon_names or ("east_west", "lon", "longitude"))
    lat = _find_variable(dataset, lat_names or ("north_south", "lat", "latitude"))
    time = _find_variable(dataset, time_names or ("time", "month", "date"))
    return lon, lat, time


def purge_path(path: str | Path) -> None:
    """Remove one file or directory tree if it exists."""
    target = Path(path)
    if target.is_dir():
        shutil.rmtree(target)
    elif target.exists():
        target.unlink()


def purge_old_init(directory: str | Path, current_init: str) -> None:
    """Remove non-JSON entries that do not belong to the current initialization."""
    for entry in Path(directory).iterdir():
        if entry.suffix == ".json":
            continue
        if current_init not in entry.name:
            purge_path(entry)
            print(f"Deleted (old init): {entry}")


def initialize_working_cond(
    cwd : str | Path,
    surface_model_dir : str | Path,
    hcst_start_year : int = 2001, 
    hcst_end_year : int = 2020,
    fcst_init_date : datetime | None = None,
) -> WorkingConditions:
    """Discover model inputs and prepare output directories for an updater run.

    Git operations intentionally remain in ``updater.py``. This helper only manages
    filesystem state and returns the paths the update workflow needs.
    """
    working_directory = Path(cwd).expanduser().resolve()
    fcst, hcst, fcst_date = split_fcst_hcst(
        Path(surface_model_dir).expanduser(),
        hcst_start_year,
        hcst_end_year,
        fcst_init_date,
    )
    if not hcst:
        raise FileNotFoundError(
            "No hindcast files matched the requested year range "
            f" {hcst_start_year}-{hcst_end_year} and forecast month. "
        )

    init_date = (
        f"{fcst_date.year}_{fcst_date.strftime('%b').lower()}"
    )

    # probabilistic forecast output directory(s)
    prob_fcst_dir = working_directory / "get_ldas_probabilistic_output"
    prob_fcst_cache_dir = prob_fcst_dir / "tmp" # tmp zarr files
    prob_fcst_subsampled_dir = prob_fcst_dir / "subsampled" # final sub-sampled output

    # zonal averaged forecast tabular directory
    zonal_avg_fcst_tab_dir = working_directory / "get_zonal_averages_fcst"

    # zonal averaged climatology cache & tabular directory
    zonal_avg_climatology_dir = (
        working_directory / "get_zonal_averages_climatology"
    )
    climatology_cache_dir = zonal_avg_climatology_dir / "tmp"
    zonal_avg_climatology_tab_dir = zonal_avg_climatology_dir / "zmean"

    for directory in (
        prob_fcst_cache_dir,
        prob_fcst_subsampled_dir,
        zonal_avg_fcst_tab_dir,
        climatology_cache_dir,
        zonal_avg_climatology_tab_dir,
    ):
        directory.mkdir(exist_ok=True, parents=True)

    purge_old_init(prob_fcst_cache_dir, current_init=init_date)
    purge_old_init(prob_fcst_subsampled_dir, current_init=init_date)
    purge_old_init(climatology_cache_dir, current_init=init_date)

    conditions = WorkingConditions(
        cwd=working_directory,
        fcst_file=Path(fcst),
        hcst_files=tuple(Path(path) for path in hcst),
        fcst_date=fcst_date,
        init_date=init_date,
        prob_fcst_dir=prob_fcst_dir,
        prob_fcst_cache_dir=prob_fcst_cache_dir,
        prob_fcst_subsampled_dir=prob_fcst_subsampled_dir,
        zonal_avg_fcst_tab_dir=zonal_avg_fcst_tab_dir,
        zonal_avg_climatology_dir=zonal_avg_climatology_dir,
        climatology_cache_dir=climatology_cache_dir,
        zonal_avg_climatology_tab_dir=zonal_avg_climatology_tab_dir,
    )

    print("Found latest forecast file:", conditions.fcst_file)
    print("Hindcasts   :", len(conditions.hcst_files), "files")
    print("Forecast initialization date:", conditions.init_date)
    print("Output directory:", conditions.prob_fcst_dir)
    print("Subsampled directory:", conditions.prob_fcst_subsampled_dir)
    return conditions
